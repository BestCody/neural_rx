// The channel filter of the proposed NRX as CUDA kernels (FP32). They do what ChannelFilter.__call__ and
// PilotLayout.expand_channel do in TensorFlow (src/neural_rx/receiver/channel_filter.py); filter_kernels.py launches
// them and time_fused.py checks their output against the TensorFlow receiver.
//
// Before the slot arrives (only the memory is needed):
//   filter_gram     normal equations of each of the newest entries, in parts: one block per (UE, DMRS symbol, entry,
//                   slice of the symbol's cells) loads the slice's inputs and targets into shared memory, each thread
//                   accumulates its elements of the Gram matrix and of the right-hand side, and adds them to the
//                   entry's normal equations, which the host zeroes before (FilterKernels.fit)
//   filter_solve    one block per (UE, DMRS symbol, number K of entries used): sums the normal equations of the
//                   stored TBs that have all K lagged entries, adds the ridge term, and solves by Gauss-Jordan
//                   elimination, in registers of one warp up to 31 unknowns
//   filter_history  one block per (UE, pilot cell): the part of every K's prediction that comes from the entries
// On the slot (timed with the network):
//   filter_predict  one block per (UE, DMRS symbol, group of the symbol's cells), one thread per (cell, antenna): adds
//                   the part from the current TB's estimates around the cell, and sums over the group the squared
//                   difference of every K's prediction to the cells' own LS estimates, and the LS noise variance
//   filter_expand   one thread per (UE, grid position). Every block first adds the groups' sums of its UE, takes for
//                   each DMRS symbol the K with the smallest mean squared difference, and from it the weight of the
//                   prediction against the LS estimate; then the first-order model of the position's cell from the
//                   refined estimates of the cell and of its neighbors in time and frequency
//
// Lookups that only depend on the pilot layout are tables prepared on the host (own_neighbors, position_cells,
// position_packs), so a kernel reaches every estimate with at most two dependent memory reads.
//
// All sizes are fields of the context, which the kernels receive by value; every field is 8 bytes, so the host fills
// it as an array of 64-bit words. Limits the host checks: history <= 16, neighbors <= 8, inputs+1 <= solve threads,
// pairs+inputs <= ACCUMULATORS * gram threads.

#define ACCUMULATORS 4
#define MAX_RECORDS 16
#define MAX_NEIGHBORS 8
#define WARP 32

struct Context {
    const float* ls;                    // [U, P, 2B] LS estimate on the grid (nearest-pilot interpolation)
    const float* noise;                 // [1] noise variance
    const float* memory;                // [U, H, C, 2B+1] memory entries, newest first
    const int* age;                     // [U, H] entry age in TBs
    const int* ports;                   // [U, H] DMRS port of each entry
    const float* valid;                 // [U, H] entry validity (0/1)
    const float* active;                // [U] scheduled UEs (0/1)
    const int* pack;                    // [U, C] grid position of each pilot cell's estimate
    const float* noise_factors;         // [U, C] LS error variance per unit noise
    const int* neighbor_index;          // [U, U, C, N] cells of the neighbors, per port of the centre and of the record
    const unsigned char* neighbor_valid;// [U, U, C, N]
    const float* neighbor_delta;        // [U, U, C, N, 2] offset in time and frequency
    const int* cell_symbol;             // [U, C] DMRS symbol of each cell
    const int* symbol_cells;            // [U, S, C/S] cells of each DMRS symbol
    const int* own_neighbors;           // [U, C, N] grid position of each neighbor in the current TB, -1 if none or the
                                        // cell itself
    const int* position_cells;          // [U, P, 4] the position's cell, its cell on the other DMRS symbol, its lower
                                        // and upper neighbor in frequency
    const int* position_packs;          // [U, P, 4] grid positions of these cells' LS estimates
    const float* time_factor;           // [U, P] offset of the position, in units of the difference to the other symbol
    const float* frequency_factor;      // [U, P] the same for the difference between the frequency neighbors
    const int* pair_row;                // [pairs] upper triangle of the Gram matrix
    const int* pair_column;             // [pairs]
    float2* gram;                       // [U, S, R, pairs] normal equations, added up from parts (zeroed by the host)
    float2* rhs;                        // [U, S, R, M]
    float2* weights;                    // [U, S, E+1, M] zero beyond the inputs a K uses
    int* available;                     // [U, S, E+1] whether a K has a fit
    float* memory_part;                 // [U, C, E+1, 2B] part of each K's prediction that comes from the entries
    float* prediction;                  // [U, C, E+1, 2B]
    float* sums;                        // [U, S, G, E+2] per group of cells: squared difference to the own LS estimates
                                        // per K, summed over cells and antennas; last, the LS noise variance summed
    int* choice;                        // [U, S] chosen K
    float* gain;                        // [U, S] weight of the prediction; 0 without a fit
    float* out;                         // [U, P, 2B]
    long long users, cells, positions, antennas, history, neighbors, symbols, symbol_cells_count, entries, fit_tbs, pairs,
              groups, splits, chunk;
    double ridge;
};

__device__ __forceinline__ float2 multiply(float2 a, float2 b) { return make_float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
__device__ __forceinline__ float2 conjugate_multiply(float2 a, float2 b) {     // conj(a)*b
    return make_float2(a.x*b.x+a.y*b.y, a.x*b.y-a.y*b.x);
}
__device__ __forceinline__ float2 divide(float2 a, float2 b) {
    const float d = b.x*b.x+b.y*b.y;
    return make_float2((a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d);
}

// A UE's memory metadata in shared memory: for each entry its age, and its DMRS port if the entry is valid and on a
// known port (else -1). load() is called by all threads of a block, followed by __syncthreads().
struct Entries {
    int age[MAX_RECORDS], port[MAX_RECORDS];
    __device__ void load(const Context& x, int u) {
        for (int j = threadIdx.y*blockDim.x+threadIdx.x; j < x.history; j += blockDim.x*blockDim.y) {
            const int p = x.ports[u*x.history+j];
            age[j] = x.age[u*x.history+j];
            port[j] = x.valid[u*x.history+j] > .5f && p >= 0 && p < x.users ? p : -1;
        }
    }
    // The entry that is `lag` TBs older than entry `target`, or -1.
    __device__ int lagged(int history, int target, int lag) const {
        for (int j = 0; j < history; ++j)
            if (port[j] >= 0 && age[j] == age[target]+lag) return j;
        return -1;
    }
    // Record k (0: the target itself, k >= 1: the entry as much older than it as the current TB's kth newest entry is
    // older than the current TB) of the normal equations of entry `target`; -1 if it does not exist.
    __device__ int fit_record(int history, int target, int k) const {
        if (k == 0) return target;
        return port[k-1] >= 0 ? lagged(history, target, age[k-1]) : -1;
    }
};

extern "C" __global__ void filter_gram(const Context x) {
    // grid: (splits, UEs x DMRS symbols x newest entries); block: any number of threads
    const int inputs = (x.entries+1)*x.neighbors, antennas = x.antennas, width = 2*antennas;
    const int threads = blockDim.x, tid = threadIdx.x, numbers = x.entries+1, columns = inputs+1;
    const int split = blockIdx.x, r = blockIdx.y % x.fit_tbs, s = (blockIdx.y/x.fit_tbs) % x.symbols;
    const int u = blockIdx.y/(x.fit_tbs*x.symbols);
    __shared__ Entries memory;
    __shared__ int record[MAX_RECORDS], record_port[MAX_RECORDS], centre;
    extern __shared__ float2 slice[];                            // [rows of a chunk, inputs+1], the target last
    memory.load(x, u);
    __syncthreads();
    if (tid < numbers) {
        const int target = x.active[u] > .5f ? memory.port[r] : -1;
        const int entry = target >= 0 ? memory.fit_record(x.history, r, tid) : -1;
        record_port[tid] = entry >= 0 ? memory.port[entry] : -1;
        record[tid] = record_port[tid] >= 0 ? entry : -1;
        if (tid == 0) centre = target;
    }
    __syncthreads();
    // the elements this thread accumulates: the upper triangle of the Gram matrix, then the right-hand side
    const int elements = x.pairs+inputs;
    int left[ACCUMULATORS], right[ACCUMULATORS];
    float2 total[ACCUMULATORS];
    for (int a = 0; a < ACCUMULATORS; ++a) {
        const int e = tid+a*threads;
        left[a] = e < x.pairs ? x.pair_row[e] : e < elements ? e-x.pairs : -1;
        right[a] = e < x.pairs ? x.pair_column[e] : inputs;
        total[a] = make_float2(0.f, 0.f);
    }
    const int count = x.symbol_cells_count, first = (int)((long)count*split/x.splits);
    const int last = (int)((long)count*(split+1)/x.splits);
    const int chunk = (int)x.chunk;                              // cells per chunk (the host sizes shared memory)
    if (centre >= 0) {
        const int* cells = x.symbol_cells+((long)centre*x.symbols+s)*count;
        for (int start = first; start < last; start += chunk) {
            const int held = min(chunk, last-start);
            for (int item = tid; item < held*columns; item += threads) {
                const int local = item/columns, i = item % columns, c = cells[start+local];
                const float* source = nullptr;
                if (i == inputs) source = x.memory+(((long)u*x.history+r)*x.cells+c)*(width+1);
                else {
                    const int k = i/x.neighbors, n = i % x.neighbors;
                    if (record[k] >= 0) {
                        const long base = (((long)centre*x.users+record_port[k])*x.cells+c)*x.neighbors+n;
                        const bool own = k == 0 && x.neighbor_delta[2*base] == 0.f && x.neighbor_delta[2*base+1] == 0.f;
                        if (x.neighbor_valid[base] != 0 && !own)
                            source = x.memory+(((long)u*x.history+record[k])*x.cells+x.neighbor_index[base])*(width+1);
                    }
                }
                for (int v = 0; v < antennas; ++v)
                    slice[(local*antennas+v)*columns+i] = source ? make_float2(source[v], source[antennas+v])
                                                                 : make_float2(0.f, 0.f);
            }
            __syncthreads();
            for (int a = 0; a < ACCUMULATORS; ++a) {
                if (left[a] < 0) continue;
                float2 sum = total[a];
                for (int row = 0; row < held*antennas; ++row) {
                    const float2 p = conjugate_multiply(slice[row*columns+left[a]], slice[row*columns+right[a]]);
                    sum.x += p.x; sum.y += p.y;
                }
                total[a] = sum;
            }
            __syncthreads();
        }
    }
    if (centre < 0) return;
    const long target = ((long)u*x.symbols+s)*x.fit_tbs+r;      // the parts are added up (the host zeroes them first)
    for (int a = 0; a < ACCUMULATORS; ++a) {
        const int e = tid+a*threads;
        float2* sum = e < x.pairs ? x.gram+target*x.pairs+e : e < elements ? x.rhs+target*inputs+e-x.pairs : nullptr;
        if (sum) { atomicAdd(&sum->x, total[a].x); atomicAdd(&sum->y, total[a].y); }
    }
}

extern "C" __global__ void filter_solve(const Context x) {
    // grid: (numbers of entries used, UEs x DMRS symbols); block: (columns, rows) of at least inputs+1 threads
    const int inputs = (x.entries+1)*x.neighbors, tid = threadIdx.y*blockDim.x+threadIdx.x, threads = blockDim.x*blockDim.y;
    const int number = blockIdx.x, s = blockIdx.y % x.symbols, u = blockIdx.y/x.symbols;
    const int size = (number+1)*x.neighbors, columns = size+1;
    extern __shared__ float2 shared[];
    float2* system = shared;                                     // [size, size+1], the right-hand side last
    float2* row = system+inputs*(inputs+1);                      // [size+1]
    float2* column = row+inputs+1;                               // [size]
    __shared__ Entries memory;
    __shared__ bool complete[MAX_RECORDS], fitted;
    memory.load(x, u);
    __syncthreads();
    // entries whose normal equations have all of these lags, and whether the current TB has these entries
    if (tid < x.fit_tbs) {
        bool all = x.active[u] > .5f && memory.port[tid] >= 0;
        for (int k = 1; k <= number && all; ++k) all = memory.fit_record(x.history, tid, k) >= 0;
        complete[tid] = all;
    }
    __syncthreads();
    if (tid == 0) {
        bool any = false;
        for (int r = 0; r < x.fit_tbs; ++r) any = any || complete[r];
        for (int k = 0; k < number; ++k) any = any && memory.port[k] >= 0;
        fitted = any;
    }
    __syncthreads();
    for (int e = tid; e < size*columns; e += threads) {
        const int i = e/columns, j = e % columns;
        const int a = i < j ? i : j, b = i < j ? j : i;          // stored: row a <= column b
        const long pair = (long)a*inputs-(long)a*(a-1)/2+(b-a);
        float2 value = make_float2(0.f, 0.f);
        for (int r = 0; r < x.fit_tbs; ++r) {
            if (!complete[r]) continue;
            const long block = ((long)u*x.symbols+s)*x.fit_tbs+r;
            const float2 term = j == size ? x.rhs[block*inputs+i] : x.gram[block*x.pairs+pair];
            value.x += term.x; value.y += j != size && i > j ? -term.y : term.y;
        }
        system[e] = value;
    }
    __syncthreads();
    if (tid == 0) {                                              // ridge term, or the identity if there is nothing to fit
        float trace = 0.f;
        for (int i = 0; i < size; ++i) trace += system[i*columns+i].x;
        const float loading = (float)x.ridge*trace/size+(trace <= 0.f ? 1.f : 0.f);
        for (int i = 0; i < size; ++i) system[i*columns+i].x += loading;
    }
    __syncthreads();
    // Gauss-Jordan; positive definite, so no pivoting. Up to 31 unknowns, one warp keeps the system in registers, a
    // column per lane, and passes the pivot column by shuffles; larger systems use shared memory.
    if (columns <= WARP) {
        if (tid < WARP) {
            const int j = tid;
            float2 a[WARP];                                      // column j of the system
            #pragma unroll
            for (int i = 0; i < WARP; ++i) a[i] = i < size && j < columns ? system[i*columns+j] : make_float2(0.f, 0.f);
            for (int k = 0; k < size; ++k) {
                float2 mine = make_float2(0.f, 0.f);              // this lane's element in row k
                #pragma unroll
                for (int i = 0; i < WARP; ++i) if (i == k) mine = a[i];
                const float2 pivot = make_float2(__shfl_sync(0xffffffffu, mine.x, k), __shfl_sync(0xffffffffu, mine.y, k));
                const float2 value = divide(mine, pivot);
                #pragma unroll
                for (int i = 0; i < WARP; ++i) {                 // rows beyond the system are zero and stay so
                    const float2 factor = make_float2(__shfl_sync(0xffffffffu, a[i].x, k), __shfl_sync(0xffffffffu, a[i].y, k));
                    if (i == k) a[i] = value;
                    else { const float2 change = multiply(factor, value); a[i].x -= change.x; a[i].y -= change.y; }
                }
            }
            #pragma unroll
            for (int i = 0; i < WARP; ++i) if (i < size && j < columns) system[i*columns+j] = a[i];
        }
        __syncthreads();
    }
    else for (int k = 0; k < size; ++k) {
        if (tid < columns) row[tid] = divide(system[k*columns+tid], system[k*columns+k]);
        if (tid < size) column[tid] = system[tid*columns+k];
        __syncthreads();
        for (int j = k+threadIdx.x; j < columns; j += blockDim.x) {
            const float2 value = row[j];
            for (int i = threadIdx.y; i < size; i += blockDim.y) {
                float2* element = system+i*columns+j;
                if (i == k) *element = value;
                else {
                    const float2 change = multiply(column[i], value);
                    element->x -= change.x; element->y -= change.y;
                }
            }
        }
        __syncthreads();
    }
    const long block = ((long)u*x.symbols+s)*(x.entries+1)+number;
    for (int i = tid; i < inputs; i += threads)
        x.weights[block*inputs+i] = fitted && i < size ? system[i*columns+size] : make_float2(0.f, 0.f);
    if (tid == 0) x.available[block] = fitted ? 1 : 0;
}

extern "C" __global__ void filter_history(const Context x) {
    // grid: (cells, UEs); block: neighbors x entries threads
    const int u = blockIdx.y, c = blockIdx.x, threads = blockDim.x*blockDim.y, tid = threadIdx.y*blockDim.x+threadIdx.x;
    const int antennas = x.antennas, width = 2*antennas, inputs = (x.entries+1)*x.neighbors, numbers = x.entries+1;
    extern __shared__ float2 value[];                            // [inputs, B]; the current TB's inputs stay unused
    __shared__ Entries memory;
    memory.load(x, u);
    __syncthreads();
    const int* port = memory.port;
    const bool active = x.active[u] > .5f;
    for (int j = x.neighbors+tid; j < inputs; j += threads) {
        const int entry = j/x.neighbors-1, n = j % x.neighbors;
        const float* source = nullptr;
        if (active && port[entry] >= 0) {
            const long base = (((long)u*x.users+port[entry])*x.cells+c)*x.neighbors+n;
            if (x.neighbor_valid[base] != 0)
                source = x.memory+(((long)u*x.history+entry)*x.cells+x.neighbor_index[base])*(width+1);
        }
        for (int v = 0; v < antennas; ++v)
            value[j*antennas+v] = source ? make_float2(source[v], source[antennas+v]) : make_float2(0.f, 0.f);
    }
    __syncthreads();
    const long symbol = (long)u*x.symbols+x.cell_symbol[u*x.cells+c];
    for (int t = tid; t < numbers*antennas; t += threads) {
        const int number = t/antennas, v = t % antennas, size = (number+1)*x.neighbors;
        const float2* w = x.weights+(symbol*numbers+number)*inputs;
        float2 sum = make_float2(0.f, 0.f);
        for (int j = x.neighbors; j < size; ++j) {
            const float2 p = multiply(w[j], value[j*antennas+v]);
            sum.x += p.x; sum.y += p.y;
        }
        float* target = x.memory_part+(((long)u*x.cells+c)*numbers+number)*width;
        target[v] = sum.x; target[antennas+v] = sum.y;
    }
}

extern "C" __global__ void filter_predict(const Context x) {
    // grid: (groups of cells, UEs x DMRS symbols); block: (antennas, cells of a group)
    const int s = blockIdx.y % x.symbols, u = blockIdx.y/x.symbols, v = threadIdx.x, antennas = x.antennas;
    const int threads = blockDim.x*blockDim.y, tid = threadIdx.y*blockDim.x+threadIdx.x, width = 2*antennas;
    const int numbers = x.entries+1, inputs = numbers*x.neighbors;
    const int q = blockIdx.x*blockDim.y+threadIdx.y;             // the cell's place among the cells of its DMRS symbol
    extern __shared__ float difference[];                        // [numbers+1, threads]
    if (q < x.symbol_cells_count && v < antennas) {
        const int c = x.symbol_cells[((long)u*x.symbols+s)*x.symbol_cells_count+q];
        const bool active = x.active[u] > .5f;
        int position[MAX_NEIGHBORS];
        float2 value[MAX_NEIGHBORS];
        #pragma unroll
        for (int n = 0; n < MAX_NEIGHBORS; ++n)
            position[n] = n < x.neighbors && active ? x.own_neighbors[((long)u*x.cells+c)*x.neighbors+n] : -1;
        #pragma unroll
        for (int n = 0; n < MAX_NEIGHBORS; ++n) {
            const float* source = x.ls+((long)u*x.positions+max(position[n], 0))*width;
            value[n] = position[n] >= 0 ? make_float2(source[v], source[antennas+v]) : make_float2(0.f, 0.f);
        }
        const float* own = x.ls+((long)u*x.positions+x.pack[u*x.cells+c])*width;
        const float2 mine = make_float2(own[v], own[antennas+v]);
        const float* part = x.memory_part+((long)u*x.cells+c)*numbers*width;
        float* target = x.prediction+((long)u*x.cells+c)*numbers*width;
        for (int number = 0; number < numbers; ++number) {
            const float2* w = x.weights+(((long)u*x.symbols+s)*numbers+number)*inputs;
            float2 sum = make_float2(part[number*width+v], part[number*width+antennas+v]);
            #pragma unroll
            for (int n = 0; n < MAX_NEIGHBORS; ++n) {
                if (n >= x.neighbors) break;
                const float2 p = multiply(w[n], value[n]);
                sum.x += p.x; sum.y += p.y;
            }
            target[number*width+v] = sum.x; target[number*width+antennas+v] = sum.y;
            const float dr = mine.x-sum.x, di = mine.y-sum.y;
            difference[number*threads+tid] = dr*dr+di*di;
        }
        difference[numbers*threads+tid] = v == 0 ? fmaxf(x.noise[0]*x.noise_factors[u*x.cells+c], 0.f) : 0.f;
    } else {
        for (int number = 0; number <= numbers; ++number) difference[number*threads+tid] = 0.f;
    }
    __syncthreads();
    for (int row = tid; row <= numbers; row += threads) {
        float sum = 0.f;
        for (int t = 0; t < threads; ++t) sum += difference[row*threads+t];
        x.sums[(((long)u*x.symbols+s)*x.groups+blockIdx.x)*(numbers+1)+row] = sum;
    }
}

extern "C" __global__ void filter_expand(const Context x) {
    // grid: (blocks of positions, UEs)
    const int u = blockIdx.y, threads = blockDim.x, tid = threadIdx.x, numbers = x.entries+1, width = 2*x.antennas;
    const int rows = x.symbols*(numbers+1);                      // per DMRS symbol: one sum per K, and the noise
    extern __shared__ float parts[];                             // [rows, groups]
    float* total = parts+rows*x.groups;                          // [rows]
    float* decided = total+rows;                                 // [symbols, 2]: chosen K, weight of the prediction
    for (int item = tid; item < rows*x.groups; item += threads) {   // all groups' sums of this UE, in parallel
        const int row = item/x.groups, g = item % x.groups;
        parts[item] = x.sums[(((long)u*x.symbols+row/(numbers+1))*x.groups+g)*(numbers+1)+row % (numbers+1)];
    }
    __syncthreads();
    for (int row = tid; row < rows; row += threads) {
        float sum = 0.f;
        for (int g = 0; g < x.groups; ++g) sum += parts[row*x.groups+g];
        total[row] = sum;
    }
    __syncthreads();
    for (int s = tid; s < x.symbols; s += threads) {
        const long symbol = (long)u*x.symbols+s;
        const float noise = total[s*(numbers+1)+numbers]/x.symbol_cells_count;
        float best = 0.f;
        int chosen = -1;
        for (int number = 0; number < numbers; ++number) {
            const float mean = total[s*(numbers+1)+number]/(x.symbol_cells_count*x.antennas);
            if (x.available[symbol*numbers+number] && (chosen < 0 || mean < best)) { best = mean; chosen = number; }
        }
        const float denominator = noise+fmaxf(best-noise, 1e-3f*noise);
        decided[2*s] = chosen < 0 ? 0.f : (float)chosen;
        // a negative weight marks "no fit", which the slopes need apart from a zero weight
        decided[2*s+1] = chosen < 0 ? -1.f : (denominator <= 0.f ? 0.f : noise/denominator);
        if (blockIdx.x == 0) { x.choice[symbol] = chosen < 0 ? 0 : chosen; x.gain[symbol] = decided[2*s+1]; }
    }
    __syncthreads();
    const long p = (long)blockIdx.x*threads+tid;
    if (p >= x.positions) return;
    const long i = (long)u*x.positions+p;
    float* out = x.out+i*width;
    if (x.active[u] <= .5f) {
        for (int v = 0; v < width; ++v) out[v] = 0.f;
        return;
    }
    bool sloped = true;                                          // every DMRS symbol has a fit
    for (int s = 0; s < x.symbols; ++s) sloped = sloped && decided[2*s+1] >= 0.f;
    const float* own[4];
    const float* predicted[4];
    float gain[4];
    for (int q = 0; q < 4; ++q) {
        const int cell = x.position_cells[i*4+q], s = x.cell_symbol[u*x.cells+cell];
        own[q] = x.ls+((long)u*x.positions+x.position_packs[i*4+q])*width;
        predicted[q] = x.prediction+(((long)u*x.cells+cell)*numbers+(int)decided[2*s])*width;
        gain[q] = fmaxf(decided[2*s+1], 0.f);
    }
    const float in_time = sloped ? x.time_factor[i] : 0.f, in_frequency = sloped ? x.frequency_factor[i] : 0.f;
    if (width % 4 == 0) {                                        // 16-byte accesses (all rows are 16-byte aligned)
        for (int v = 0; v < width; v += 4) {
            float4 refined[4];
            for (int q = 0; q < 4; ++q) {
                const float4 a = *reinterpret_cast<const float4*>(own[q]+v);
                const float4 b = *reinterpret_cast<const float4*>(predicted[q]+v);
                refined[q] = make_float4(a.x+gain[q]*(b.x-a.x), a.y+gain[q]*(b.y-a.y), a.z+gain[q]*(b.z-a.z),
                                         a.w+gain[q]*(b.w-a.w));
            }
            *reinterpret_cast<float4*>(out+v) = make_float4(
                refined[0].x+in_time*(refined[1].x-refined[0].x)+in_frequency*(refined[3].x-refined[2].x),
                refined[0].y+in_time*(refined[1].y-refined[0].y)+in_frequency*(refined[3].y-refined[2].y),
                refined[0].z+in_time*(refined[1].z-refined[0].z)+in_frequency*(refined[3].z-refined[2].z),
                refined[0].w+in_time*(refined[1].w-refined[0].w)+in_frequency*(refined[3].w-refined[2].w));
        }
        return;
    }
    for (int v = 0; v < width; ++v) {
        float refined[4];
        for (int q = 0; q < 4; ++q) refined[q] = own[q][v]+gain[q]*(predicted[q][v]-own[q][v]);
        out[v] = refined[0]+in_time*(refined[1]-refined[0])+in_frequency*(refined[3]-refined[2]);
    }
}
