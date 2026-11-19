// The channel filter of the proposed NRX as CUDA kernels (FP32). They do what ChannelFilter.__call__ and
// PilotLayout.expand_channel do in TensorFlow (src/neural_rx/receiver/channel_filter.py); filter_kernels.py launches
// them and time_fused.py checks their output against the TensorFlow receiver.
//
// Before the slot arrives (only the memory is needed):
//   filter_gram   normal equations of each of the newest entries: one block per (UE, DMRS symbol, entry) and per
//                 element of the upper triangle of the Gram matrix or of the right-hand side
//   filter_solve  for each number K of entries used (0..entries): sums the normal equations of the stored TBs that
//                 have all K lagged entries, adds the ridge term, and solves by Gauss-Jordan elimination
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
// All sizes are fields of the context, which the kernels receive by value; every field is 8 bytes, so the host fills
// it as an array of 64-bit words.

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
    const int* slope_cells;             // [U, C, 3] cell at the other DMRS symbol, lower and upper neighbor in frequency
    const int* cell_of_position;        // [U, P]
    const float* time_factor;           // [U, P] offset of the position, in units of the difference to the other symbol
    const float* frequency_factor;      // [U, P] the same for the difference between the frequency neighbors
    const int* pair_row;                // [pairs] upper triangle of the Gram matrix
    const int* pair_column;             // [pairs]
    float2* gram;                       // [U, S, R, pairs]
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
    long long users, cells, positions, antennas, history, neighbors, symbols, symbol_cells_count, entries, fit_tbs, pairs, groups;
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

// The estimate [2B] of neighbor n of cell c in one record, or null if it does not enter. The record is the current
// TB (entry < 0, received on port u) or memory entry `entry`; `centre` is the port whose cells are the centres, and
// `own` marks the record of the cell itself, whose central neighbor is left out.
__device__ __forceinline__ const float* neighbor(const Context& x, int u, int centre, int entry, bool own, int c, int n) {
    const int users = x.users, cells = x.cells, width = 2*x.antennas;
    int port = u;
    if (entry >= 0) {
        if (x.valid[u*x.history+entry] <= .5f) return nullptr;
        port = x.ports[u*x.history+entry];
        if (port < 0 || port >= users) return nullptr;
    }
    const long base = (((long)centre*users+port)*cells+c)*x.neighbors+n;
    if (x.neighbor_valid[base] == 0) return nullptr;
    if (own && x.neighbor_delta[2*base] == 0.f && x.neighbor_delta[2*base+1] == 0.f) return nullptr;
    const int index = x.neighbor_index[base];
    return entry < 0 ? x.ls+((long)u*x.positions+x.pack[u*cells+index])*width
                     : x.memory+(((long)u*x.history+entry)*cells+index)*(width+1);
}

// The memory entry that is `lag` TBs older than entry `target`, or -1.
__device__ __forceinline__ int lagged(const Context& x, int u, int target, int lag) {
    const int wanted = x.age[u*x.history+target]+lag;
    for (int j = 0; j < x.history; ++j)
        if (x.valid[u*x.history+j] > .5f && x.age[u*x.history+j] == wanted) return j;
    return -1;
}

// Record k (0: the target itself, k >= 1: the entry as much older than it as the current TB's kth newest entry is
// older than the current TB) of the normal equations of memory entry `target`; -1 if it does not exist.
__device__ __forceinline__ int fit_record(const Context& x, int u, int target, int k) {
    if (k == 0) return target;
    if (x.valid[u*x.history+k-1] <= .5f) return -1;
    return lagged(x, u, target, x.age[u*x.history+k-1]);
}

extern "C" __global__ void filter_gram(const Context x) {
    const int inputs = (x.entries+1)*x.neighbors, antennas = x.antennas, threads = blockDim.x, tid = threadIdx.x;
    const int item = blockIdx.x, r = blockIdx.y % x.fit_tbs, s = (blockIdx.y/x.fit_tbs) % x.symbols;
    const int u = blockIdx.y/(x.fit_tbs*x.symbols);
    extern __shared__ float2 partial[];
    const bool right = item >= x.pairs;                          // an element of the right-hand side
    const int i = right ? item-x.pairs : x.pair_row[item], j = right ? 0 : x.pair_column[item];
    const bool usable = x.active[u] > .5f && x.valid[u*x.history+r] > .5f;
    float2 sum = make_float2(0.f, 0.f);
    if (usable) {
        const int centre = x.ports[u*x.history+r];
        const int ri = fit_record(x, u, r, i/x.neighbors), rj = right ? r : fit_record(x, u, r, j/x.neighbors);
        if (ri >= 0 && rj >= 0 && centre >= 0 && centre < x.users) {
            const int* cells = x.symbol_cells+((long)centre*x.symbols+s)*x.symbol_cells_count;
            for (int q = tid; q < x.symbol_cells_count; q += threads) {
                const int c = cells[q];
                const float* a = neighbor(x, u, centre, ri, i < x.neighbors, c, i % x.neighbors);
                const float* b = right ? x.memory+(((long)u*x.history+r)*x.cells+c)*(2*antennas+1)
                                       : neighbor(x, u, centre, rj, j < x.neighbors, c, j % x.neighbors);
                if (a == nullptr || b == nullptr) continue;
                for (int v = 0; v < antennas; ++v) {
                    const float2 p = conjugate_multiply(make_float2(a[v], a[antennas+v]), make_float2(b[v], b[antennas+v]));
                    sum.x += p.x; sum.y += p.y;
                }
            }
        }
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float2 total = make_float2(0.f, 0.f);
        for (int t = 0; t < threads; ++t) { total.x += partial[t].x; total.y += partial[t].y; }
        const long block = ((long)u*x.symbols+s)*x.fit_tbs+r;
        if (right) x.rhs[block*inputs+i] = total; else x.gram[block*x.pairs+item] = total;
    }
}

extern "C" __global__ void filter_solve(const Context x) {
    const int inputs = (x.entries+1)*x.neighbors, tid = threadIdx.x;
    const int number = blockIdx.x, s = blockIdx.y % x.symbols, u = blockIdx.y/x.symbols;
    const int size = (number+1)*x.neighbors, columns = size+1;
    extern __shared__ float2 shared[];
    float2* system = shared;                                     // [size, size+1], the right-hand side last
    float2* row = system+inputs*(inputs+1);                      // [size+1]
    float2* column = row+inputs+1;                               // [size]
    // entries whose normal equations have all of these lags, and whether the current TB has these entries
    bool fitted = false, complete[16];                           // fit_tbs <= 16 (checked by the host)
    for (int r = 0; r < x.fit_tbs && r < 16; ++r) {
        complete[r] = x.active[u] > .5f && x.valid[u*x.history+r] > .5f;
        for (int k = 1; k <= number && complete[r]; ++k) complete[r] = fit_record(x, u, r, k) >= 0;
        fitted = fitted || complete[r];
    }
    for (int k = 0; k < number; ++k) fitted = fitted && x.valid[u*x.history+k] > .5f;
    for (int e = tid; e < size*columns; e += blockDim.x) {
        const int i = e/columns, j = e % columns;
        float2 value = make_float2(0.f, 0.f);
        for (int r = 0; r < x.fit_tbs && r < 16; ++r) {
            if (!complete[r]) continue;
            const long block = ((long)u*x.symbols+s)*x.fit_tbs+r;
            float2 term;
            if (j == size) term = x.rhs[block*inputs+i];
            else {
                const int a = i < j ? i : j, b = i < j ? j : i;          // stored: row a <= column b
                term = x.gram[block*x.pairs+(long)a*inputs-(long)a*(a-1)/2+(b-a)];
                if (i > j) term.y = -term.y;
            }
            value.x += term.x; value.y += term.y;
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
    for (int k = 0; k < size; ++k) {                             // Gauss-Jordan; positive definite, so no pivoting
        const float2 pivot = system[k*columns+k];
        for (int j = tid; j < columns; j += blockDim.x) row[j] = divide(system[k*columns+j], pivot);
        for (int i = tid; i < size; i += blockDim.x) column[i] = system[i*columns+k];
        __syncthreads();
        for (int j = tid; j < columns; j += blockDim.x)
            for (int i = 0; i < size; ++i) {
                float2 factor = make_float2(-column[i].x, -column[i].y);
                if (i == k) factor.x += 1.f;
                const float2 change = multiply(factor, row[j]);
                system[i*columns+j].x += change.x; system[i*columns+j].y += change.y;
            }
        __syncthreads();
    }
    const long block = ((long)u*x.symbols+s)*(x.entries+1)+number;
    for (int i = tid; i < inputs; i += blockDim.x)
        x.weights[block*inputs+i] = fitted && i < size ? system[i*columns+size] : make_float2(0.f, 0.f);
    if (tid == 0) x.available[block] = fitted ? 1 : 0;
}

extern "C" __global__ void filter_history(const Context x) {
    // grid: (cells, UEs); block: neighbors x (entries+1) threads
    const int u = blockIdx.y, c = blockIdx.x, threads = blockDim.x*blockDim.y, tid = threadIdx.y*blockDim.x+threadIdx.x;
    const int antennas = x.antennas, width = 2*antennas, inputs = (x.entries+1)*x.neighbors, numbers = x.entries+1;
    extern __shared__ float2 value[];                            // [inputs, B]; the current TB's inputs stay unused
    for (int j = x.neighbors+tid; j < inputs; j += threads) {
        const float* source = x.active[u] > .5f ? neighbor(x, u, u, j/x.neighbors-1, false, c, j % x.neighbors) : nullptr;
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
    extern __shared__ float2 shared[];
    float2* value = shared+tid*x.neighbors;                      // [neighbors] the thread's inputs
    float* difference = reinterpret_cast<float*>(shared+threads*x.neighbors);      // [numbers+1, threads]
    if (q < x.symbol_cells_count && v < antennas) {
        const int c = x.symbol_cells[((long)u*x.symbols+s)*x.symbol_cells_count+q];
        const bool active = x.active[u] > .5f;
        for (int n = 0; n < x.neighbors; ++n) {
            const float* source = active ? neighbor(x, u, u, -1, true, c, n) : nullptr;
            value[n] = source ? make_float2(source[v], source[antennas+v]) : make_float2(0.f, 0.f);
        }
        const float* own = x.ls+((long)u*x.positions+x.pack[u*x.cells+c])*width;
        const float* part = x.memory_part+((long)u*x.cells+c)*numbers*width;
        float* target = x.prediction+((long)u*x.cells+c)*numbers*width;
        for (int number = 0; number < numbers; ++number) {
            const float2* w = x.weights+(((long)u*x.symbols+s)*numbers+number)*inputs;
            float2 sum = make_float2(part[number*width+v], part[number*width+antennas+v]);
            for (int n = 0; n < x.neighbors; ++n) {
                const float2 p = multiply(w[n], value[n]);
                sum.x += p.x; sum.y += p.y;
            }
            target[number*width+v] = sum.x; target[number*width+antennas+v] = sum.y;
            const float dr = own[v]-sum.x, di = own[antennas+v]-sum.y;
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
    extern __shared__ float total[];                             // [rows]
    float* decided = total+rows;                                 // [symbols, 2]: chosen K, weight of the prediction
    for (int row = tid; row < rows; row += threads) {
        const float* sums = x.sums+((long)u*x.symbols+row/(numbers+1))*x.groups*(numbers+1)+row % (numbers+1);
        float sum = 0.f;
        for (int g = 0; g < x.groups; ++g) sum += sums[g*(numbers+1)];
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
    const int c = x.cell_of_position[i];
    const int* others = x.slope_cells+((long)u*x.cells+c)*3;
    const int cell[4] = {c, others[0], others[1], others[2]};
    const float* own[4];
    const float* predicted[4];
    float gain[4];
    for (int q = 0; q < 4; ++q) {
        const int s = x.cell_symbol[u*x.cells+cell[q]];
        own[q] = x.ls+((long)u*x.positions+x.pack[u*x.cells+cell[q]])*width;
        predicted[q] = x.prediction+(((long)u*x.cells+cell[q])*numbers+(int)decided[2*s])*width;
        gain[q] = fmaxf(decided[2*s+1], 0.f);
    }
    const float in_time = sloped ? x.time_factor[i] : 0.f, in_frequency = sloped ? x.frequency_factor[i] : 0.f;
    for (int v = 0; v < width; ++v) {
        float refined[4];
        for (int q = 0; q < 4; ++q) refined[q] = own[q][v]+gain[q]*(predicted[q][v]-own[q][v]);
        out[v] = refined[0]+in_time*(refined[1]-refined[0])+in_frequency*(refined[3]-refined[2]);
    }
}
