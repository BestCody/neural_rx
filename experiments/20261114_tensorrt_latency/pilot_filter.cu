// The channel filter of the proposed NRX as one CUDA kernel (FP32).
//
// One thread block per (UE, pilot cell). It does everything that ChannelMemory.call (defer_uncertainty=True) and
// PilotLayout.expand_channel do in TensorFlow, without writing intermediate results to memory:
//   1. one thread per neighbor j = (entry e, neighbor n): gather the LS estimate of the current TB (e = 0) or the
//      memory entry (e = 1..7), apply the validity rules, compute the 13 features, run the 13-16-7 MLP;
//   2. softmax over the 48 neighbor logits and one zero logit;
//   3. 12 threads form the coefficients theta_r (3 terms x 4 antennas, complex);
//   4. the block writes the refined estimate to every grid position that belongs to its cell.
// The output has the layout of the LS estimate, [U, S*T, 2B] (real parts, then imaginary parts), and replaces it at
// the input of StateInit. Checked against the TensorFlow receiver by time_fused.py.

#define B 4                      // BS antennas
#define ENTRIES 8                // current TB and seven memory entries
#define NEIGHBORS 6              // per entry: two DMRS symbols x three nearest pilots
#define J (ENTRIES*NEIGHBORS)
#define IN 13
#define HIDDEN 16
#define OUT 7
#define HISTORY (ENTRIES-1)
#define THREADS 64
#define WEIGHTS (IN*HIDDEN+HIDDEN+HIDDEN*OUT+OUT)

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

extern "C" __global__ void __launch_bounds__(THREADS) pilot_filter(
        const float* __restrict__ ls,              // [U, P, 2B] LS estimate on the grid (nearest-pilot interpolation)
        const float* __restrict__ noise,           // [1] noise variance
        const float* __restrict__ memory,          // [U, HISTORY, C, 2B+1] memory entries, newest first
        const int* __restrict__ age,               // [U, HISTORY] entry age in TBs
        const int* __restrict__ ports,             // [U, HISTORY] DMRS port of each entry
        const float* __restrict__ valid,           // [U, HISTORY] entry validity (0/1)
        const float* __restrict__ active,          // [U] scheduled UEs (0/1)
        const int* __restrict__ pack,              // [U, C] grid position of each distinct pilot estimate
        const float* __restrict__ noise_factors,   // [U, C] LS error variance per unit noise
        const int* __restrict__ neighbor_index,    // [U, U, C, NEIGHBORS] pilot cells of the neighbors, per port
        const unsigned char* __restrict__ neighbor_valid,  // [U, U, C, NEIGHBORS]
        const float* __restrict__ neighbor_delta,  // [U, U, C, NEIGHBORS, 2] offset in time (ms) and frequency (MHz)
        const float* __restrict__ geometry,        // [U, C, 4]
        const float* __restrict__ weights,         // kernel1 [IN, HIDDEN], bias1, kernel2 [HIDDEN, OUT], bias2
        const float* __restrict__ phi,             // [U, P, 3] local first-order model
        const int* __restrict__ cell_offsets,      // [U, C+1] start of each cell's grid positions
        const int* __restrict__ cell_positions,    // [U, P] grid positions sorted by cell
        float* __restrict__ out,                   // [U, P, 2B]
        int users, int cells, int positions, float tb_ms) {
    const float EPS = 1.1920928955078125e-7f;
    const int u = blockIdx.x/cells, c = blockIdx.x % cells, tid = threadIdx.x;
    __shared__ float w[WEIGHTS];
    __shared__ float logit[J], weight[J];
    __shared__ float2 term[J][3];
    __shared__ float2 innovation[J][B];
    __shared__ float2 coefficient[B][3];
    __shared__ float current[2*B+1];
    __shared__ float softmax[2];

    for (int i = tid; i < WEIGHTS; i += THREADS) w[i] = weights[i];
    const bool user_active = active[u] > .5f;
    const float noise_variance = noise[0];
    if (tid < 2*B) current[tid] = ls[((long)u*positions+pack[u*cells+c])*2*B+tid];
    if (tid == 2*B) current[2*B] = noise_variance*noise_factors[u*cells+c];
    __syncthreads();

    float h0r[B], h0i[B], p0 = 0.f;
    for (int r = 0; r < B; ++r) {
        h0r[r] = current[r];
        h0i[r] = current[B+r];
        p0 += h0r[r]*h0r[r]+h0i[r]*h0i[r];
    }
    p0 /= B;
    const float q0 = fmaxf(current[2*B], 0.f);
    const float reference = fmaxf(p0+q0, 1e-30f);
    const float floor_ = EPS*reference;

    if (tid < J) {
        const int e = tid/NEIGHBORS, n = tid % NEIGHBORS;
        bool entry_valid = user_active;
        int port = u, entry_age = 0;
        if (e > 0) {
            entry_valid = entry_valid && valid[u*HISTORY+e-1] > .5f;
            port = ports[u*HISTORY+e-1];
            entry_age = age[u*HISTORY+e-1];
            if (port < 0 || port >= users) { port = 0; entry_valid = false; }
        }
        const int base = ((u*users+port)*cells+c)*NEIGHBORS+n;
        const int index = neighbor_index[base];
        const float dt = neighbor_delta[2*base], df = neighbor_delta[2*base+1];
        const bool central = dt == 0.f && df == 0.f;
        const bool allowed = entry_valid && neighbor_valid[base] != 0 && !(e == 0 && central);
        float record[2*B+1];
        for (int k = 0; k < 2*B+1; ++k) record[k] = 0.f;
        if (allowed) {
            if (e == 0) {
                const float* source = ls+((long)u*positions+pack[u*cells+index])*2*B;
                for (int k = 0; k < 2*B; ++k) record[k] = source[k];
                record[2*B] = noise_variance*noise_factors[u*cells+index];
            } else {
                const float* source = memory+(((long)u*HISTORY+e-1)*cells+index)*(2*B+1);
                for (int k = 0; k < 2*B+1; ++k) record[k] = source[k];
            }
        }
        const float qj = fmaxf(record[2*B], 0.f);
        float pj = 0.f, cr = 0.f, ci = 0.f, disagreement = 0.f;
        for (int r = 0; r < B; ++r) {
            const float hr = record[r], hi = record[B+r];
            pj += hr*hr+hi*hi;
            cr += h0r[r]*hr+h0i[r]*hi;          // conj(h0)*hj
            ci += h0r[r]*hi-h0i[r]*hr;
            const float ir = allowed ? hr-h0r[r] : 0.f, ii = allowed ? hi-h0i[r] : 0.f;
            innovation[tid][r] = make_float2(ir, ii);
            disagreement += ir*ir+ii*ii;
        }
        pj /= B;
        cr /= B;
        ci /= B;
        disagreement /= B;
        const float scale = sqrtf(fmaxf(p0, floor_)*fmaxf(pj, floor_));
        const float age_ms = entry_age*tb_ms;
        const float* g = geometry+(u*cells+c)*4;
        const float x[IN] = {cr/scale, ci/scale, logf(fmaxf(pj+qj, floor_)/reference), q0/reference,
                             qj/fmaxf(pj+qj, floor_), disagreement/fmaxf(reference+pj+qj, floor_),
                             log1pf(age_ms), dt-age_ms, df, g[0], g[1], g[2], g[3]};
        float hidden[HIDDEN];
        for (int h = 0; h < HIDDEN; ++h) {
            float acc = w[IN*HIDDEN+h];
            for (int i = 0; i < IN; ++i) acc += x[i]*w[i*HIDDEN+h];
            hidden[h] = fmaxf(acc, 0.f);
        }
        const float* w2 = w+IN*HIDDEN+HIDDEN;
        const float* b2 = w2+HIDDEN*OUT;
        float y[OUT];
        for (int o = 0; o < OUT; ++o) {
            float acc = b2[o];
            for (int h = 0; h < HIDDEN; ++h) acc += hidden[h]*w2[h*OUT+o];
            y[o] = acc;
        }
        logit[tid] = allowed ? y[0] : -1e9f;
        for (int k = 0; k < 3; ++k) term[tid][k] = make_float2(tanhf(y[1+2*k]), tanhf(y[2+2*k]));
    }
    __syncthreads();

    if (tid < 32) {  // softmax over the J logits and one zero logit
        float m = 0.f;
        for (int j = tid; j < J; j += 32) m = fmaxf(m, logit[j]);
        m = warp_max(m);
        float s = tid == 0 ? expf(-m) : 0.f;
        for (int j = tid; j < J; j += 32) s += expf(logit[j]-m);
        s = warp_sum(s);
        if (tid == 0) { softmax[0] = m; softmax[1] = s; }
    }
    __syncthreads();
    if (tid < J) weight[tid] = expf(logit[tid]-softmax[0])/softmax[1];
    __syncthreads();

    if (tid < 3*B) {  // theta_{r,k} = sum_j beta_{j,k} innovation_{j,r} (+ h0 for k = 0)
        const int r = tid/3, k = tid % 3;
        float ar = 0.f, ai = 0.f;
        for (int j = 0; j < J; ++j) {
            const float2 t = term[j][k], v = innovation[j][r];
            const float br = weight[j]*t.x, bi = weight[j]*t.y;
            ar += br*v.x-bi*v.y;
            ai += br*v.y+bi*v.x;
        }
        if (k == 0) { ar += h0r[r]; ai += h0i[r]; }
        coefficient[r][k] = user_active ? make_float2(ar, ai) : make_float2(0.f, 0.f);
    }
    __syncthreads();

    const int start = cell_offsets[u*(cells+1)+c], stop = cell_offsets[u*(cells+1)+c+1];
    for (int i = start+tid; i < stop; i += THREADS) {
        const int p = cell_positions[u*positions+i];
        const float* f = phi+((long)u*positions+p)*3;
        float* o = out+((long)u*positions+p)*2*B;
        for (int r = 0; r < B; ++r) {
            o[r] = coefficient[r][0].x*f[0]+coefficient[r][1].x*f[1]+coefficient[r][2].x*f[2];
            o[B+r] = coefficient[r][0].y*f[0]+coefficient[r][1].y*f[1]+coefficient[r][2].y*f[2];
        }
    }
}
