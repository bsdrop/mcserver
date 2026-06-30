/*
 * Canvas GPU Chunk Generation Acceleration
 * OpenCL kernel: Minecraft Improved Perlin Noise (exact replica of ImprovedNoise.java)
 *
 * Gradient table matches SimplexNoise.GRADIENT in MC 1.21:
 *   {{1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
 *    {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
 *    {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1},
 *    {1,1,0},{0,-1,1},{-1,1,0},{0,-1,-1}}
 *
 * Permutation table p[] is 256-byte, passed as int16 buffer.
 * For each noise instance we pass p[0..255] as the first 256 ints.
 */

constant int GRAD_X[16] = { 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1, 0,-1, 0};
constant int GRAD_Y[16] = { 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1};
constant int GRAD_Z[16] = { 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 0, 1, 0,-1};

inline double mc_grad_dot(int hash, double x, double y, double z) {
    int h = hash & 15;
    return (double)GRAD_X[h] * x + (double)GRAD_Y[h] * y + (double)GRAD_Z[h] * z;
}

inline double mc_fade(double t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

inline double mc_lerp(double t, double a, double b) {
    return a + t * (b - a);
}

/*
 * Single Perlin noise sample.
 * perm: 256 ints (permutation table for this noise instance)
 * xo,yo,zo: per-instance offsets (added to x,y,z)
 */
double mc_perlin_sample(
    __global const int* perm,
    double x, double y, double z,
    double xo, double yo, double zo,
    double yScale, double yFudge
) {
    x = x + xo;
    y = y + yo;
    z = z + zo;

    double xf = floor(x);
    double yf = floor(y);
    double zf = floor(z);
    double xr = x - xf;
    double yr = y - yf;
    double zr = z - zf;

    // yScale/yFudge handling (matches ImprovedNoise.java exactly)
    double yrFudge = 0.0;
    if (yScale != 0.0) {
        double fudgeLimit = (yFudge >= 0.0 && yFudge < yr) ? yFudge : yr;
        yrFudge = floor(fudgeLimit / yScale + 1e-7f) * yScale;
    }
    yr = yr - yrFudge;

    int xi = (int)xf & 0xFF;
    int yi = (int)yf & 0xFF;
    int zi = (int)zf & 0xFF;

    // Perm table lookups
    int a  = perm[xi]       & 0xFF;
    int b  = perm[(xi+1)&0xFF] & 0xFF;
    int aa = perm[(a+yi)&0xFF] & 0xFF;
    int ab = perm[(a+yi+1)&0xFF] & 0xFF;
    int ba = perm[(b+yi)&0xFF] & 0xFF;
    int bb = perm[(b+yi+1)&0xFF] & 0xFF;

    int h000 = perm[(aa+zi)&0xFF] & 15;
    int h001 = perm[(ba+zi)&0xFF] & 15;
    int h010 = perm[(ab+zi)&0xFF] & 15;
    int h011 = perm[(bb+zi)&0xFF] & 15;
    int h100 = perm[(aa+zi+1)&0xFF] & 15;
    int h101 = perm[(ba+zi+1)&0xFF] & 15;
    int h110 = perm[(ab+zi+1)&0xFF] & 15;
    int h111 = perm[(bb+zi+1)&0xFF] & 15;

    double xm1 = xr - 1.0, ym1 = yr - 1.0, zm1 = zr - 1.0;

    double u = mc_fade(xr);
    double v = mc_fade(yr);  // original yr for fade, not the fudged one
    double w = mc_fade(zr);

    double x0 = mc_lerp(u,
        mc_grad_dot(h000, xr,   yr, zr),
        mc_grad_dot(h001, xm1,  yr, zr));
    double x1 = mc_lerp(u,
        mc_grad_dot(h010, xr,   ym1, zr),
        mc_grad_dot(h011, xm1,  ym1, zr));
    double x2 = mc_lerp(u,
        mc_grad_dot(h100, xr,   yr, zm1),
        mc_grad_dot(h101, xm1,  yr, zm1));
    double x3 = mc_lerp(u,
        mc_grad_dot(h110, xr,   ym1, zm1),
        mc_grad_dot(h111, xm1,  ym1, zm1));

    return mc_lerp(w, mc_lerp(v, x0, x1), mc_lerp(v, x2, x3));
}

/*
 * Batch Perlin evaluation: compute N noise values simultaneously.
 *
 * Layout of 'input' (8 doubles per sample):
 *   [x, y, z, xo, yo, zo, yScale, yFudge] × N
 *
 * 'perm_tables': perm_table_idx[i] selects which 256-int block to use
 *                from perm_storage (each noise instance has its own 256 ints)
 * 'results': N doubles
 */
__kernel void perlin_batch(
    __global const double* input,
    __global const int* perm_storage,   // [numInstances × 256]
    __global const int* perm_instance,  // [N] which perm block for sample i
    __global double* results,
    const int N
) {
    int id = get_global_id(0);
    if (id >= N) return;

    int base = id * 8;
    double x      = input[base+0];
    double y      = input[base+1];
    double z      = input[base+2];
    double xo     = input[base+3];
    double yo     = input[base+4];
    double zo     = input[base+5];
    double yScale = input[base+6];
    double yFudge = input[base+7];

    int pi = perm_instance[id];
    __global const int* perm = perm_storage + pi * 256;

    results[id] = mc_perlin_sample(perm, x, y, z, xo, yo, zo, yScale, yFudge);
}

/*
 * OctaveNoise batch: compute one cell-corner value from multi-octave Perlin.
 *
 * 'octave_params' layout per octave: [xo, yo, zo, amplXZ, amplY, lacunarityXZ, lacunarityY]
 *   = 7 doubles × numOctaves
 * 'octave_perms': [numOctaves × 256] perm tables concatenated
 * 'query' layout per point: [x, y, z] = 3 doubles
 * 'results': N doubles (sum of octaves × amplitude)
 */
__kernel void octave_batch(
    __global const double* query,       // [N × 3]: x, y, z coordinates to sample
    __global const double* octave_params, // [numOctaves × 7]
    __global const int* octave_perms,   // [numOctaves × 256]
    __global double* results,           // [N]
    const int N,
    const int numOctaves,
    const double persistence            // amplitude multiplier per octave (e.g. 0.5 for lacunarity 2)
) {
    int id = get_global_id(0);
    if (id >= N) return;

    double qx = query[id*3+0];
    double qy = query[id*3+1];
    double qz = query[id*3+2];

    double value = 0.0;
    double freqXZ = 1.0, freqY = 1.0, amp = 1.0;

    for (int oct = 0; oct < numOctaves; oct++) {
        int pbase = oct * 7;
        double xo        = octave_params[pbase+0];
        double yo        = octave_params[pbase+1];
        double zo        = octave_params[pbase+2];
        double lacXZ     = octave_params[pbase+3];
        double lacY      = octave_params[pbase+4];

        __global const int* perm = octave_perms + oct * 256;

        double sx = qx * freqXZ;
        double sy = qy * freqY;
        double sz = qz * freqXZ;

        value += mc_perlin_sample(perm, sx, sy, sz, xo, yo, zo, 0.0, 0.0) * amp;

        freqXZ *= lacXZ;
        freqY  *= lacY;
        amp    *= persistence;
    }

    results[id] = value;
}

/*
 * PerlinNoise batch v2 — matches MC PerlinNoise.getValue() exactly.
 *
 * oct_params layout per octave (6 doubles):
 *   [xo, yo, zo, freqFactor, effectiveAmplitude, 0]
 * where:
 *   freqFactor = lowestFreqInputFactor * 2^i
 *   effectiveAmplitude = amplitudes[i] * lowestFreqValueFactor / 2^i
 * Null octaves (noiseLevels[i]==null) have effectiveAmplitude=0.
 *
 * query: [N*3] (x, y, z) already scaled by xzScale/yScale.
 */
__kernel void perlin_noise_v2(
    __global const double* query,       // [N*3]
    __global const double* oct_params,  // [numOctaves*6]
    __global const int*   oct_perms,    // [numOctaves*256]
    __global double* results,           // [N]
    const int N,
    const int numOctaves
) {
    int id = get_global_id(0);
    if (id >= N) return;
    double x = query[id*3+0];
    double y = query[id*3+1];
    double z = query[id*3+2];
    double val = 0.0;
    for (int i = 0; i < numOctaves; i++) {
        double amp = oct_params[i*6+4];
        if (amp == 0.0) continue;
        double xo  = oct_params[i*6+0];
        double yo  = oct_params[i*6+1];
        double zo  = oct_params[i*6+2];
        double freq = oct_params[i*6+3];
        __global const int* perm = oct_perms + i * 256;
        val += amp * mc_perlin_sample(perm, x*freq, y*freq, z*freq, xo, yo, zo, 0.0, 0.0);
    }
    results[id] = val;
}

/*
 * NormalNoise batch v2 — matches MC NormalNoise.getValue() exactly.
 * Uses two PerlinNoise evaluations (first + second*inputFactor) × valueFactor.
 *
 * For 'b' params, freqFactor already includes INPUT_FACTOR:
 *   freq_b[i] = INPUT_FACTOR * lowestFreqInputFactor_b * 2^i
 */
__kernel void normal_noise_v2(
    __global const double* query,       // [N*3] x,y,z (scaled by xzScale/yScale)
    __global const double* oct_a,       // [numA*6] first PerlinNoise octave params
    __global const int*   perm_a,       // [numA*256]
    const int numA,
    __global const double* oct_b,       // [numB*6] second PerlinNoise (freq includes INPUT_FACTOR)
    __global const int*   perm_b,       // [numB*256]
    const int numB,
    __global double* results,           // [N]
    const int N,
    const double normFactor
) {
    int id = get_global_id(0);
    if (id >= N) return;
    double x = query[id*3+0];
    double y = query[id*3+1];
    double z = query[id*3+2];

    double va = 0.0;
    for (int i = 0; i < numA; i++) {
        double amp = oct_a[i*6+4];
        if (amp == 0.0) continue;
        __global const int* pa = perm_a + i * 256;
        va += amp * mc_perlin_sample(pa, x*oct_a[i*6+3], y*oct_a[i*6+3], z*oct_a[i*6+3],
                                     oct_a[i*6+0], oct_a[i*6+1], oct_a[i*6+2], 0.0, 0.0);
    }

    double vb = 0.0;
    for (int i = 0; i < numB; i++) {
        double amp = oct_b[i*6+4];
        if (amp == 0.0) continue;
        __global const int* pb = perm_b + i * 256;
        vb += amp * mc_perlin_sample(pb, x*oct_b[i*6+3], y*oct_b[i*6+3], z*oct_b[i*6+3],
                                     oct_b[i*6+0], oct_b[i*6+1], oct_b[i*6+2], 0.0, 0.0);
    }

    results[id] = (va + vb) * normFactor;
}

/*
 * NormalNoise batch: two OctaveNoise instances blended.
 * Used for continentalness, erosion, weirdness, etc.
 *
 * NormalNoise(x,y,z) = (first.sample(x,y,z) + second.sample(x+s,y+s,z+s)) × normFactor
 *   where s = 337/331
 */
__kernel void normal_noise_batch(
    __global const double* query,
    __global const double* oct_params_a, // first set of octaves
    __global const int*   oct_perms_a,
    __global const double* oct_params_b, // second set
    __global const int*   oct_perms_b,
    __global double* results,
    const int N,
    const int numOctaves,
    const double persistence,
    const double normFactor,
    const double shift              // = 337.0/331.0
) {
    int id = get_global_id(0);
    if (id >= N) return;

    double qx = query[id*3+0];
    double qy = query[id*3+1];
    double qz = query[id*3+2];

    // First octave set
    double va = 0.0, freqXZ = 1.0, freqY = 1.0, amp = 1.0;
    for (int oct = 0; oct < numOctaves; oct++) {
        double xo = oct_params_a[oct*7+0];
        double yo = oct_params_a[oct*7+1];
        double zo = oct_params_a[oct*7+2];
        double lXZ = oct_params_a[oct*7+3];
        double lY  = oct_params_a[oct*7+4];
        __global const int* perm = oct_perms_a + oct * 256;
        va += mc_perlin_sample(perm, qx*freqXZ, qy*freqY, qz*freqXZ, xo, yo, zo, 0.0, 0.0) * amp;
        freqXZ *= lXZ; freqY *= lY; amp *= persistence;
    }

    // Second octave set (shifted)
    double vb = 0.0; freqXZ = 1.0; freqY = 1.0; amp = 1.0;
    for (int oct = 0; oct < numOctaves; oct++) {
        double xo = oct_params_b[oct*7+0];
        double yo = oct_params_b[oct*7+1];
        double zo = oct_params_b[oct*7+2];
        double lXZ = oct_params_b[oct*7+3];
        double lY  = oct_params_b[oct*7+4];
        __global const int* perm = oct_perms_b + oct * 256;
        double sx = qx*freqXZ + shift, sy = qy*freqY + shift, sz = qz*freqXZ + shift;
        vb += mc_perlin_sample(perm, sx, sy, sz, xo, yo, zo, 0.0, 0.0) * amp;
        freqXZ *= lXZ; freqY *= lY; amp *= persistence;
    }

    results[id] = (va + vb) * normFactor;
}
