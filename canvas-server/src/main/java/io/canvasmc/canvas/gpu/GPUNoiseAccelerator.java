package io.canvasmc.canvas.gpu;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicLong;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import net.minecraft.world.level.levelgen.synth.NormalNoise;
import net.minecraft.world.level.levelgen.synth.PerlinNoise;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Canvas GPU Noise Accelerator — OpenCL-based batch evaluation of MC NormalNoise.
 *
 * Activation: GPU path activates when chunk gen queue depth >= BATCH_THRESHOLD.
 * Fallback: any GPU error disables the path permanently for this session and logs the cause.
 * Startup test: verifies GPU computes finite values before enabling.
 */
public final class GPUNoiseAccelerator {

    private static final Logger LOGGER = LogManager.getLogger("CanvasGPU");
    private static final boolean GPU_ENABLED = !Boolean.getBoolean("canvas.gpu.noise.disabled");

    // Canvas - auto-tune to the measured GPU crossover (startup benchmark) when unset; -Dcanvas.gpu.noise.batch_threshold=N to pin.
    private static final Integer BATCH_THRESHOLD_OVERRIDE = Integer.getInteger("canvas.gpu.noise.batch_threshold");
    public static volatile int BATCH_THRESHOLD = BATCH_THRESHOLD_OVERRIDE != null ? BATCH_THRESHOLD_OVERRIDE : 8;

    private static volatile boolean initialized = false;
    private static volatile boolean available   = false;
    private static volatile long    clContext    = 0L;
    private static volatile long    clQueue      = 0L;
    private static volatile long    kernelPerlinV2 = 0L;
    private static volatile long    kernelNormalV2 = 0L;

    // Log GPU usage at most once per 5 seconds to avoid spam
    private static final AtomicLong lastGpuLogMs = new AtomicLong(0);

    // Minimum output.length for GPU dispatch — from benchmark crossover (N=512)
    // Below this, GPU overhead (buffer copy + launch) exceeds computation savings
    // Canvas - auto-tune to the measured crossover when unset; -Dcanvas.gpu.noise.min_n=N to pin.
    private static final Integer MIN_GPU_N_OVERRIDE = Integer.getInteger("canvas.gpu.noise.min_n");
    public static volatile int MIN_GPU_N = MIN_GPU_N_OVERRIDE != null ? MIN_GPU_N_OVERRIDE : 512;

    // GPU dispatch is NOT thread-safe (single cl_command_queue)
    // tryLock: if another region thread is using GPU, fall back to CPU
    private static final java.util.concurrent.locks.ReentrantLock GPU_LOCK =
        new java.util.concurrent.locks.ReentrantLock();

    // OpenCL flags
    private static final int CL_SUCCESS            = 0;
    private static final int CL_DEVICE_TYPE_GPU    = 4;
    private static final int CL_MEM_READ_ONLY      = 0x4;
    private static final int CL_MEM_WRITE_ONLY     = 0x2;
    private static final int CL_MEM_COPY_HOST_PTR  = 0x20;
    private static final int CL_TRUE               = 1;
    private static final int CL_PROGRAM_BUILD_LOG  = 0x1183;

    private static MethodHandle clGetPlatformIDs;
    private static MethodHandle clGetDeviceIDs;
    private static MethodHandle clCreateContext;
    private static MethodHandle clCreateCommandQueue;
    private static MethodHandle clCreateProgramWithSource;
    private static MethodHandle clBuildProgram;
    private static MethodHandle clCreateKernel;
    private static MethodHandle clCreateBuffer;
    private static MethodHandle clSetKernelArg;
    private static MethodHandle clEnqueueWriteBuffer;
    private static MethodHandle clEnqueueNDRangeKernel;
    private static MethodHandle clEnqueueReadBuffer;
    private static MethodHandle clFinish;
    private static MethodHandle clReleaseMemObject;
    private static MethodHandle clGetProgramBuildInfo;
    private static MethodHandle clReleaseKernel;
    private static MethodHandle clReleaseProgram;
    private static MethodHandle clReleaseCommandQueue;
    private static MethodHandle clReleaseContext;

    // Cached device pointer for benchmarks
    private static MemorySegment cachedDevice;

    // ── Canvas GPU World Gen: accessors for GPUWorldGenContext ──────────────────
    public static long canvas$clContext()                           { return clContext; }
    public static long canvas$clDevice()                           { return cachedDevice != null ? cachedDevice.address() : 0L; }
    public static MethodHandle canvas$mhCreateCommandQueue()       { return clCreateCommandQueue; }
    public static MethodHandle canvas$mhCreateProgramWithSource()  { return clCreateProgramWithSource; }
    public static MethodHandle canvas$mhBuildProgram()             { return clBuildProgram; }
    public static MethodHandle canvas$mhGetProgramBuildInfo()      { return clGetProgramBuildInfo; }
    public static MethodHandle canvas$mhCreateKernel()             { return clCreateKernel; }
    public static MethodHandle canvas$mhCreateBuffer()             { return clCreateBuffer; }
    public static MethodHandle canvas$mhSetKernelArg()             { return clSetKernelArg; }
    public static MethodHandle canvas$mhEnqueueWriteBuffer()       { return clEnqueueWriteBuffer; }
    public static MethodHandle canvas$mhEnqueueNDRangeKernel()     { return clEnqueueNDRangeKernel; }
    public static MethodHandle canvas$mhEnqueueReadBuffer()        { return clEnqueueReadBuffer; }
    public static MethodHandle canvas$mhFinish()                   { return clFinish; }
    public static MethodHandle canvas$mhReleaseMemObject()         { return clReleaseMemObject; }
    public static MethodHandle canvas$mhReleaseKernel()            { return clReleaseKernel; }
    public static MethodHandle canvas$mhReleaseProgram()           { return clReleaseProgram; }
    public static MethodHandle canvas$mhReleaseCommandQueue()      { return clReleaseCommandQueue; }

    public static boolean isAvailable() {
        ensureInit();
        return available;
    }

    public static synchronized void ensureInit() {
        if (initialized) return;
        initialized = true;
        if (!GPU_ENABLED) {
            LOGGER.info("[CanvasGPU] Disabled via -Dcanvas.gpu.noise.disabled=true");
            return;
        }
        try {
            initOpenCL();
        } catch (Throwable t) {
            LOGGER.warn("[CanvasGPU] Init failed, GPU path disabled: {}", t.getMessage());
        }
    }

    /** Called on server startup: smoke-test the GPU with known inputs. */
    public static void runStartupTest() {
        ensureInit();
        if (!available) {
            LOGGER.info("[CanvasGPU] Skipping startup test (GPU unavailable)");
            return;
        }
        try {
            // Run perlin_v2 with N=8 dummy points — just verify no crash + finite results
            int N = 8;
            int numOctaves = 1;
            double[] query = new double[N * 3];
            for (int i = 0; i < N; i++) {
                query[i*3]   = i * 1.5;
                query[i*3+1] = i * 0.7;
                query[i*3+2] = i * 2.3;
            }
            double[] octParams = new double[numOctaves * 6];
            // Single octave: xo=0, yo=0, zo=0, freq=1.0, amp=1.0, pad=0
            octParams[3] = 1.0;
            octParams[4] = 1.0;
            int[] octPerms = new int[numOctaves * 256];
            // Identity permutation
            for (int i = 0; i < 256; i++) octPerms[i] = i;

            double[] results = runPerlinV2(query, octParams, octPerms, N, numOctaves);

            boolean allFinite = true;
            for (double r : results) {
                if (!Double.isFinite(r)) { allFinite = false; break; }
            }
            if (!allFinite) throw new RuntimeException("GPU startup test: non-finite results");

            LOGGER.info("[CanvasGPU] Startup test PASSED — GPU computing finite noise values (N={}, batch_threshold={})", N, BATCH_THRESHOLD);
            LOGGER.info("[CanvasGPU] sample[0]={} sample[7]={}", String.format("%.6f", results[0]), String.format("%.6f", results[7]));
        } catch (Throwable t) {
            disableOnError("Startup test", t);
        }
    }

    public static void disableOnError(String context, Throwable t) {
        available = false;
        LOGGER.error("[CanvasGPU] Disabled after error in {}: {}", context, t.getMessage());
        if (LOGGER.isDebugEnabled()) LOGGER.debug("[CanvasGPU] Stack trace:", t);
    }

    // ── Benchmark ──────────────────────────────────────────────────────────────

    /**
     * GPU vs CPU benchmark across batch sizes.
     * Returns a formatted table string. Does NOT modify BATCH_THRESHOLD.
     * Call from startup after runStartupTest().
     */
    public static String runBenchmark() {
        if (!available) return "[CanvasGPU] Benchmark skipped: GPU unavailable";

        int[] batchSizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
        int warmup = 3, reps = 10;
        int numOctaves = 8; // typical MC terrain noise octave count

        // Build octave params (8 octaves, all with amplitude)
        double[] octParams = buildDummyOctParams(numOctaves);
        int[] octPerms = buildDummyPerms(numOctaves);

        StringBuilder sb = new StringBuilder();
        sb.append("[CanvasGPU] Benchmark: MC NormalNoise (N octaves=").append(numOctaves).append(")\n");
        sb.append(String.format("  %-8s  %-12s  %-12s  %-10s%n", "N", "CPU (ns/pt)", "GPU (ns/pt)", "ratio"));
        sb.append("  " + "-".repeat(48) + "\n");

        int crossover = -1;

        for (int N : batchSizes) {
            double[] query = buildDummyQuery(N);

            // CPU: direct Java loop
            long cpuTotalNs = 0;
            for (int r = 0; r < warmup + reps; r++) {
                long t0 = System.nanoTime();
                double[] cpuOut = new double[N];
                for (int i = 0; i < N; i++) {
                    double x = query[i*3], y = query[i*3+1], z = query[i*3+2];
                    cpuOut[i] = evalCpuNoise(octParams, octPerms, numOctaves, x, y, z);
                }
                long elapsed = System.nanoTime() - t0;
                if (r >= warmup) cpuTotalNs += elapsed;
            }
            double cpuNsPerPt = (double) cpuTotalNs / (reps * N);

            // GPU
            long gpuTotalNs = 0;
            boolean gpuOk = true;
            for (int r = 0; r < warmup + reps; r++) {
                try {
                    long t0 = System.nanoTime();
                    runPerlinV2(query, octParams, octPerms, N, numOctaves);
                    long elapsed = System.nanoTime() - t0;
                    if (r >= warmup) gpuTotalNs += elapsed;
                } catch (Throwable t) {
                    gpuOk = false;
                    break;
                }
            }

            if (!gpuOk) {
                sb.append(String.format("  %-8d  %-12.1f  %-12s  %s%n", N, cpuNsPerPt, "FAILED", ""));
                continue;
            }
            double gpuNsPerPt = (double) gpuTotalNs / (reps * N);
            double ratio = cpuNsPerPt / gpuNsPerPt;
            String faster = ratio >= 1.0 ? "GPU " + String.format("%.1fx", ratio) : "CPU " + String.format("%.1fx", 1.0/ratio);

            sb.append(String.format("  %-8d  %-12.1f  %-12.1f  %s%n", N, cpuNsPerPt, gpuNsPerPt, faster));

            if (ratio >= 1.0 && crossover < 0) crossover = N;
        }

        if (crossover > 0) {
            // Canvas - auto-tune the GPU-dispatch gates to the measured crossover unless explicitly pinned
            if (BATCH_THRESHOLD_OVERRIDE == null) BATCH_THRESHOLD = crossover;
            if (MIN_GPU_N_OVERRIDE == null) MIN_GPU_N = crossover;
            sb.append(String.format("  → GPU crossover: N >= %d (GPU faster above this batch size)%n", crossover));
            sb.append(String.format("  → %s BATCH_THRESHOLD=%d, MIN_GPU_N=%d%n",
                BATCH_THRESHOLD_OVERRIDE == null ? "auto-tuned to crossover:" : "pinned:", BATCH_THRESHOLD, MIN_GPU_N));
        } else {
            sb.append("  → GPU did not outperform CPU in tested range. Consider disabling GPU noise path.\n");
        }
        return sb.toString();
    }

    private static double[] buildDummyQuery(int N) {
        double[] q = new double[N * 3];
        for (int i = 0; i < N; i++) {
            q[i*3]   = i * 1.234;
            q[i*3+1] = i * 0.567;
            q[i*3+2] = i * 2.345;
        }
        return q;
    }

    private static double[] buildDummyOctParams(int numOctaves) {
        double[] p = new double[numOctaves * 6];
        double freq = 1.0 / (1 << (numOctaves - 1));
        double amp  = 1.0 / (2 * (1 - Math.pow(0.5, numOctaves)));
        for (int i = 0; i < numOctaves; i++) {
            p[i*6]   = i * 0.1;  // xo
            p[i*6+1] = i * 0.2;  // yo
            p[i*6+2] = i * 0.3;  // zo
            p[i*6+3] = freq;     // freqFactor
            p[i*6+4] = amp;      // effectiveAmplitude
            freq *= 2.0;
            amp  /= 2.0;
        }
        return p;
    }

    private static int[] buildDummyPerms(int numOctaves) {
        int[] perms = new int[numOctaves * 256];
        for (int oct = 0; oct < numOctaves; oct++) {
            for (int i = 0; i < 256; i++) perms[oct*256 + i] = (i + oct * 37) & 0xFF;
        }
        return perms;
    }

    private static double evalCpuNoise(double[] octParams, int[] octPerms, int numOctaves, double x, double y, double z) {
        double val = 0.0;
        for (int i = 0; i < numOctaves; i++) {
            double amp  = octParams[i*6+4];
            if (amp == 0.0) continue;
            double freq = octParams[i*6+3];
            double xo   = octParams[i*6+0], yo = octParams[i*6+1], zo = octParams[i*6+2];
            val += amp * cpuPerlin(octPerms, i * 256, x*freq + xo, y*freq + yo, z*freq + zo);
        }
        return val;
    }

    private static double cpuPerlin(int[] perm, int base, double x, double y, double z) {
        double xf = Math.floor(x), yf = Math.floor(y), zf = Math.floor(z);
        double xr = x - xf, yr = y - yf, zr = z - zf;
        int xi = (int)xf & 0xFF, yi = (int)yf & 0xFF, zi = (int)zf & 0xFF;
        int a  = perm[base + xi] & 0xFF;
        int b  = perm[base + (xi+1) & 0xFF] & 0xFF;
        int aa = perm[base + (a+yi) & 0xFF] & 0xFF;
        int ab = perm[base + (a+yi+1) & 0xFF] & 0xFF;
        int ba = perm[base + (b+yi) & 0xFF] & 0xFF;
        int bb = perm[base + (b+yi+1) & 0xFF] & 0xFF;
        int[] GIDX = {aa+zi, ba+zi, ab+zi, bb+zi, aa+zi+1, ba+zi+1, ab+zi+1, bb+zi+1};
        double[] GX = {xr, xr-1, xr, xr-1, xr, xr-1, xr, xr-1};
        double[] GY = {yr, yr, yr-1, yr-1, yr, yr, yr-1, yr-1};
        double[] GZ = {zr, zr, zr, zr, zr-1, zr-1, zr-1, zr-1};
        int[][] GRAD = {{1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},{1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
                        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1},{1,1,0},{0,-1,1},{-1,1,0},{0,-1,-1}};
        double[] g = new double[8];
        for (int k = 0; k < 8; k++) {
            int[] gv = GRAD[perm[base + GIDX[k] & 0xFF] & 15];
            g[k] = gv[0]*GX[k] + gv[1]*GY[k] + gv[2]*GZ[k];
        }
        double u = fade(xr), v = fade(yr), w = fade(zr);
        return lerp(w, lerp(v, lerp(u, g[0], g[1]), lerp(u, g[2], g[3])),
                       lerp(v, lerp(u, g[4], g[5]), lerp(u, g[6], g[7])));
    }

    private static double fade(double t) { return t*t*t*(t*(t*6-15)+10); }
    private static double lerp(double t, double a, double b) { return a + t*(b-a); }

    // ── DensityFunctions.Noise integration ────────────────────────────────────

    /**
     * GPU replacement for DensityFunctions.Noise.fillArray().
     * Collects coordinates from contextProvider, evaluates NormalNoise on GPU.
     *
     * Requirements:
     *   - N >= MIN_GPU_N (512): below crossover, GPU overhead exceeds savings
     *   - GPU_LOCK acquired: cl_command_queue is not thread-safe
     */
    public static void fillNormalNoise(
        double[] output,
        DensityFunction.ContextProvider contextProvider,
        NormalNoise normalNoise,
        double xzScale,
        double yScale
    ) throws Throwable {
        int N = output.length;
        if (N < MIN_GPU_N) throw new IllegalStateException("N=" + N + " below MIN_GPU_N=" + MIN_GPU_N);
        if (!GPU_LOCK.tryLock()) throw new IllegalStateException("GPU busy");
        try {
        double[] query = new double[N * 3];
        for (int i = 0; i < N; i++) {
            DensityFunction.FunctionContext ctx = contextProvider.forIndex(i);
            query[i*3]   = ctx.blockX() * xzScale;
            query[i*3+1] = ctx.blockY() * yScale;
            query[i*3+2] = ctx.blockZ() * xzScale;
        }

        PerlinNoise first  = normalNoise.canvas$getFirst();
        PerlinNoise second = normalNoise.canvas$getSecond();
        double normFactor  = normalNoise.canvas$getValueFactor();
        double inputFactor = NormalNoise.canvas$getInputFactor();

        double[] octA   = buildOctParams(first,  1.0);
        int[]    permA  = buildPermBuffer(first);
        double[] octB   = buildOctParams(second, inputFactor);
        int[]    permB  = buildPermBuffer(second);

        int numA = octA.length / 6;
        int numB = octB.length / 6;

        long t0 = System.nanoTime();
        double[] results = runNormalNoiseV2(query, octA, permA, numA, octB, permB, numB, N, normFactor);
        long ms = (System.nanoTime() - t0) / 1_000_000;

        System.arraycopy(results, 0, output, 0, N);

        // Throttled log
        long now = System.currentTimeMillis();
        long last = lastGpuLogMs.get();
        if (now - last > 5000 && lastGpuLogMs.compareAndSet(last, now)) {
            LOGGER.info("[CanvasGPU] GPU chunk noise: N={} pts, numA={} numB={} octaves, {}ms (queue depth={})",
                N, numA, numB, ms, GPUChunkGenQueue.getDepth());
        }
        } finally {
            GPU_LOCK.unlock();
        }
    }

    // ── Low-level GPU dispatch ─────────────────────────────────────────────────

    /** Builds per-octave params [xo,yo,zo,freqFactor,effectiveAmplitude,0] for GPU kernel. */
    private static double[] buildOctParams(PerlinNoise pn, double inputScale) {
        ImprovedNoise[] levels = pn.canvas$getNoiseLevels();
        double[] amps   = pn.canvas$getAmplitudesArray();
        double freq0    = pn.canvas$getLowestFreqInputFactor();
        double valFac   = pn.canvas$getLowestFreqValueFactor();
        int n = levels.length;
        double[] params = new double[n * 6];
        double freq = freq0 * inputScale;
        double vf   = valFac;
        for (int i = 0; i < n; i++) {
            ImprovedNoise lvl = levels[i];
            if (lvl != null && amps[i] != 0.0) {
                params[i*6]   = lvl.xo;
                params[i*6+1] = lvl.yo;
                params[i*6+2] = lvl.zo;
                params[i*6+3] = freq;
                params[i*6+4] = amps[i] * vf;
            }
            // else amplitude stays 0.0 → kernel skips via `if (amp==0) continue`
            freq *= 2.0;
            vf   /= 2.0;
        }
        return params;
    }

    /** Builds concatenated perm tables as int[] from PerlinNoise octaves. */
    private static int[] buildPermBuffer(PerlinNoise pn) {
        ImprovedNoise[] levels = pn.canvas$getNoiseLevels();
        int n = levels.length;
        int[] buf = new int[n * 256];
        for (int i = 0; i < n; i++) {
            ImprovedNoise lvl = levels[i];
            if (lvl != null) {
                byte[] p = lvl.getPermTable();
                for (int j = 0; j < 256; j++) buf[i*256 + j] = p[j] & 0xFF;
            }
            // else leave as 0 (amplitude is also 0, kernel skips)
        }
        return buf;
    }

    /** Run perlin_noise_v2 kernel. Used for benchmark and startup test. */
    static double[] runPerlinV2(double[] query, double[] octParams, int[] octPerms, int N, int numOctaves) throws Throwable {
        double[] out = new double[N];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment q   = seg(arena, query);
            MemorySegment op  = seg(arena, octParams);
            MemorySegment opm = segI(arena, octPerms);
            MemorySegment res = arena.allocate(ValueLayout.JAVA_DOUBLE, N);

            MemorySegment ctxSeg = MemorySegment.ofAddress(clContext);
            MemorySegment qSeg   = MemorySegment.ofAddress(clQueue);
            MemorySegment kSeg   = MemorySegment.ofAddress(kernelPerlinV2);
            MemorySegment clErr  = arena.allocate(ValueLayout.JAVA_INT);

            MemorySegment bq   = clBuf(arena, ctxSeg, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  q.byteSize(),   q,   clErr);
            MemorySegment bop  = clBuf(arena, ctxSeg, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  op.byteSize(),  op,  clErr);
            MemorySegment bopm = clBuf(arena, ctxSeg, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  opm.byteSize(), opm, clErr);
            MemorySegment bres = clBuf(arena, ctxSeg, CL_MEM_WRITE_ONLY,                        (long)N * 8,    MemorySegment.NULL, clErr);

            setArg(arena, kSeg, 0, bq);
            setArg(arena, kSeg, 1, bop);
            setArg(arena, kSeg, 2, bopm);
            setArg(arena, kSeg, 3, bres);
            setIntArg(arena, kSeg, 4, N);
            setIntArg(arena, kSeg, 5, numOctaves);

            dispatch(arena, qSeg, kSeg, N);

            clEnqueueReadBuffer.invoke(qSeg, bres, CL_TRUE, 0L, (long)N*8, res, 0, MemorySegment.NULL, MemorySegment.NULL);
            clFinish.invoke(qSeg);
            for (int i = 0; i < N; i++) out[i] = res.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            relBufs(bq, bop, bopm, bres);
        }
        return out;
    }

    /** Run normal_noise_v2 kernel for NormalNoise batch evaluation. */
    private static double[] runNormalNoiseV2(
        double[] query,
        double[] octA, int[] permA, int numA,
        double[] octB, int[] permB, int numB,
        int N, double normFactor
    ) throws Throwable {
        double[] out = new double[N];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment q    = seg(arena, query);
            MemorySegment oA   = seg(arena, octA);
            MemorySegment pA   = segI(arena, permA);
            MemorySegment oB   = seg(arena, octB);
            MemorySegment pB   = segI(arena, permB);
            MemorySegment res  = arena.allocate(ValueLayout.JAVA_DOUBLE, N);

            MemorySegment ctxSeg = MemorySegment.ofAddress(clContext);
            MemorySegment qSeg   = MemorySegment.ofAddress(clQueue);
            MemorySegment kSeg   = MemorySegment.ofAddress(kernelNormalV2);
            MemorySegment clErr  = arena.allocate(ValueLayout.JAVA_INT);

            MemorySegment bq   = clBuf(arena, ctxSeg, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, q.byteSize(),   q,   clErr);
            MemorySegment boA  = clBuf(arena, ctxSeg, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, oA.byteSize(),  oA,  clErr);
            MemorySegment bpA  = clBuf(arena, ctxSeg, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pA.byteSize(),  pA,  clErr);
            MemorySegment boB  = clBuf(arena, ctxSeg, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, oB.byteSize(),  oB,  clErr);
            MemorySegment bpB  = clBuf(arena, ctxSeg, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pB.byteSize(),  pB,  clErr);
            MemorySegment bres = clBuf(arena, ctxSeg, CL_MEM_WRITE_ONLY, (long)N * 8, MemorySegment.NULL,       clErr);

            setArg(arena, kSeg, 0, bq);
            setArg(arena, kSeg, 1, boA);
            setArg(arena, kSeg, 2, bpA);
            setIntArg(arena, kSeg, 3, numA);
            setArg(arena, kSeg, 4, boB);
            setArg(arena, kSeg, 5, bpB);
            setIntArg(arena, kSeg, 6, numB);
            setArg(arena, kSeg, 7, bres);
            setIntArg(arena, kSeg, 8, N);
            setDoubleArg(arena, kSeg, 9, normFactor);

            dispatch(arena, qSeg, kSeg, N);

            clEnqueueReadBuffer.invoke(qSeg, bres, CL_TRUE, 0L, (long)N*8, res, 0, MemorySegment.NULL, MemorySegment.NULL);
            clFinish.invoke(qSeg);
            for (int i = 0; i < N; i++) out[i] = res.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            relBufs(bq, boA, bpA, boB, bpB, bres);
        }
        return out;
    }

    // ── OpenCL helpers ─────────────────────────────────────────────────────────

    private static MemorySegment seg(Arena a, double[] arr) {
        MemorySegment s = a.allocate(ValueLayout.JAVA_DOUBLE, arr.length);
        for (int i = 0; i < arr.length; i++) s.setAtIndex(ValueLayout.JAVA_DOUBLE, i, arr[i]);
        return s;
    }

    private static MemorySegment segI(Arena a, int[] arr) {
        MemorySegment s = a.allocate(ValueLayout.JAVA_INT, arr.length);
        for (int i = 0; i < arr.length; i++) s.setAtIndex(ValueLayout.JAVA_INT, i, arr[i]);
        return s;
    }

    private static MemorySegment clBuf(Arena a, MemorySegment ctx, int flags, long size, MemorySegment hostPtr, MemorySegment errOut) throws Throwable {
        return (MemorySegment) clCreateBuffer.invoke(ctx, (long) flags, size, hostPtr, errOut);
    }

    private static void setArg(Arena a, MemorySegment k, int idx, MemorySegment buf) throws Throwable {
        MemorySegment argSlot = a.allocate(ValueLayout.ADDRESS);
        argSlot.set(ValueLayout.ADDRESS, 0, buf);
        clSetKernelArg.invoke(k, idx, (long) ValueLayout.ADDRESS.byteSize(), argSlot);
    }

    private static void setIntArg(Arena a, MemorySegment k, int idx, int val) throws Throwable {
        MemorySegment slot = a.allocate(ValueLayout.JAVA_INT);
        slot.set(ValueLayout.JAVA_INT, 0, val);
        clSetKernelArg.invoke(k, idx, (long) ValueLayout.JAVA_INT.byteSize(), slot);
    }

    private static void setDoubleArg(Arena a, MemorySegment k, int idx, double val) throws Throwable {
        MemorySegment slot = a.allocate(ValueLayout.JAVA_DOUBLE);
        slot.set(ValueLayout.JAVA_DOUBLE, 0, val);
        clSetKernelArg.invoke(k, idx, (long) ValueLayout.JAVA_DOUBLE.byteSize(), slot);
    }

    private static void dispatch(Arena a, MemorySegment queue, MemorySegment kernel, int N) throws Throwable {
        MemorySegment globalSize = a.allocate(ValueLayout.JAVA_LONG, 1);
        globalSize.setAtIndex(ValueLayout.JAVA_LONG, 0, N);
        clEnqueueNDRangeKernel.invoke(queue, kernel, 1, MemorySegment.NULL, globalSize, MemorySegment.NULL, 0, MemorySegment.NULL, MemorySegment.NULL);
    }

    private static void relBufs(MemorySegment... bufs) throws Throwable {
        for (MemorySegment buf : bufs) clReleaseMemObject.invoke(buf);
    }

    // ── OpenCL init ────────────────────────────────────────────────────────────

    private static void initOpenCL() throws Throwable {
        Linker linker = Linker.nativeLinker();
        SymbolLookup ocl = SymbolLookup.libraryLookup("libOpenCL.so.1", Arena.global());

        clGetPlatformIDs = mh(linker, ocl, "clGetPlatformIDs",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clGetDeviceIDs = mh(linker, ocl, "clGetDeviceIDs",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clCreateContext = mh(linker, ocl, "clCreateContext",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clCreateCommandQueue = mh(linker, ocl, "clCreateCommandQueue",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
        clCreateProgramWithSource = mh(linker, ocl, "clCreateProgramWithSource",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clBuildProgram = mh(linker, ocl, "clBuildProgram",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clCreateKernel = mh(linker, ocl, "clCreateKernel",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clCreateBuffer = mh(linker, ocl, "clCreateBuffer",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clSetKernelArg = mh(linker, ocl, "clSetKernelArg",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
        clEnqueueWriteBuffer = mh(linker, ocl, "clEnqueueWriteBuffer",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clEnqueueNDRangeKernel = mh(linker, ocl, "clEnqueueNDRangeKernel",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clEnqueueReadBuffer = mh(linker, ocl, "clEnqueueReadBuffer",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clFinish = mh(linker, ocl, "clFinish",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        clReleaseMemObject = mh(linker, ocl, "clReleaseMemObject",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        clGetProgramBuildInfo = mh(linker, ocl, "clGetProgramBuildInfo",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clReleaseKernel = mh(linker, ocl, "clReleaseKernel",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        clReleaseProgram = mh(linker, ocl, "clReleaseProgram",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        clReleaseCommandQueue = mh(linker, ocl, "clReleaseCommandQueue",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        clReleaseContext = mh(linker, ocl, "clReleaseContext",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        try (Arena arena = Arena.ofConfined()) {
            // Find GPU platform + device
            MemorySegment numPlatforms = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment platforms    = arena.allocate(ValueLayout.ADDRESS, 8);
            int err = (int) clGetPlatformIDs.invoke(8, platforms, numPlatforms);
            if (err != CL_SUCCESS) throw new RuntimeException("clGetPlatformIDs: " + err);
            int np = numPlatforms.get(ValueLayout.JAVA_INT, 0);
            if (np == 0) throw new RuntimeException("No OpenCL platforms found");

            MemorySegment device = null, platform = null;
            for (int pi = 0; pi < np; pi++) {
                MemorySegment plat     = platforms.getAtIndex(ValueLayout.ADDRESS, pi);
                MemorySegment devCount = arena.allocate(ValueLayout.JAVA_INT);
                MemorySegment devs     = arena.allocate(ValueLayout.ADDRESS, 4);
                int e = (int) clGetDeviceIDs.invoke(plat, (long) CL_DEVICE_TYPE_GPU, 4, devs, devCount);
                if (e == CL_SUCCESS && devCount.get(ValueLayout.JAVA_INT, 0) > 0) {
                    device   = devs.getAtIndex(ValueLayout.ADDRESS, 0);
                    platform = plat;
                    break;
                }
            }
            if (device == null) throw new RuntimeException("No GPU device found");
            cachedDevice = device;

            MemorySegment ctxErr = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment devArr = arena.allocate(ValueLayout.ADDRESS, 1);
            devArr.setAtIndex(ValueLayout.ADDRESS, 0, device);
            MemorySegment ctx = (MemorySegment) clCreateContext.invoke(
                MemorySegment.NULL, 1, devArr, MemorySegment.NULL, MemorySegment.NULL, ctxErr);
            if (ctxErr.get(ValueLayout.JAVA_INT, 0) != CL_SUCCESS)
                throw new RuntimeException("clCreateContext: " + ctxErr.get(ValueLayout.JAVA_INT, 0));

            MemorySegment qErr  = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment queue = (MemorySegment) clCreateCommandQueue.invoke(ctx, device, 0L, qErr);
            if (qErr.get(ValueLayout.JAVA_INT, 0) != CL_SUCCESS)
                throw new RuntimeException("clCreateCommandQueue: " + qErr.get(ValueLayout.JAVA_INT, 0));

            // Load + build kernel source
            String kernelSrc;
            try (var in = GPUNoiseAccelerator.class.getClassLoader().getResourceAsStream("opencl/mc_noise.cl")) {
                if (in == null) throw new RuntimeException("mc_noise.cl not found in classpath");
                kernelSrc = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            }
            byte[] srcBytes = kernelSrc.getBytes(StandardCharsets.UTF_8);
            MemorySegment srcMem = arena.allocateFrom(kernelSrc, StandardCharsets.UTF_8);
            MemorySegment srcArr = arena.allocate(ValueLayout.ADDRESS, 1);
            srcArr.setAtIndex(ValueLayout.ADDRESS, 0, srcMem);
            MemorySegment lenArr = arena.allocate(ValueLayout.JAVA_LONG, 1);
            lenArr.setAtIndex(ValueLayout.JAVA_LONG, 0, srcBytes.length);
            MemorySegment prgErr = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment program = (MemorySegment) clCreateProgramWithSource.invoke(ctx, 1, srcArr, lenArr, prgErr);
            if (prgErr.get(ValueLayout.JAVA_INT, 0) != CL_SUCCESS)
                throw new RuntimeException("clCreateProgramWithSource: " + prgErr.get(ValueLayout.JAVA_INT, 0));

            int buildErr = (int) clBuildProgram.invoke(program, 1, devArr, MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL);
            if (buildErr != CL_SUCCESS) {
                MemorySegment logLen = arena.allocate(ValueLayout.JAVA_LONG);
                clGetProgramBuildInfo.invoke(program, device, CL_PROGRAM_BUILD_LOG, 0L, MemorySegment.NULL, logLen);
                long ll = logLen.get(ValueLayout.JAVA_LONG, 0);
                MemorySegment log = arena.allocate(ll);
                clGetProgramBuildInfo.invoke(program, device, CL_PROGRAM_BUILD_LOG, ll, log, MemorySegment.NULL);
                throw new RuntimeException("clBuildProgram failed:\n" + log.getString(0, StandardCharsets.UTF_8));
            }

            MemorySegment kErr = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment kP2  = (MemorySegment) clCreateKernel.invoke(
                program, arena.allocateFrom("perlin_noise_v2", StandardCharsets.UTF_8), kErr);
            if (kErr.get(ValueLayout.JAVA_INT, 0) != CL_SUCCESS)
                throw new RuntimeException("clCreateKernel perlin_noise_v2: " + kErr.get(ValueLayout.JAVA_INT, 0));
            MemorySegment kN2  = (MemorySegment) clCreateKernel.invoke(
                program, arena.allocateFrom("normal_noise_v2", StandardCharsets.UTF_8), kErr);
            if (kErr.get(ValueLayout.JAVA_INT, 0) != CL_SUCCESS)
                throw new RuntimeException("clCreateKernel normal_noise_v2: " + kErr.get(ValueLayout.JAVA_INT, 0));

            clContext      = ctx.address();
            clQueue        = queue.address();
            kernelPerlinV2 = kP2.address();
            kernelNormalV2 = kN2.address();
        }

        available = true;
        LOGGER.info("[CanvasGPU] OpenCL initialized. Kernels: perlin_noise_v2, normal_noise_v2. Batch threshold: {} chunks", BATCH_THRESHOLD);
    }

    private static MethodHandle mh(Linker l, SymbolLookup lib, String name, FunctionDescriptor fd) {
        return l.downcallHandle(lib.find(name).orElseThrow(() -> new RuntimeException("Symbol not found: " + name)), fd);
    }

    public static String statusLine() {
        if (!initialized) return "not initialized";
        if (!available)   return "unavailable (no compatible GPU, OpenCL error, or startup test failed)";
        return String.format("active — kernels: perlin_noise_v2, normal_noise_v2 | batch_threshold=%d", BATCH_THRESHOLD);
    }
}
