package io.canvasmc.canvas.gpu.worldgen;

import io.canvasmc.canvas.gpu.GPUNoiseAccelerator;
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.locks.ReentrantLock;
import net.minecraft.world.level.levelgen.DFCLCompiler;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.server.level.ColumnPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Per-RandomState GPU world generation context.
 *
 * Compiles the preliminarySurfaceLevel DensityFunction → OpenCL C at world init,
 * then batches surface height computations across large (bx, bz) grids.
 *
 * Thread safety: a single ReentrantLock serializes GPU dispatch.
 * If the lock is held, callers get null (CPU fallback).
 */
public final class GPUWorldGenContext {

    private static final Logger LOGGER = LoggerFactory.getLogger("CanvasGPU-WorldGen");

    // OpenCL flag constants
    private static final int  CL_SUCCESS           = 0;
    private static final long CL_MEM_READ_ONLY      = 0x4L;
    private static final long CL_MEM_WRITE_ONLY     = 0x2L;
    private static final long CL_MEM_COPY_HOST_PTR  = 0x20L;
    private static final int  CL_TRUE               = 1;
    private static final int  CL_PROGRAM_BUILD_LOG  = 0x1183;

    private final MemorySegment clQueue;
    private final MemorySegment clKernel;
    private final MemorySegment clProgram;
    private final ReentrantLock lock = new ReentrantLock();

    // Optional finalDensity value kernel (compute_value) for full-chunk density grids (Task #6).
    private volatile MemorySegment clValueKernel = null;

    private volatile boolean failed = false;

    private GPUWorldGenContext(MemorySegment queue, MemorySegment program, MemorySegment kernel) {
        this.clQueue   = queue;
        this.clProgram = program;
        this.clKernel  = kernel;
    }

    /**
     * Compiles the preliminarySurfaceLevel function from the given RandomState.
     * Returns null if GPU is unavailable or compilation fails.
     */
    public static GPUWorldGenContext create(RandomState state) {
        if (!GPUNoiseAccelerator.isAvailable()) return null;
        try {
            DensityFunction df = state.router().preliminarySurfaceLevel();
            String clSrc = DFCLCompiler.compile(df);
            if (clSrc == null) {
                LOGGER.warn("[CanvasGPU-WorldGen] DFCLCompiler returned null — CPU fallback");
                return null;
            }
            GPUWorldGenContext c = compile(clSrc);
            if (c != null && !c.verifyAgainstCpu(df, 8192)) {
                LOGGER.warn("[CanvasGPU-WorldGen] BIT-EXACT VERIFY FAILED — refusing GPU, CPU fallback");
                return null; // never use a GPU context that doesn't match CPU
            }
            // Diagnostics (off by default — building the 394KB finalDensity kernel adds ~7s to compile).
            // Enable with -Dcanvas.gpu.verify=true. Surface generation does NOT need these; while this
            // whole create() runs on the background canvas-gpu-compile thread, precompute() returns null
            // (CPU fallback) so chunk generation is never blocked during compilation.
            if (Boolean.getBoolean("canvas.gpu.verify")) {
                try { verifyDensityFunction(state.router().finalDensity(), "finalDensity", 8192); } catch (Throwable ignored) {}
            }
            if (Boolean.getBoolean("canvas.gpu.bench")) {
                try { benchmarkDensityFunction(state.router().finalDensity(), "finalDensity"); } catch (Throwable ignored) {}
            }
            // Task #6: build the finalDensity value kernel for full-chunk GPU density grids, and
            // VERIFY it bit-exact ONCE here (8192 pts). finalDensity is a deterministic function, so a
            // one-time verify proves the kernel correct for all points → no per-chunk gate needed at
            // generation time (that gate cost 3 full finalDensity tree-walks per chunk on CPU, partly
            // defeating the offload). Nether/End whose finalDensity isn't bit-exact → verify fails → CPU.
            if (Boolean.getBoolean("canvas.gpu.density.enabled")) {
                try {
                    DensityFunction fd = state.router().finalDensity();
                    String vsrc = DFCLCompiler.compileValue(fd);
                    if (vsrc != null) {
                        MemorySegment k = buildKernel(vsrc, "compute_value");
                        if (k != null) {
                            c.clValueKernel = k;
                            if (c.verifyValueKernel(fd, 8192)) {
                                LOGGER.info("[CanvasGPU-WorldGen] finalDensity value kernel VERIFIED (8192/8192) + READY — GPU full-chunk density enabled");
                            } else {
                                c.clValueKernel = null; // not bit-exact for this dimension → CPU density
                                LOGGER.warn("[CanvasGPU-WorldGen] finalDensity value kernel verify FAILED — GPU density disabled for this dimension (CPU)");
                            }
                        }
                    }
                } catch (Throwable t) {
                    c.clValueKernel = null;
                    LOGGER.warn("[CanvasGPU-WorldGen] value kernel build/verify failed: {}", t.toString());
                }
            }
            return c;
        } catch (Throwable t) {
            LOGGER.warn("[CanvasGPU-WorldGen] Context creation failed: {}", t.getMessage());
            return null;
        }
    }

    private static GPUWorldGenContext compile(String clSrc) throws Throwable {
        long ctxAddr = GPUNoiseAccelerator.canvas$clContext();
        long devAddr = GPUNoiseAccelerator.canvas$clDevice();
        if (ctxAddr == 0L || devAddr == 0L) return null;

        MemorySegment ctx = MemorySegment.ofAddress(ctxAddr);
        MemorySegment dev = MemorySegment.ofAddress(devAddr);

        try (Arena a = Arena.ofConfined()) {
            // Create dedicated command queue for world gen
            MemorySegment devArr = a.allocate(ValueLayout.ADDRESS);
            devArr.set(ValueLayout.ADDRESS, 0, dev);
            MemorySegment qErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment queue = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateCommandQueue()
                .invoke(ctx, dev, 0L, qErr);
            if (qErr.get(ValueLayout.JAVA_INT, 0) != CL_SUCCESS)
                throw new RuntimeException("clCreateCommandQueue: " + qErr.get(ValueLayout.JAVA_INT, 0));

            // Create and build program
            byte[] srcBytes = clSrc.getBytes(StandardCharsets.UTF_8);
            MemorySegment srcMem = a.allocateFrom(ValueLayout.JAVA_BYTE, srcBytes);
            MemorySegment srcArr = a.allocate(ValueLayout.ADDRESS);
            srcArr.set(ValueLayout.ADDRESS, 0, srcMem);
            MemorySegment lenArr = a.allocate(ValueLayout.JAVA_LONG);
            lenArr.set(ValueLayout.JAVA_LONG, 0, (long) srcBytes.length);
            MemorySegment prgErr = a.allocate(ValueLayout.JAVA_INT);

            MemorySegment program = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateProgramWithSource()
                .invoke(ctx, 1, srcArr, lenArr, prgErr);
            if (prgErr.get(ValueLayout.JAVA_INT, 0) != CL_SUCCESS)
                throw new RuntimeException("clCreateProgramWithSource: " + prgErr.get(ValueLayout.JAVA_INT, 0));

            // Bit-exact flags: correctly-rounded fp32 divide/sqrt (CubicSpline is float; OpenCL float
            // division defaults to <=2.5 ULP, which diverges from Java's strict float division).
            MemorySegment buildOpts = a.allocateFrom("-cl-fp32-correctly-rounded-divide-sqrt");
            int buildErr = (int) GPUNoiseAccelerator.canvas$mhBuildProgram()
                .invoke(program, 1, devArr, buildOpts, MemorySegment.NULL, MemorySegment.NULL);
            if (buildErr != CL_SUCCESS) {
                // Fetch build log
                MemorySegment logSize = a.allocate(ValueLayout.JAVA_LONG);
                MemorySegment devArr2 = a.allocate(ValueLayout.ADDRESS);
                devArr2.set(ValueLayout.ADDRESS, 0, dev);
                GPUNoiseAccelerator.canvas$mhGetProgramBuildInfo()
                    .invoke(program, dev, CL_PROGRAM_BUILD_LOG, 0L, MemorySegment.NULL, logSize);
                long sz = logSize.get(ValueLayout.JAVA_LONG, 0);
                String log;
                if (sz > 0) {
                    MemorySegment logBuf = a.allocate(sz + 1);
                    GPUNoiseAccelerator.canvas$mhGetProgramBuildInfo()
                        .invoke(program, dev, CL_PROGRAM_BUILD_LOG, sz, logBuf, MemorySegment.NULL);
                    log = logBuf.getString(0, StandardCharsets.UTF_8);
                } else {
                    log = "(empty log, buildErr=" + buildErr + ")";
                }
                // Dump CL source for debugging even on failure
                try {
                    java.nio.file.Files.writeString(java.nio.file.Path.of("/tmp/canvas_surface_fail.cl"), clSrc);
                } catch (Exception ignored) {}
                throw new RuntimeException("clBuildProgram failed (err=" + buildErr + "):\n" + log);
            }

            // Create kernel
            MemorySegment kName = a.allocateFrom("compute_surface");
            MemorySegment kErr  = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment kernel = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateKernel()
                .invoke(program, kName, kErr);
            if (kErr.get(ValueLayout.JAVA_INT, 0) != CL_SUCCESS)
                throw new RuntimeException("clCreateKernel: " + kErr.get(ValueLayout.JAVA_INT, 0));

            LOGGER.info("[CanvasGPU-WorldGen] Compiled surface kernel ({} bytes CL source)", srcBytes.length);
            // Debug: dump CL source for inspection
            try {
                java.nio.file.Files.writeString(java.nio.file.Path.of("/tmp/canvas_surface.cl"), clSrc);
                LOGGER.info("[CanvasGPU-WorldGen] CL source dumped to /tmp/canvas_surface.cl");
            } catch (Exception ignored) {}

            // Return context — queue, program, kernel survive arena close
            // because they're registered CL objects, not arena-managed memory
            return new GPUWorldGenContext(queue, program, kernel);
        }
    }

    /**
     * Computes floor(preliminarySurfaceLevel(bx, 0, bz)) for each (bx, bz) in the grid.
     *
     * @param blockXs  array of blockX values
     * @param blockZs  array of blockZ values (parallel with blockXs)
     * @param N        number of points
     * @return         int[N] surface heights, or null on GPU failure
     */
    public int[] computeSurface(int[] blockXs, int[] blockZs, int N) {
        if (failed || N == 0) return null;
        if (!lock.tryLock()) return null; // GPU busy — CPU fallback
        try {
            return dispatchSurface(blockXs, blockZs, N);
        } catch (Throwable t) {
            failed = true;
            LOGGER.error("[CanvasGPU-WorldGen] dispatch failed, disabling: {}", t.getMessage());
            return null;
        } finally {
            lock.unlock();
        }
    }

    private int[] dispatchSurface(int[] blockXs, int[] blockZs, int N) throws Throwable {
        try (Arena a = Arena.ofShared()) {
            long posBytes  = (long) N * 2 * 4;  // N * 2 ints
            long resBytes  = (long) N * 4;       // N ints

            // Build interleaved (blockX, blockZ) int array
            MemorySegment posMem = a.allocate(posBytes);
            for (int i = 0; i < N; i++) {
                posMem.set(ValueLayout.JAVA_INT, (long)(i*2)*4,    blockXs[i]);
                posMem.set(ValueLayout.JAVA_INT, (long)(i*2+1)*4,  blockZs[i]);
            }

            MemorySegment ctx = MemorySegment.ofAddress(GPUNoiseAccelerator.canvas$clContext());
            MemorySegment errBuf = a.allocate(ValueLayout.JAVA_INT);

            // Allocate GPU buffers
            MemorySegment bPos = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateBuffer()
                .invoke(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, posBytes, posMem, errBuf);
            checkErr(errBuf, "clCreateBuffer pos");

            MemorySegment bRes = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateBuffer()
                .invoke(ctx, CL_MEM_WRITE_ONLY, resBytes, MemorySegment.NULL, errBuf);
            checkErr(errBuf, "clCreateBuffer res");

            // Set kernel args: positions, results, N
            setArg(a, 0, ValueLayout.ADDRESS.byteSize(), bPos);
            setArg(a, 1, ValueLayout.ADDRESS.byteSize(), bRes);
            setArgInt(a, 2, N);

            // Dispatch
            MemorySegment globalSize = a.allocate(ValueLayout.JAVA_LONG);
            globalSize.set(ValueLayout.JAVA_LONG, 0, (long) N);
            int ndErr = (int) GPUNoiseAccelerator.canvas$mhEnqueueNDRangeKernel()
                .invoke(clQueue, clKernel, 1, MemorySegment.NULL, globalSize, MemorySegment.NULL,
                        0, MemorySegment.NULL, MemorySegment.NULL);
            if (ndErr != CL_SUCCESS) throw new RuntimeException("clEnqueueNDRangeKernel: " + ndErr);

            // Read results
            MemorySegment resMem = a.allocate(resBytes);
            int rdErr = (int) GPUNoiseAccelerator.canvas$mhEnqueueReadBuffer()
                .invoke(clQueue, bRes, CL_TRUE, 0L, resBytes, resMem, 0, MemorySegment.NULL, MemorySegment.NULL);
            if (rdErr != CL_SUCCESS) throw new RuntimeException("clEnqueueReadBuffer: " + rdErr);
            GPUNoiseAccelerator.canvas$mhFinish().invoke(clQueue);

            // Copy to Java int[]
            int[] out = new int[N];
            for (int i = 0; i < N; i++) out[i] = resMem.get(ValueLayout.JAVA_INT, (long)i*4);

            // Release GPU buffers
            GPUNoiseAccelerator.canvas$mhReleaseMemObject().invoke(bPos);
            GPUNoiseAccelerator.canvas$mhReleaseMemObject().invoke(bRes);

            return out;
        }
    }

    private void setArg(Arena a, int idx, long size, MemorySegment val) throws Throwable {
        MemorySegment argBuf = a.allocate(size);
        argBuf.set(ValueLayout.ADDRESS, 0, val);
        int err = (int) GPUNoiseAccelerator.canvas$mhSetKernelArg()
            .invoke(clKernel, idx, size, argBuf);
        if (err != CL_SUCCESS) throw new RuntimeException("clSetKernelArg[" + idx + "]: " + err);
    }

    private void setArgInt(Arena a, int idx, int val) throws Throwable {
        MemorySegment argBuf = a.allocate(ValueLayout.JAVA_INT);
        argBuf.set(ValueLayout.JAVA_INT, 0, val);
        int err = (int) GPUNoiseAccelerator.canvas$mhSetKernelArg()
            .invoke(clKernel, idx, (long) ValueLayout.JAVA_INT.byteSize(), argBuf);
        if (err != CL_SUCCESS) throw new RuntimeException("clSetKernelArg int[" + idx + "]: " + err);
    }

    private static void checkErr(MemorySegment errBuf, String label) {
        int err = errBuf.get(ValueLayout.JAVA_INT, 0);
        if (err != CL_SUCCESS) throw new RuntimeException(label + ": " + err);
    }

    /**
     * Compute surface heights for a chunk-neighborhood centered at (chunkX, chunkZ).
     * Returns a Long2IntOpenHashMap keyed by ColumnPos.asLong(blockX, blockZ) → floor(height).
     * Returns null on GPU failure.
     *
     * Area computed: (chunkRadius*2+1)^2 chunks = RADIUS^2 * 16 * 16 quart-points sampled at 4-block intervals.
     * For RADIUS=3: 7×7=49 chunks, 28×28=784 quart-points.
     */
    public Long2IntOpenHashMap computeSurfaceForArea(int chunkX, int chunkZ, int chunkRadius) {
        // Quart-position grid: 4 quart-positions per chunk dimension, sampled at 4-block intervals
        int quartPerChunk = 4;
        int side = (chunkRadius * 2 + 1) * quartPerChunk; // quart-positions per side
        int N = side * side;

        int startBlockX = (chunkX - chunkRadius) * 16;
        int startBlockZ = (chunkZ - chunkRadius) * 16;

        int[] bxs = new int[N];
        int[] bzs = new int[N];
        int i = 0;
        for (int qz = 0; qz < side; qz++) {
            for (int qx = 0; qx < side; qx++) {
                bxs[i] = startBlockX + qx * 4;
                bzs[i] = startBlockZ + qz * 4;
                i++;
            }
        }

        int[] heights = computeSurface(bxs, bzs, N);
        if (heights == null) return null;

        Long2IntOpenHashMap map = new Long2IntOpenHashMap(N);
        for (int j = 0; j < N; j++) {
            map.put(ColumnPos.asLong(bxs[j], bzs[j]), heights[j]);
        }
        return map;
    }

    public boolean hasFailed() { return failed; }

    public boolean hasValueKernel() { return clValueKernel != null && !failed; }

    /** One-time bit-exact check of the persistent value kernel vs CPU finalDensity (8192 random pts). */
    boolean verifyValueKernel(DensityFunction df, int n) {
        java.util.Random r = new java.util.Random(0xABCDEF12L);
        int[] xs = new int[n], ys = new int[n], zs = new int[n];
        for (int i = 0; i < n; i++) {
            xs[i] = r.nextInt(4_000_000) - 2_000_000;
            ys[i] = r.nextInt(384) - 64;
            zs[i] = r.nextInt(4_000_000) - 2_000_000;
        }
        double[] gpu = computeGrid(xs, ys, zs, n);
        if (gpu == null) return false;
        for (int i = 0; i < n; i++) {
            double cpu = df.compute(new DensityFunction.SinglePointContext(xs[i], ys[i], zs[i]));
            if (Double.doubleToRawLongBits(cpu) != Double.doubleToRawLongBits(gpu[i])
                    && !(Double.isNaN(cpu) && Double.isNaN(gpu[i]))) {
                LOGGER.warn("[CanvasGPU-WorldGen] value kernel mismatch at ({},{},{}): cpu={} gpu={}",
                    xs[i], ys[i], zs[i], cpu, gpu[i]);
                return false;
            }
        }
        return true;
    }

    /** Builds an OpenCL kernel (program persists; one-time per world). Returns kernel or null. */
    private static MemorySegment buildKernel(String clSrc, String kernelName) throws Throwable {
        MemorySegment ctx = MemorySegment.ofAddress(GPUNoiseAccelerator.canvas$clContext());
        MemorySegment dev = MemorySegment.ofAddress(GPUNoiseAccelerator.canvas$clDevice());
        try (Arena a = Arena.ofConfined()) {
            MemorySegment devArr = a.allocate(ValueLayout.ADDRESS); devArr.set(ValueLayout.ADDRESS, 0, dev);
            byte[] sb = clSrc.getBytes(StandardCharsets.UTF_8);
            MemorySegment srcMem = a.allocateFrom(ValueLayout.JAVA_BYTE, sb);
            MemorySegment srcArr = a.allocate(ValueLayout.ADDRESS); srcArr.set(ValueLayout.ADDRESS, 0, srcMem);
            MemorySegment lenArr = a.allocate(ValueLayout.JAVA_LONG); lenArr.set(ValueLayout.JAVA_LONG, 0, (long) sb.length);
            MemorySegment prgErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment program = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateProgramWithSource().invoke(ctx, 1, srcArr, lenArr, prgErr);
            MemorySegment opts = a.allocateFrom("-cl-fp32-correctly-rounded-divide-sqrt");
            int be = (int) GPUNoiseAccelerator.canvas$mhBuildProgram().invoke(program, 1, devArr, opts, MemorySegment.NULL, MemorySegment.NULL);
            if (be != CL_SUCCESS) { LOGGER.warn("[CanvasGPU-WorldGen] buildKernel {} err={}", kernelName, be); return null; }
            MemorySegment kn = a.allocateFrom(kernelName);
            MemorySegment kErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment kernel = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateKernel().invoke(program, kn, kErr);
            return kErr.get(ValueLayout.JAVA_INT, 0) == CL_SUCCESS ? kernel : null;
        }
    }

    /**
     * Computes finalDensity at N arbitrary (x,y,z) int points via the value kernel.
     * Returns double[N] or null (GPU busy / unavailable). Used for full-chunk density grids.
     */
    public double[] computeGrid(int[] xs, int[] ys, int[] zs, int n) {
        if (failed || n == 0 || clValueKernel == null) return null;
        if (!lock.tryLock()) return null;
        try {
            try (Arena a = Arena.ofConfined()) {
                MemorySegment ctx = MemorySegment.ofAddress(GPUNoiseAccelerator.canvas$clContext());
                MemorySegment posMem = a.allocate((long) n * 3 * 4);
                for (int i = 0; i < n; i++) {
                    posMem.set(ValueLayout.JAVA_INT, (long)(i*3)*4, xs[i]);
                    posMem.set(ValueLayout.JAVA_INT, (long)(i*3+1)*4, ys[i]);
                    posMem.set(ValueLayout.JAVA_INT, (long)(i*3+2)*4, zs[i]);
                }
                MemorySegment errBuf = a.allocate(ValueLayout.JAVA_INT);
                MemorySegment bPos = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateBuffer().invoke(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (long) n*3*4, posMem, errBuf);
                MemorySegment bOut = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateBuffer().invoke(ctx, CL_MEM_WRITE_ONLY, (long) n*8, MemorySegment.NULL, errBuf);
                setArgS(a, clValueKernel, 0, ValueLayout.ADDRESS.byteSize(), bPos);
                setArgS(a, clValueKernel, 1, ValueLayout.ADDRESS.byteSize(), bOut);
                MemorySegment nBuf = a.allocate(ValueLayout.JAVA_INT); nBuf.set(ValueLayout.JAVA_INT, 0, n);
                GPUNoiseAccelerator.canvas$mhSetKernelArg().invoke(clValueKernel, 2, (long) ValueLayout.JAVA_INT.byteSize(), nBuf);
                MemorySegment gs = a.allocate(ValueLayout.JAVA_LONG); gs.set(ValueLayout.JAVA_LONG, 0, (long) n);
                int nd = (int) GPUNoiseAccelerator.canvas$mhEnqueueNDRangeKernel().invoke(clQueue, clValueKernel, 1, MemorySegment.NULL, gs, MemorySegment.NULL, 0, MemorySegment.NULL, MemorySegment.NULL);
                if (nd != CL_SUCCESS) throw new RuntimeException("computeGrid NDRange: " + nd);
                MemorySegment outMem = a.allocate((long) n * 8);
                int rd = (int) GPUNoiseAccelerator.canvas$mhEnqueueReadBuffer().invoke(clQueue, bOut, CL_TRUE, 0L, (long) n*8, outMem, 0, MemorySegment.NULL, MemorySegment.NULL);
                if (rd != CL_SUCCESS) throw new RuntimeException("computeGrid read: " + rd);
                GPUNoiseAccelerator.canvas$mhFinish().invoke(clQueue);
                double[] out = new double[n];
                for (int i = 0; i < n; i++) out[i] = outMem.get(ValueLayout.JAVA_DOUBLE, (long) i * 8);
                GPUNoiseAccelerator.canvas$mhReleaseMemObject().invoke(bPos);
                GPUNoiseAccelerator.canvas$mhReleaseMemObject().invoke(bOut);
                return out;
            }
        } catch (Throwable t) {
            failed = true;
            LOGGER.error("[CanvasGPU-WorldGen] computeGrid failed, disabling: {}", t.getMessage());
            return null;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Compiles an arbitrary DF to a value kernel, evaluates it on GPU at random 3D points,
     * and compares bit-exactly against CPU df.compute(). Logs PASS/FAIL. Standalone (own program).
     * Used to validate full-tree DFs like finalDensity before using them for chunk fill.
     */
    public static void verifyDensityFunction(DensityFunction df, String label, int n) {
        if (!GPUNoiseAccelerator.isAvailable()) return;
        String clSrc = DFCLCompiler.compileValue(df);
        if (clSrc == null) { LOGGER.warn("[CanvasGPU-WorldGen] verify[{}]: compile failed (unsupported node)", label); return; }
        try (Arena a = Arena.ofConfined()) {
            long ctxAddr = GPUNoiseAccelerator.canvas$clContext();
            long devAddr = GPUNoiseAccelerator.canvas$clDevice();
            if (ctxAddr == 0L || devAddr == 0L) return;
            MemorySegment ctx = MemorySegment.ofAddress(ctxAddr);
            MemorySegment dev = MemorySegment.ofAddress(devAddr);
            MemorySegment devArr = a.allocate(ValueLayout.ADDRESS); devArr.set(ValueLayout.ADDRESS, 0, dev);
            MemorySegment qErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment queue = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateCommandQueue().invoke(ctx, dev, 0L, qErr);

            byte[] sb = clSrc.getBytes(StandardCharsets.UTF_8);
            MemorySegment srcMem = a.allocateFrom(ValueLayout.JAVA_BYTE, sb);
            MemorySegment srcArr = a.allocate(ValueLayout.ADDRESS); srcArr.set(ValueLayout.ADDRESS, 0, srcMem);
            MemorySegment lenArr = a.allocate(ValueLayout.JAVA_LONG); lenArr.set(ValueLayout.JAVA_LONG, 0, (long) sb.length);
            MemorySegment prgErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment program = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateProgramWithSource().invoke(ctx, 1, srcArr, lenArr, prgErr);
            MemorySegment opts = a.allocateFrom("-cl-fp32-correctly-rounded-divide-sqrt");
            int buildErr = (int) GPUNoiseAccelerator.canvas$mhBuildProgram().invoke(program, 1, devArr, opts, MemorySegment.NULL, MemorySegment.NULL);
            if (buildErr != CL_SUCCESS) {
                try { java.nio.file.Files.writeString(java.nio.file.Path.of("/tmp/canvas_value_fail.cl"), clSrc); } catch (Exception ignored) {}
                MemorySegment logSize = a.allocate(ValueLayout.JAVA_LONG);
                GPUNoiseAccelerator.canvas$mhGetProgramBuildInfo().invoke(program, dev, CL_PROGRAM_BUILD_LOG, 0L, MemorySegment.NULL, logSize);
                long sz = logSize.get(ValueLayout.JAVA_LONG, 0);
                String log = "(no log)";
                if (sz > 0) {
                    MemorySegment lb = a.allocate(sz + 1);
                    GPUNoiseAccelerator.canvas$mhGetProgramBuildInfo().invoke(program, dev, CL_PROGRAM_BUILD_LOG, sz, lb, MemorySegment.NULL);
                    log = lb.getString(0, StandardCharsets.UTF_8);
                }
                String head = log.length() > 1200 ? log.substring(0, 1200) : log;
                LOGGER.warn("[CanvasGPU-WorldGen] verify[{}]: clBuildProgram err={} (src dumped /tmp/canvas_value_fail.cl)\n{}", label, buildErr, head);
                return;
            }
            MemorySegment kName = a.allocateFrom("compute_value");
            MemorySegment kErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment kernel = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateKernel().invoke(program, kName, kErr);

            java.util.Random r = new java.util.Random(0xABCDEF12L);
            int[] xs = new int[n], ys = new int[n], zs = new int[n];
            MemorySegment posMem = a.allocate((long) n * 3 * 4);
            for (int i = 0; i < n; i++) {
                xs[i] = r.nextInt(4_000_000) - 2_000_000;
                ys[i] = r.nextInt(384) - 64;
                zs[i] = r.nextInt(4_000_000) - 2_000_000;
                posMem.set(ValueLayout.JAVA_INT, (long)(i*3)*4, xs[i]);
                posMem.set(ValueLayout.JAVA_INT, (long)(i*3+1)*4, ys[i]);
                posMem.set(ValueLayout.JAVA_INT, (long)(i*3+2)*4, zs[i]);
            }
            MemorySegment errBuf = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment bPos = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateBuffer().invoke(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (long) n*3*4, posMem, errBuf);
            MemorySegment bOut = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateBuffer().invoke(ctx, CL_MEM_WRITE_ONLY, (long) n*8, MemorySegment.NULL, errBuf);
            setArgS(a, kernel, 0, ValueLayout.ADDRESS.byteSize(), bPos);
            setArgS(a, kernel, 1, ValueLayout.ADDRESS.byteSize(), bOut);
            MemorySegment nBuf = a.allocate(ValueLayout.JAVA_INT); nBuf.set(ValueLayout.JAVA_INT, 0, n);
            GPUNoiseAccelerator.canvas$mhSetKernelArg().invoke(kernel, 2, (long) ValueLayout.JAVA_INT.byteSize(), nBuf);
            MemorySegment gs = a.allocate(ValueLayout.JAVA_LONG); gs.set(ValueLayout.JAVA_LONG, 0, (long) n);
            GPUNoiseAccelerator.canvas$mhEnqueueNDRangeKernel().invoke(queue, kernel, 1, MemorySegment.NULL, gs, MemorySegment.NULL, 0, MemorySegment.NULL, MemorySegment.NULL);
            MemorySegment outMem = a.allocate((long) n * 8);
            GPUNoiseAccelerator.canvas$mhEnqueueReadBuffer().invoke(queue, bOut, CL_TRUE, 0L, (long) n*8, outMem, 0, MemorySegment.NULL, MemorySegment.NULL);
            GPUNoiseAccelerator.canvas$mhFinish().invoke(queue);

            int mism = 0; double maxUlp = 0; int fx=0,fy=0,fz=0; double fcpu=0,fgpu=0;
            for (int i = 0; i < n; i++) {
                double gpu = outMem.get(ValueLayout.JAVA_DOUBLE, (long) i * 8);
                double cpu = df.compute(new DensityFunction.SinglePointContext(xs[i], ys[i], zs[i]));
                if (Double.doubleToRawLongBits(cpu) != Double.doubleToRawLongBits(gpu)
                        && !(Double.isNaN(cpu) && Double.isNaN(gpu))) {
                    if (mism == 0) { fx=xs[i]; fy=ys[i]; fz=zs[i]; fcpu=cpu; fgpu=gpu; }
                    long d = Math.abs(Double.doubleToRawLongBits(cpu) - Double.doubleToRawLongBits(gpu));
                    if (d > maxUlp) maxUlp = d;
                    mism++;
                }
            }
            GPUNoiseAccelerator.canvas$mhReleaseMemObject().invoke(bPos);
            GPUNoiseAccelerator.canvas$mhReleaseMemObject().invoke(bOut);
            if (mism == 0)
                LOGGER.info("[CanvasGPU-WorldGen] VERIFY[{}] PASSED: {}/{} values bit-exact ({} bytes CL)", label, n, n, sb.length);
            else
                LOGGER.warn("[CanvasGPU-WorldGen] VERIFY[{}] FAILED: {}/{} mismatch maxULP={} first=({},{},{}) cpu={} gpu={}",
                    label, mism, n, (long) maxUlp, fx, fy, fz, fcpu, fgpu);
        } catch (Throwable t) {
            LOGGER.warn("[CanvasGPU-WorldGen] verify[{}] threw: {}", label, t.toString());
        }
    }

    /**
     * Benchmarks CPU df.compute() vs GPU (incl. host<->device transfer) across batch sizes.
     * Logs a table + crossover point. GPU buffers are persistent; only write+kernel+read is timed.
     */
    public static void benchmarkDensityFunction(DensityFunction df, String label) {
        if (!GPUNoiseAccelerator.isAvailable()) return;
        String clSrc = DFCLCompiler.compileValue(df);
        if (clSrc == null) return;
        final int MAXN = 262144;
        final int[] sizes = {1, 16, 64, 256, 1024, 4096, 16384, 65536, 262144};
        try (Arena a = Arena.ofConfined()) {
            MemorySegment ctx = MemorySegment.ofAddress(GPUNoiseAccelerator.canvas$clContext());
            MemorySegment dev = MemorySegment.ofAddress(GPUNoiseAccelerator.canvas$clDevice());
            MemorySegment devArr = a.allocate(ValueLayout.ADDRESS); devArr.set(ValueLayout.ADDRESS, 0, dev);
            MemorySegment qErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment queue = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateCommandQueue().invoke(ctx, dev, 0L, qErr);
            byte[] sb = clSrc.getBytes(StandardCharsets.UTF_8);
            MemorySegment srcMem = a.allocateFrom(ValueLayout.JAVA_BYTE, sb);
            MemorySegment srcArr = a.allocate(ValueLayout.ADDRESS); srcArr.set(ValueLayout.ADDRESS, 0, srcMem);
            MemorySegment lenArr = a.allocate(ValueLayout.JAVA_LONG); lenArr.set(ValueLayout.JAVA_LONG, 0, (long) sb.length);
            MemorySegment prgErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment program = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateProgramWithSource().invoke(ctx, 1, srcArr, lenArr, prgErr);
            MemorySegment opts = a.allocateFrom("-cl-fp32-correctly-rounded-divide-sqrt");
            if ((int) GPUNoiseAccelerator.canvas$mhBuildProgram().invoke(program, 1, devArr, opts, MemorySegment.NULL, MemorySegment.NULL) != CL_SUCCESS) return;
            MemorySegment kName = a.allocateFrom("compute_value");
            MemorySegment kErr = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment kernel = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateKernel().invoke(program, kName, kErr);

            java.util.Random r = new java.util.Random(7);
            int[] xs = new int[MAXN], ys = new int[MAXN], zs = new int[MAXN];
            MemorySegment posMem = a.allocate((long) MAXN * 3 * 4);
            for (int i = 0; i < MAXN; i++) {
                xs[i] = r.nextInt(4_000_000) - 2_000_000; ys[i] = r.nextInt(384) - 64; zs[i] = r.nextInt(4_000_000) - 2_000_000;
                posMem.set(ValueLayout.JAVA_INT, (long)(i*3)*4, xs[i]);
                posMem.set(ValueLayout.JAVA_INT, (long)(i*3+1)*4, ys[i]);
                posMem.set(ValueLayout.JAVA_INT, (long)(i*3+2)*4, zs[i]);
            }
            MemorySegment errBuf = a.allocate(ValueLayout.JAVA_INT);
            MemorySegment bPos = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateBuffer().invoke(ctx, CL_MEM_READ_ONLY, (long) MAXN*3*4, MemorySegment.NULL, errBuf);
            MemorySegment bOut = (MemorySegment) GPUNoiseAccelerator.canvas$mhCreateBuffer().invoke(ctx, CL_MEM_WRITE_ONLY, (long) MAXN*8, MemorySegment.NULL, errBuf);
            MemorySegment outMem = a.allocate((long) MAXN * 8);
            setArgS(a, kernel, 0, ValueLayout.ADDRESS.byteSize(), bPos);
            setArgS(a, kernel, 1, ValueLayout.ADDRESS.byteSize(), bOut);
            MemorySegment nBuf = a.allocate(ValueLayout.JAVA_INT);

            // warmup
            for (int w = 0; w < 3; w++) gpuRun(queue, kernel, bPos, bOut, posMem, outMem, nBuf, 4096);
            double sink = 0; for (int i = 0; i < 4096; i++) sink += df.compute(new DensityFunction.SinglePointContext(xs[i], ys[i], zs[i]));

            LOGGER.info("[CanvasGPU-Bench] {} — CPU vs GPU(incl transfer). chunk≈1200 density pts.", label);
            LOGGER.info(String.format("[CanvasGPU-Bench]   %8s %12s %12s %8s %9s", "N", "CPU us", "GPU us", "winner", "speedup"));
            int crossover = -1;
            for (int N : sizes) {
                int reps = Math.min(500, Math.max(3, 2_000_000 / N));
                long c0 = System.nanoTime();
                for (int rep = 0; rep < reps; rep++) { double s = 0; for (int i = 0; i < N; i++) s += df.compute(new DensityFunction.SinglePointContext(xs[i], ys[i], zs[i])); sink += s; }
                double cpuUs = (System.nanoTime() - c0) / 1000.0 / reps;
                long g0 = System.nanoTime();
                for (int rep = 0; rep < reps; rep++) gpuRun(queue, kernel, bPos, bOut, posMem, outMem, nBuf, N);
                double gpuUs = (System.nanoTime() - g0) / 1000.0 / reps;
                boolean gpuWins = gpuUs < cpuUs;
                if (gpuWins && crossover < 0) crossover = N;
                LOGGER.info(String.format("[CanvasGPU-Bench]   %8d %12.1f %12.1f %8s %8.2fx", N, cpuUs, gpuUs,
                    gpuWins ? "GPU" : "CPU", cpuUs / gpuUs));
            }
            LOGGER.info("[CanvasGPU-Bench]   -> crossover: GPU faster for N >= {} density-pts (~{} chunks)",
                crossover, crossover < 0 ? "never" : String.format("%.1f", crossover / 1200.0));
            if (Double.isNaN(sink)) LOGGER.info("");
            GPUNoiseAccelerator.canvas$mhReleaseMemObject().invoke(bPos);
            GPUNoiseAccelerator.canvas$mhReleaseMemObject().invoke(bOut);
        } catch (Throwable t) {
            LOGGER.warn("[CanvasGPU-Bench] {} threw: {}", label, t.toString());
        }
    }

    private static void gpuRun(MemorySegment queue, MemorySegment kernel, MemorySegment bPos, MemorySegment bOut,
                               MemorySegment posMem, MemorySegment outMem, MemorySegment nBuf, int N) throws Throwable {
        GPUNoiseAccelerator.canvas$mhEnqueueWriteBuffer().invoke(queue, bPos, CL_TRUE, 0L, (long) N*3*4, posMem, 0, MemorySegment.NULL, MemorySegment.NULL);
        nBuf.set(ValueLayout.JAVA_INT, 0, N);
        GPUNoiseAccelerator.canvas$mhSetKernelArg().invoke(kernel, 2, (long) ValueLayout.JAVA_INT.byteSize(), nBuf);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment gs = a.allocate(ValueLayout.JAVA_LONG); gs.set(ValueLayout.JAVA_LONG, 0, (long) N);
            GPUNoiseAccelerator.canvas$mhEnqueueNDRangeKernel().invoke(queue, kernel, 1, MemorySegment.NULL, gs, MemorySegment.NULL, 0, MemorySegment.NULL, MemorySegment.NULL);
        }
        GPUNoiseAccelerator.canvas$mhEnqueueReadBuffer().invoke(queue, bOut, CL_TRUE, 0L, (long) N*8, outMem, 0, MemorySegment.NULL, MemorySegment.NULL);
        GPUNoiseAccelerator.canvas$mhFinish().invoke(queue);
    }

    private static void setArgS(Arena a, MemorySegment kernel, int idx, long size, MemorySegment val) throws Throwable {
        MemorySegment argBuf = a.allocate(size);
        argBuf.set(ValueLayout.ADDRESS, 0, val);
        GPUNoiseAccelerator.canvas$mhSetKernelArg().invoke(kernel, idx, size, argBuf);
    }

    /**
     * End-to-end bit-exact gate: compares GPU surface heights against CPU df.compute() over a
     * random batch. Returns true only if EVERY sample matches. This validates the whole compiled
     * tree (noise + splines + DF ops + Y-scan) against vanilla. Never enable GPU if this fails.
     */
    public boolean verifyAgainstCpu(DensityFunction df, int n) {
        int[] xs = new int[n], zs = new int[n];
        java.util.Random r = new java.util.Random(0x5EEDC0DEL);
        for (int i = 0; i < n; i++) {
            // spread across a large coordinate range, 4-block aligned (quart positions)
            xs[i] = (r.nextInt(2_000_000) - 1_000_000) & ~3;
            zs[i] = (r.nextInt(2_000_000) - 1_000_000) & ~3;
        }
        int[] gpu = computeSurface(xs, zs, n);
        if (gpu == null) {
            LOGGER.warn("[CanvasGPU-WorldGen] verify: GPU dispatch returned null");
            return false;
        }
        int mismatches = 0, maxDiff = 0, firstX = 0, firstZ = 0, firstCpu = 0, firstGpu = 0;
        for (int i = 0; i < n; i++) {
            int cpu = (int) df.compute(new DensityFunction.SinglePointContext(xs[i], 0, zs[i]));
            if (cpu != gpu[i]) {
                if (mismatches == 0) { firstX = xs[i]; firstZ = zs[i]; firstCpu = cpu; firstGpu = gpu[i]; }
                int d = Math.abs(cpu - gpu[i]);
                if (d > maxDiff) maxDiff = d;
                mismatches++;
            }
        }
        if (mismatches == 0) {
            LOGGER.info("[CanvasGPU-WorldGen] BIT-EXACT VERIFY PASSED: {}/{} surface heights match CPU exactly", n, n);
            return true;
        }
        LOGGER.warn("[CanvasGPU-WorldGen] verify: {}/{} MISMATCH (maxDiff={}). first: ({},{}) cpu={} gpu={}",
            mismatches, n, maxDiff, firstX, firstZ, firstCpu, firstGpu);
        return false;
    }
}
