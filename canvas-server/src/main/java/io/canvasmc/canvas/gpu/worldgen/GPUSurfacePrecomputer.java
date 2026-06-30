package io.canvasmc.canvas.gpu.worldgen;

import io.canvasmc.canvas.gpu.GPUNoiseAccelerator;
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.server.level.ColumnPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GPU-accelerated surface height precomputation.
 *
 * Async model:
 *   - First precompute() call per RandomState starts async GPU compile on background thread
 *   - While compiling → CPU fallback (null)
 *   - After success  → GPU used for subsequent new-chunk requests
 *   - After failure  → CPU used permanently (null)
 *
 * Called ONLY from doFill() (new chunk generation path, never for existing chunks loaded from disk).
 *
 * GPU dispatches 17×17 chunk neighborhood (PRECOMPUTE_RADIUS=8) per trigger,
 * results cached globally so adjacent chunks get cache-hits.
 */
public final class GPUSurfacePrecomputer {

    private static final Logger LOGGER = LoggerFactory.getLogger("CanvasGPU-Surface");

    // 17×17 chunk radius = 289 chunks × 16 quart-points = 4624 points/dispatch
    // (well above the MIN_GPU_N=512 threshold for GPU benefit)
    public static final int PRECOMPUTE_RADIUS = 8;

    // Debug flag: enable verbose per-chunk logging (set via system property)
    public static final boolean VERBOSE =
        Boolean.getBoolean("canvas.gpu.surface.verbose");

    // Single background thread for GPU compilation (never blocks world-gen workers)
    private static final Executor GPU_COMPILE_EXEC =
        Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "canvas-gpu-compile");
            t.setDaemon(true);
            return t;
        });

    private enum CompileState { COMPILING, READY, FAILED }

    private static final class ContextEntry {
        volatile CompileState state;
        volatile GPUWorldGenContext ctx; // non-null only when state == READY

        // Statistics
        final AtomicLong gpuDispatches = new AtomicLong();
        final AtomicLong gpuTotalMs   = new AtomicLong();
        final AtomicLong cacheHits    = new AtomicLong();
        final AtomicLong cpuFallbacks = new AtomicLong();

        // Per-RandomState (per-dimension) density grid cache. MUST be per-state: different dimensions
        // have different cell configs (cellCountY) → different grid sizes; a global chunk-coord-keyed
        // cache collides across dimensions (overworld 1225 vs nether 196) → wrong-size grid → AIOOBE.
        final ConcurrentHashMap<Long, double[]> densityCache = new ConcurrentHashMap<>(16384);
        final ConcurrentHashMap<Long, Boolean> tileInProgress = new ConcurrentHashMap<>();
        // Per-RandomState surface-height cache (same multi-world/dimension isolation as densityCache).
        final ConcurrentHashMap<Long, Integer> surfaceCache = new ConcurrentHashMap<>(131072);

        ContextEntry() { this.state = CompileState.COMPILING; }
    }

    // Per-RandomState compile state — each world/dimension gets its own GPU program
    private static final ConcurrentHashMap<RandomState, ContextEntry> entries =
        new ConcurrentHashMap<>();

    // Surface height cache is per-RandomState (in ContextEntry.surfaceCache) to avoid cross-world/
    // dimension collisions. ColumnPos.asLong(blockX, blockZ) → floor(height).
    private static final int MAX_CACHE_ENTRIES = 2_000_000;

    private GPUSurfacePrecomputer() {}

    // ─── Main API ────────────────────────────────────────────────────────────────

    /**
     * Returns GPU-computed surface heights for chunk (chunkX, chunkZ) or null (CPU fallback).
     *
     * Thread-safe. Non-blocking. Only for newly generated chunks (doFill path).
     */
    // Default OFF: GPU surface only logs bit-exact verify result; generation stays pure CPU.
    // Flip to true (-Dcanvas.gpu.surface.enabled=true) only after verify PASSES in logs.
    public static final boolean SURFACE_ENABLED =
        Boolean.getBoolean("canvas.gpu.surface.enabled");

    // Full-chunk GPU density (Task #6). Off by default — opt in with -Dcanvas.gpu.density.enabled=true.
    public static final boolean DENSITY_ENABLED =
        Boolean.getBoolean("canvas.gpu.density.enabled");

    // Tile-aligned batching: chunks are grouped into fixed TILE×TILE tiles (aligned to multiples of
    // TILE). A miss computes the whole containing tile in ONE GPU dispatch (TILE²×cornersPerChunk
    // points, above the GPU crossover) and caches each chunk's grid. Because tiles are ALIGNED (not
    // centered on the requesting chunk), adjacent chunks share the same tile → no overlapping/redundant
    // dispatches, and each tile is computed at most once (tileInProgress dedup). TILE=4 → 16 chunks/
    // dispatch (~3136 pts ≫ 512 crossover). Tune with -Dcanvas.gpu.density.tile.
    private static final int DENSITY_TILE = Math.max(1, Math.min(8, Integer.getInteger("canvas.gpu.density.tile", 4)));
    private static final int MAX_DENSITY_CACHE = 100_000; // chunk grids per dimension
    private static final boolean INTERP_ENABLED = Boolean.getBoolean("canvas.gpu.interp");

    /**
     * GPU per-block trilinear interpolation of a cell-corner grid (offloads MC's CPU NoiseInterpolator).
     * Returns the per-block density (layout ((lx*bZ)+lz)*bY+ly) or null → caller keeps the CPU interp path.
     * Opt-in via -Dcanvas.gpu.interp (the per-block readback is ~80x the corner data, so its net value
     * depends on transfer/GC vs interp saved — measure before enabling by default).
     */
    public static double[] computeInterpolated(RandomState state, double[] corners,
            int cellWidth, int cellHeight, int cellCountXZ, int cellCountY) {
        if (!INTERP_ENABLED || corners == null) return null;
        ContextEntry entry = entries.get(state);
        if (entry == null || entry.state != CompileState.READY) return null;
        GPUWorldGenContext ctx = entry.ctx;
        if (ctx == null || !ctx.hasInterpKernel()) return null;
        return ctx.computeInterpChunk(corners, cellCountXZ + 1, cellCountXZ + 1, cellCountY + 1, cellWidth, cellHeight);
    }

    /**
     * Returns this chunk's finalDensity cell-corner grid (ordered (ix*nz+iz)*ny+iy, matching
     * GpuGridDensity). On a cache miss, computes the whole aligned TILE×TILE tile in one GPU dispatch
     * and caches each chunk. The cache is PER-RandomState (per-dimension) so grid sizes are uniform.
     * Returns null if disabled / not ready / GPU busy / tile being computed by another worker / size
     * mismatch → CPU fallback.
     */
    public static double[] computeDensityGrid(RandomState state, int originX, int originZ,
            int cellWidth, int cellHeight, int cellCountXZ, int cellCountY, int cellNoiseMinY) {
        if (!DENSITY_ENABLED) return null;

        ContextEntry entry = getOrInitEntry(state); // triggers async compile (incl. value kernel)
        if (entry.state != CompileState.READY) return null;
        GPUWorldGenContext ctx = entry.ctx;
        if (ctx == null || !ctx.hasValueKernel()) return null;

        final int nx = cellCountXZ + 1, nz = cellCountXZ + 1, ny = cellCountY + 1;
        final int perChunk = nx * nz * ny;
        int cx = Math.floorDiv(originX, 16), cz = Math.floorDiv(originZ, 16);
        long key = ColumnPos.asLong(cx, cz);
        double[] hit = entry.densityCache.get(key);
        if (hit != null) return hit.length == perChunk ? hit : null; // size guard (defensive)

        int baseCx = Math.floorDiv(cx, DENSITY_TILE) * DENSITY_TILE;
        int baseCz = Math.floorDiv(cz, DENSITY_TILE) * DENSITY_TILE;
        long tileKey = ColumnPos.asLong(baseCx, baseCz);
        if (entry.tileInProgress.putIfAbsent(tileKey, Boolean.TRUE) != null) return null; // another worker → CPU now
        try {
            double[] recheck = entry.densityCache.get(key); // another tile dispatch may have filled it
            if (recheck != null) return recheck.length == perChunk ? recheck : null;

            final int nChunks = DENSITY_TILE * DENSITY_TILE;
            final int total = nChunks * perChunk;
            int[] xs = new int[total], ys = new int[total], zs = new int[total];
            long[] keys = new long[nChunks];
            int idx = 0, ci = 0;
            for (int dz = 0; dz < DENSITY_TILE; dz++) {
                for (int dx = 0; dx < DENSITY_TILE; dx++) {
                    int ncx = baseCx + dx, ncz = baseCz + dz;
                    keys[ci++] = ColumnPos.asLong(ncx, ncz);
                    int ox = ncx * 16, oz = ncz * 16;
                    for (int ix = 0; ix < nx; ix++) {
                        int bx = ox + ix * cellWidth;
                        for (int iz = 0; iz < nz; iz++) {
                            int bz = oz + iz * cellWidth;
                            for (int iy = 0; iy < ny; iy++) {
                                xs[idx] = bx; zs[idx] = bz; ys[idx] = (cellNoiseMinY + iy) * cellHeight; idx++;
                            }
                        }
                    }
                }
            }
            double[] all = ctx.computeGrid(xs, ys, zs, total);
            if (all == null) return null; // GPU busy → CPU for this chunk

            if (entry.densityCache.size() > MAX_DENSITY_CACHE) entry.densityCache.clear(); // crude bound
            double[] result = null;
            for (int c = 0; c < nChunks; c++) {
                double[] g = new double[perChunk];
                System.arraycopy(all, c * perChunk, g, 0, perChunk);
                entry.densityCache.putIfAbsent(keys[c], g);
                if (keys[c] == key) result = g;
            }
            gpuDensityBatches.incrementAndGet();
            return result;
        } finally {
            entry.tileInProgress.remove(tileKey);
        }
    }

    private static final java.util.concurrent.atomic.AtomicBoolean reportedGatePass =
        new java.util.concurrent.atomic.AtomicBoolean();
    private static final java.util.concurrent.atomic.AtomicBoolean reportedGateFail =
        new java.util.concurrent.atomic.AtomicBoolean();

    private static final AtomicLong gpuDensityChunks = new AtomicLong();
    private static final AtomicLong gpuDensityBatches = new AtomicLong();

    /** Counts chunks whose density was generated on GPU (kernel pre-verified, no per-chunk gate). */
    public static void reportDensityUsed(int gridPts) {
        if (reportedGatePass.compareAndSet(false, true))
            LOGGER.info("[CanvasGPU-Surface] GPU FULL-CHUNK DENSITY ACTIVE — {} corner pts/chunk, "
                + "tile-aligned batch {}x{} ({} chunks/dispatch), kernel pre-verified (no per-chunk gate)",
                gridPts, DENSITY_TILE, DENSITY_TILE, DENSITY_TILE * DENSITY_TILE);
        long n = gpuDensityChunks.incrementAndGet();
        if (n % 256 == 0) {
            long miss = gpuDensityMisses.get();
            long total = n + miss;
            LOGGER.info("[CanvasGPU-Surface] GPU density: {} chunks via {} dispatches | hit-rate {}% ({} GPU / {} CPU-fallback)",
                n, gpuDensityBatches.get(), total == 0 ? 0 : (100 * n / total), n, miss);
        }
    }

    private static final AtomicLong gpuDensityMisses = new AtomicLong();

    /** A chunk's density was computed on CPU instead of GPU (tryLock fail, off-grid, size mismatch, or blend boundary). */
    public static void reportDensityMiss() {
        gpuDensityMisses.incrementAndGet();
    }

    /** Total chunks whose density was generated on GPU (for /canvas gpu status, etc.). */
    public static long gpuDensityChunkCount() { return gpuDensityChunks.get(); }

    public static Long2IntOpenHashMap precompute(RandomState state, int chunkX, int chunkZ) {
        ContextEntry entry = getOrInitEntry(state); // triggers async compile + bit-exact verify (logs)

        if (!SURFACE_ENABLED) return null; // verify-only mode: keep generation on CPU

        if (entry.state != CompileState.READY) {
            entry.cpuFallbacks.incrementAndGet();
            if (VERBOSE) LOGGER.debug("[GPS] chunk ({},{}) → CPU (GPU {})",
                chunkX, chunkZ, entry.state);
            return null;
        }

        // Check per-state cache first (populated by previous nearby dispatches)
        Long2IntOpenHashMap cached = extractChunkFromCache(entry, chunkX, chunkZ);
        if (cached != null) {
            entry.cacheHits.incrementAndGet();
            if (VERBOSE) LOGGER.debug("[GPS] chunk ({},{}) → cache-hit ({} entries)",
                chunkX, chunkZ, cached.size());
            return cached;
        }

        // Dispatch GPU for neighborhood
        GPUWorldGenContext ctx = entry.ctx;
        if (ctx == null || ctx.hasFailed()) {
            entry.state = CompileState.FAILED;
            LOGGER.warn("[CanvasGPU-Surface] GPU context failed, reverting to CPU permanently");
            return null;
        }

        long t0 = System.currentTimeMillis();
        Long2IntOpenHashMap area = ctx.computeSurfaceForArea(chunkX, chunkZ, PRECOMPUTE_RADIUS);
        long elapsed = System.currentTimeMillis() - t0;

        if (area == null) {
            entry.cpuFallbacks.incrementAndGet();
            if (VERBOSE) LOGGER.debug("[GPS] chunk ({},{}) → CPU (GPU locked/null, {}ms)",
                chunkX, chunkZ, elapsed);
            return null;
        }

        entry.gpuDispatches.incrementAndGet();
        entry.gpuTotalMs.addAndGet(elapsed);

        // Cache results for the entire neighborhood
        int areaSz = area.size();
        if (entry.surfaceCache.size() < MAX_CACHE_ENTRIES) {
            area.forEach((key, val) -> entry.surfaceCache.put((Long) key, val));
        }

        Long2IntOpenHashMap result = extractChunkFromMap(area, chunkX, chunkZ);
        long dispatches = entry.gpuDispatches.get();
        if (VERBOSE) {
            double avgMs = dispatches > 0 ? (double) entry.gpuTotalMs.get() / dispatches : 0;
            LOGGER.info("[GPS] chunk ({},{}) -> GPU dispatch: {}ms, area={} pts, "
                + "cache={} entries, avg GPU={}ms, hits={}, fallbacks={}",
                chunkX, chunkZ, elapsed, areaSz,
                entry.surfaceCache.size(), String.format("%.1f", avgMs),
                entry.cacheHits.get(), entry.cpuFallbacks.get());
        } else if (dispatches == 1 || dispatches % 100 == 0) {
            // Always log the first dispatch and every 100th
            LOGGER.info("[CanvasGPU-Surface] GPU surface dispatch #{}: {}ms "
                + "({}x{} chunk area, cache={} entries)",
                dispatches, elapsed,
                PRECOMPUTE_RADIUS*2+1, PRECOMPUTE_RADIUS*2+1,
                entry.surfaceCache.size());
        }

        return result;
    }

    private static ContextEntry getOrInitEntry(RandomState state) {
        ContextEntry existing = entries.get(state);
        if (existing != null) return existing;

        ContextEntry newEntry = new ContextEntry();
        ContextEntry prev = entries.putIfAbsent(state, newEntry);
        if (prev != null) return prev;

        // We won the race — start async compilation
        triggerAsyncCompile(state, newEntry);
        return newEntry;
    }

    private static void triggerAsyncCompile(RandomState state, ContextEntry entry) {
        CompletableFuture.runAsync(() -> {
            LOGGER.info("[CanvasGPU-Surface] Starting async GPU surface kernel compile "
                + "(radius={}, ~{} pts/dispatch)...",
                PRECOMPUTE_RADIUS, (PRECOMPUTE_RADIUS*2+1)*(PRECOMPUTE_RADIUS*2+1)*16);
            long t0 = System.currentTimeMillis();
            try {
                GPUWorldGenContext ctx = GPUWorldGenContext.create(state);
                long elapsed = System.currentTimeMillis() - t0;
                if (ctx == null) {
                    entry.state = CompileState.FAILED;
                    LOGGER.warn("[CanvasGPU-Surface] GPU surface compile FAILED after {}ms "
                        + "— using CPU permanently for this RandomState", elapsed);
                } else {
                    entry.ctx = ctx;
                    entry.state = CompileState.READY;
                    LOGGER.info("[CanvasGPU-Surface] GPU surface kernel READY in {}ms "
                        + "— GPU will accelerate new chunk surface generation", elapsed);
                    // Run a quick dispatch sanity check
                    validateGpuOutput(ctx);
                }
            } catch (Throwable t) {
                entry.state = CompileState.FAILED;
                LOGGER.warn("[CanvasGPU-Surface] GPU compile threw {}: {}",
                    t.getClass().getSimpleName(), t.getMessage());
            }
        }, GPU_COMPILE_EXEC);
    }

    /** Quick sanity: dispatch 4 known-coordinate points and log results. */
    private static void validateGpuOutput(GPUWorldGenContext ctx) {
        try {
            int[] xs = {0,   16,  32,  48};
            int[] zs = {0,   0,   0,   0};
            int[] heights = ctx.computeSurface(xs, zs, 4);
            if (heights != null) {
                LOGGER.info("[CanvasGPU-Surface] GPU sanity check: "
                    + "surface(0,0)={}, (16,0)={}, (32,0)={}, (48,0)={}",
                    heights[0], heights[1], heights[2], heights[3]);
            } else {
                LOGGER.warn("[CanvasGPU-Surface] GPU sanity check returned null");
            }
        } catch (Exception e) {
            LOGGER.warn("[CanvasGPU-Surface] GPU sanity check threw: {}", e.getMessage());
        }
    }

    // ─── Cache helpers ───────────────────────────────────────────────────────────

    /** Returns this chunk's 16 quart-points from the per-state cache, or null if any miss. */
    private static Long2IntOpenHashMap extractChunkFromCache(ContextEntry entry, int cx, int cz) {
        int sx = cx * 16, sz = cz * 16;
        Long2IntOpenHashMap map = new Long2IntOpenHashMap(16);
        for (int qz = 0; qz < 4; qz++) {
            for (int qx = 0; qx < 4; qx++) {
                long key = ColumnPos.asLong(sx + qx*4, sz + qz*4);
                Integer v = entry.surfaceCache.get(key);
                if (v == null) return null;
                map.put(key, (int) v);
            }
        }
        return map;
    }

    /** Extracts this chunk's 16 quart-points from an area map. */
    private static Long2IntOpenHashMap extractChunkFromMap(Long2IntOpenHashMap area, int cx, int cz) {
        int sx = cx * 16, sz = cz * 16;
        Long2IntOpenHashMap map = new Long2IntOpenHashMap(16);
        for (int qz = 0; qz < 4; qz++) {
            for (int qx = 0; qx < 4; qx++) {
                long key = ColumnPos.asLong(sx + qx*4, sz + qz*4);
                if (area.containsKey(key)) map.put(key, area.get(key));
            }
        }
        return map.isEmpty() ? null : map;
    }

    // ─── Lifecycle ──────────────────────────────────────────────────────────────

    public static void onWorldUnload(RandomState state) {
        ContextEntry entry = entries.remove(state);
        if (entry != null && entry.gpuDispatches.get() > 0) {
            LOGGER.info("[CanvasGPU-Surface] World unload stats: "
                + "GPU dispatches={}, avg={}ms, cache-hits={}, cpu-fallbacks={}",
                entry.gpuDispatches.get(),
                entry.gpuDispatches.get() > 0
                    ? entry.gpuTotalMs.get() / entry.gpuDispatches.get() : 0,
                entry.cacheHits.get(),
                entry.cpuFallbacks.get());
        }
    }

    public static void clearAllCaches() {
        entries.clear(); // per-entry surfaceCache/densityCache/tileInProgress go with the entries
    }

    /** Print current statistics to log. */
    public static void logStats() {
        entries.forEach((state, entry) -> {
            LOGGER.info("[CanvasGPU-Surface] State@{}: GPU={} dispatches, avg={}ms, "
                + "hits={}, fallbacks={}, cache={} entries, status={}",
                Integer.toHexString(System.identityHashCode(state)),
                entry.gpuDispatches.get(),
                entry.gpuDispatches.get() > 0
                    ? entry.gpuTotalMs.get() / entry.gpuDispatches.get() : 0,
                entry.cacheHits.get(),
                entry.cpuFallbacks.get(),
                entry.surfaceCache.size(),
                entry.state);
        });
    }
}
