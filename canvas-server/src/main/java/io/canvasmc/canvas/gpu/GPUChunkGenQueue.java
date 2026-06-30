package io.canvasmc.canvas.gpu;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Canvas GPU Chunk Gen Queue Monitor
 *
 * Tracks the depth of the chunk generation task queue.
 * When depth exceeds GPUNoiseAccelerator.BATCH_THRESHOLD, triggers
 * GPU-accelerated batch processing for pending noise computations.
 *
 * This class is the glue between Folia's chunk gen pipeline and the GPU accelerator.
 * It acts as a load balancer: CPU handles <= threshold chunks normally,
 * GPU handles overflow bursts (e.g., from forceload or mass player movement).
 *
 * Thread safety: all methods are thread-safe (Folia multi-region).
 */
public final class GPUChunkGenQueue {

    private static final Logger LOGGER = LogManager.getLogger("CanvasGPU");
    private static final AtomicInteger pendingCount = new AtomicInteger(0);
    private static final AtomicInteger gpuBatchCount = new AtomicInteger(0);
    private static final AtomicInteger cpuFallbackCount = new AtomicInteger(0);

    // Running stats for /canvas gpu stats
    private static volatile long lastBatchTime = 0;
    private static volatile int lastBatchSize  = 0;

    private GPUChunkGenQueue() {}

    /** Called when a chunk gen task is submitted to the queue. */
    public static void onChunkQueued() {
        int depth = pendingCount.incrementAndGet();
        if (depth > GPUNoiseAccelerator.BATCH_THRESHOLD && GPUNoiseAccelerator.isAvailable()) {
            // Signal that GPU batch mode should activate for the next chunk gen cycle
            // The actual GPU dispatch happens in NoiseChunkGPUBridge when fillArray is called
        }
    }

    /** Called when a chunk gen task completes (success or fail). */
    public static void onChunkDequeued() {
        pendingCount.decrementAndGet();
    }

    // The per-fillArray noise GPU interception is OFF by default: it measured NEUTRAL (no speedup)
    // and, while a worldgen-router kernel is compiling, its dispatches can queue behind the build and
    // STALL worker threads (observed as "chunks don't generate during compile"). OpenCL stays
    // initialized for the surface/density paths regardless. Opt in with -Dcanvas.gpu.noise.fillarray=true.
    private static final boolean FILLARRAY_ENABLED = Boolean.getBoolean("canvas.gpu.noise.fillarray");

    /** Returns true if GPU batch mode should be used for noise computation right now. */
    public static boolean shouldUseGPU() {
        return FILLARRAY_ENABLED
            && GPUNoiseAccelerator.isAvailable()
            && pendingCount.get() >= GPUNoiseAccelerator.BATCH_THRESHOLD;
    }

    /** Current pending chunk gen depth. */
    public static int getDepth() { return pendingCount.get(); }

    /** Record a GPU batch execution. */
    public static void recordGPUBatch(int size, long millis) {
        gpuBatchCount.incrementAndGet();
        lastBatchSize = size;
        lastBatchTime = millis;
    }

    /** Record a CPU fallback execution. */
    public static void recordCPUFallback() {
        cpuFallbackCount.incrementAndGet();
    }

    public static String statsLine() {
        return String.format(
            "pending=%d gpu_batches=%d cpu_fallbacks=%d last_batch=%d chunks in %dms",
            pendingCount.get(), gpuBatchCount.get(), cpuFallbackCount.get(),
            lastBatchSize, lastBatchTime
        );
    }
}
