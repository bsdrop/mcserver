package io.canvasmc.canvas.simd;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Loads a pre-compiled Rust native library (libcanvas_native.so) bundled
 * in the JAR under native/<platform>/<variant>/canvas-native.so.
 *
 * Selection priority (first match wins):
 *   Linux x86_64:  avx512 → avx2 → compat (SSE4.2)
 *   Linux aarch64: a78 → a76 → baseline
 *   macOS arm64:   darwin_aarch64
 *   macOS x86_64:  darwin_x86_64
 *
 * If no matching library is found or loading fails, isLoaded() returns false
 * and callers fall back to Java Vector API (BatchAABB.intersectBatch).
 */
public final class NativeBatchAABB {

    private static final Logger LOGGER = LogManager.getLogger("NativeBatchAABB");

    private static final boolean LOADED;

    static {
        LOADED = tryLoad();
    }

    private NativeBatchAABB() {}

    public static boolean isLoaded() {
        return LOADED;
    }

    // ── JNI entry point ───────────────────────────────────────────────────────

    public static native void nativeIntersectBatch(
        float[] minX, float[] minY, float[] minZ,
        float[] maxX, float[] maxY, float[] maxZ,
        int count,
        float qMinX, float qMinY, float qMinZ,
        float qMaxX, float qMaxY, float qMaxZ,
        boolean[] out
    );

    // ── loader ────────────────────────────────────────────────────────────────

    private static boolean tryLoad() {
        String[] candidates = buildCandidateList();
        for (String resource : candidates) {
            if (tryLoadResource(resource)) {
                LOGGER.info("Loaded native AABB library: {}", resource);
                return true;
            }
        }
        LOGGER.info("No native AABB library found — using Java Vector API fallback");
        return false;
    }

    private static String[] buildCandidateList() {
        String os   = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch",  "").toLowerCase();

        if (os.contains("linux")) {
            if (arch.equals("amd64") || arch.equals("x86_64")) {
                // MUST gate by actual CPU features: loading an AVX-512 .so on a CPU without AVX-512
                // succeeds but SIGILLs on first call (crashes the JVM). Only offer variants the CPU runs.
                java.util.List<String> c = new java.util.ArrayList<>();
                if (x86HasFlag("avx512f")) c.add("native/linux_x86_64/canvas-native-avx512.so");
                if (x86HasFlag("avx2"))    c.add("native/linux_x86_64/canvas-native.so");
                c.add("native/linux_x86_64/canvas-native-compat.so"); // SSE baseline fallback
                return c.toArray(new String[0]);
            }
            if (arch.equals("aarch64") || arch.equals("arm64")) {
                String cpu = detectArmCpu();
                if (cpu.equals("a78")) {
                    return new String[]{
                        "native/linux_aarch64/canvas-native-a78.so",
                        "native/linux_aarch64/canvas-native-a76.so",
                        "native/linux_aarch64/canvas-native.so",
                    };
                }
                if (cpu.equals("a76")) {
                    return new String[]{
                        "native/linux_aarch64/canvas-native-a76.so",
                        "native/linux_aarch64/canvas-native.so",
                    };
                }
                return new String[]{"native/linux_aarch64/canvas-native.so"};
            }
        }
        if (os.contains("mac")) {
            if (arch.equals("aarch64") || arch.equals("arm64")) {
                return new String[]{"native/darwin_aarch64/canvas-native.dylib"};
            }
            return new String[]{"native/darwin_x86_64/canvas-native.dylib"};
        }
        return new String[0];
    }

    /**
     * Reads /proc/cpuinfo and maps known Cortex-A part numbers to a tier.
     * Cortex-A78 = 0xD41, Cortex-A76 = 0xD0B, A55 = 0xD05.
     * Returns "a78", "a76", or "generic".
     */
    /** True if /proc/cpuinfo advertises the given x86 feature flag (exact token match). */
    private static boolean x86HasFlag(String flag) {
        try {
            for (String line : Files.readString(Path.of("/proc/cpuinfo")).split("\n")) {
                if (line.startsWith("flags") || line.startsWith("Features")) {
                    for (String tok : line.substring(line.indexOf(':') + 1).trim().split("\\s+"))
                        if (tok.equals(flag)) return true;
                }
            }
        } catch (Throwable ignored) {}
        return false;
    }

    private static String detectArmCpu() {
        try {
            String info = Files.readString(Path.of("/proc/cpuinfo"));
            // Find the highest-end core: A78 > A76 > A55
            if (info.contains("0xd41") || info.contains("Cortex-A78")) return "a78";
            if (info.contains("0xd0b") || info.contains("Cortex-A76")) return "a76";
        } catch (IOException ignored) {}
        return "generic";
    }

    private static boolean tryLoadResource(String resource) {
        // Under Paperclip the class's own classloader may not serve jar resources; try several.
        InputStream in = openResource(resource);
        if (in == null) { LOGGER.debug("native resource not found on any classloader: {}", resource); return false; }
        try (in) {
            // suffix must not contain '/': use the filename (after the last slash), not the slash itself.
            Path tmp = Files.createTempFile("canvas-native-", resource.substring(resource.lastIndexOf('/') + 1));
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            System.load(tmp.toAbsolutePath().toString());
            return true;
        } catch (Throwable t) {
            LOGGER.warn("Failed to load native {}: {}", resource, t.toString()); // visible: real load failures
            return false;
        }
    }

    /** Resolve a bundled resource via multiple classloaders (Paperclip-robust). */
    private static InputStream openResource(String resource) {
        InputStream in = NativeBatchAABB.class.getResourceAsStream("/" + resource);
        if (in == null) in = NativeBatchAABB.class.getClassLoader().getResourceAsStream(resource);
        if (in == null) {
            ClassLoader tccl = Thread.currentThread().getContextClassLoader();
            if (tccl != null) in = tccl.getResourceAsStream(resource);
        }
        if (in == null) {
            ClassLoader scl = ClassLoader.getSystemClassLoader();
            if (scl != null) in = scl.getResourceAsStream(resource);
        }
        return in;
    }
}
