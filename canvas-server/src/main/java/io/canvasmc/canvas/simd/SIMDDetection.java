package io.canvasmc.canvas.simd;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;

public final class SIMDDetection {

    public static boolean isEnabled = false;
    public static boolean testRun = false;

    public static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    public static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;

    // Per-batch sizes derived from hardware capabilities
    public static final int FLOAT_LANE_COUNT = FLOAT_SPECIES.length();
    public static final int DOUBLE_LANE_COUNT = DOUBLE_SPECIES.length();

    private SIMDDetection() {}

    public static boolean canEnable(final Logger logger) {
        testRun = true;
        try {
            // Float path test
            final FloatVector fa = FloatVector.broadcast(FLOAT_SPECIES, 1.0f);
            final FloatVector fb = FloatVector.broadcast(FLOAT_SPECIES, 2.0f);
            final float[] fresult = new float[FLOAT_LANE_COUNT];
            fa.add(fb).intoArray(fresult, 0);
            if (fresult[0] != 3.0f) return false;

            // Double path test
            final DoubleVector da = DoubleVector.broadcast(DOUBLE_SPECIES, 1.0);
            final DoubleVector db = DoubleVector.broadcast(DOUBLE_SPECIES, 2.0);
            final double[] dresult = new double[DOUBLE_LANE_COUNT];
            da.add(db).intoArray(dresult, 0);
            if (dresult[0] != 3.0) return false;

            // FMA test (critical for noise perf on ARM FP NEON)
            final FloatVector fc = FloatVector.broadcast(FLOAT_SPECIES, 3.0f);
            final float[] fmaresult = new float[FLOAT_LANE_COUNT];
            fa.fma(fb, fc).intoArray(fmaresult, 0);
            if (fmaresult[0] != 5.0f) return false;

            logger.info("Canvas SIMD: float species={} ({} lanes), double species={} ({} lanes)",
                FLOAT_SPECIES, FLOAT_LANE_COUNT, DOUBLE_SPECIES, DOUBLE_LANE_COUNT);
            return true;
        } catch (final Throwable t) {
            logger.warn("Canvas SIMD: test failed ({}), disabling", t.getMessage());
            return false;
        }
    }
}
