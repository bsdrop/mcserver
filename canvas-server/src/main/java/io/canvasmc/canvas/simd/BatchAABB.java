package io.canvasmc.canvas.simd;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD batch AABB intersection check.
 *
 * Caller maintains SoA float arrays parallel to the entity list.
 * Checking N boxes against 1 query box processes SPECIES.length() at a time.
 *
 * On Cortex-A72 NEON: 4 floats/vector → 4 entities per cycle vs 1 scalar.
 * On x86 AVX2: 8 floats/vector → 8 entities per cycle.
 *
 * Usage: maintain and update the SoA arrays whenever entities move, then
 * call intersectingMask() once per query instead of N scalar intersects().
 */
public final class BatchAABB {

    private static final VectorSpecies<Float> SPECIES = SIMDDetection.FLOAT_SPECIES;

    private BatchAABB() {}

    /**
     * Check which of the N entity boxes intersect with the query box.
     *
     * Returns a boolean[] result array (caller-provided, length >= count).
     * Processes SPECIES.length() entities per iteration with SIMD.
     *
     * Intersection condition (per axis): entityMin < qMax && entityMax > qMin
     * Combined: all 3 axes must pass → AND of 6 float comparisons.
     */
    public static void intersectBatch(
        final float[] eMinX, final float[] eMinY, final float[] eMinZ,
        final float[] eMaxX, final float[] eMaxY, final float[] eMaxZ,
        final int count,
        final float qMinX, final float qMinY, final float qMinZ,
        final float qMaxX, final float qMaxY, final float qMaxZ,
        final boolean[] out
    ) {
        final FloatVector vQMinX = FloatVector.broadcast(SPECIES, qMinX);
        final FloatVector vQMinY = FloatVector.broadcast(SPECIES, qMinY);
        final FloatVector vQMinZ = FloatVector.broadcast(SPECIES, qMinZ);
        final FloatVector vQMaxX = FloatVector.broadcast(SPECIES, qMaxX);
        final FloatVector vQMaxY = FloatVector.broadcast(SPECIES, qMaxY);
        final FloatVector vQMaxZ = FloatVector.broadcast(SPECIES, qMaxZ);

        final int lanes = SPECIES.length();
        int i = 0;

        // Main SIMD loop — full vectors
        for (; i <= count - lanes; i += lanes) {
            final FloatVector vEMinX = FloatVector.fromArray(SPECIES, eMinX, i);
            final FloatVector vEMinY = FloatVector.fromArray(SPECIES, eMinY, i);
            final FloatVector vEMinZ = FloatVector.fromArray(SPECIES, eMinZ, i);
            final FloatVector vEMaxX = FloatVector.fromArray(SPECIES, eMaxX, i);
            final FloatVector vEMaxY = FloatVector.fromArray(SPECIES, eMaxY, i);
            final FloatVector vEMaxZ = FloatVector.fromArray(SPECIES, eMaxZ, i);

            // eMinX < qMaxX && eMaxX > qMinX (repeat per axis)
            final VectorMask<Float> mask =
                vEMinX.compare(VectorOperators.LT, vQMaxX)
                    .and(vEMaxX.compare(VectorOperators.GT, vQMinX))
                    .and(vEMinY.lt(vQMaxY))
                    .and(vEMaxY.compare(VectorOperators.GT, vQMinY))
                    .and(vEMinZ.lt(vQMaxZ))
                    .and(vEMaxZ.compare(VectorOperators.GT, vQMinZ));

            for (int j = 0; j < lanes; j++) {
                out[i + j] = mask.laneIsSet(j);
            }
        }

        // Scalar tail for remaining elements
        for (; i < count; i++) {
            out[i] = eMinX[i] < qMaxX && eMaxX[i] > qMinX
                   && eMinY[i] < qMaxY && eMaxY[i] > qMinY
                   && eMinZ[i] < qMaxZ && eMaxZ[i] > qMinZ;
        }
    }

    /**
     * Returns count of intersecting boxes. Useful when you only need the count.
     */
    public static int intersectCount(
        final float[] eMinX, final float[] eMinY, final float[] eMinZ,
        final float[] eMaxX, final float[] eMaxY, final float[] eMaxZ,
        final int count,
        final float qMinX, final float qMinY, final float qMinZ,
        final float qMaxX, final float qMaxY, final float qMaxZ
    ) {
        final FloatVector vQMinX = FloatVector.broadcast(SPECIES, qMinX);
        final FloatVector vQMinY = FloatVector.broadcast(SPECIES, qMinY);
        final FloatVector vQMinZ = FloatVector.broadcast(SPECIES, qMinZ);
        final FloatVector vQMaxX = FloatVector.broadcast(SPECIES, qMaxX);
        final FloatVector vQMaxY = FloatVector.broadcast(SPECIES, qMaxY);
        final FloatVector vQMaxZ = FloatVector.broadcast(SPECIES, qMaxZ);

        final int lanes = SPECIES.length();
        int total = 0;
        int i = 0;

        for (; i <= count - lanes; i += lanes) {
            final VectorMask<Float> mask =
                FloatVector.fromArray(SPECIES, eMinX, i).compare(VectorOperators.LT, vQMaxX)
                    .and(FloatVector.fromArray(SPECIES, eMaxX, i).compare(VectorOperators.GT, vQMinX))
                    .and(FloatVector.fromArray(SPECIES, eMinY, i).compare(VectorOperators.LT, vQMaxY))
                    .and(FloatVector.fromArray(SPECIES, eMaxY, i).compare(VectorOperators.GT, vQMinY))
                    .and(FloatVector.fromArray(SPECIES, eMinZ, i).compare(VectorOperators.LT, vQMaxZ))
                    .and(FloatVector.fromArray(SPECIES, eMaxZ, i).compare(VectorOperators.GT, vQMinZ));
            total += mask.trueCount();
        }

        for (; i < count; i++) {
            if (eMinX[i] < qMaxX && eMaxX[i] > qMinX
             && eMinY[i] < qMaxY && eMaxY[i] > qMinY
             && eMinZ[i] < qMaxZ && eMaxZ[i] > qMinZ) total++;
        }

        return total;
    }
}
