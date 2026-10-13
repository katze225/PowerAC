package me.katze.powerac.ai;

public final class Gelu {

    private static final float SQRT_2 = (float) Math.sqrt(2.0d);

    private Gelu() {}

    public static float erf(float x) {
        float sign = x < 0f ? -1f : 1f;
        float ax = Math.abs(x);
        float t = 1f / (1f + 0.3275911f * ax);
        float y =
            1f -
            (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
                0.284496736f) *
                t +
                0.254829592f) *
            t *
            (float) Math.exp(-(double) ax * (double) ax);
        return sign * y;
    }

    public static float apply(float x) {
        return 0.5f * x * (1f + erf(x / SQRT_2));
    }

    public static void applyInPlace(float[] values) {
        for (int i = 0; i < values.length; i++) {
            values[i] = apply(values[i]);
        }
    }
}
