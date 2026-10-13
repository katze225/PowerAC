package me.katze.powerac.ai;

public final class DetectionResult {

    public final float probability;
    public final boolean detected;
    public final float rawScore;
    public final int windowCount;
    public final String reason;

    public DetectionResult(
        float probability,
        boolean detected,
        float rawScore,
        int windowCount,
        String reason
    ) {
        this.probability = probability;
        this.detected = detected;
        this.rawScore = rawScore;
        this.windowCount = windowCount;
        this.reason = reason;
    }

    public static DetectionResult unavailable(String reason) {
        return new DetectionResult(-1f, false, -1f, 0, reason);
    }
}
