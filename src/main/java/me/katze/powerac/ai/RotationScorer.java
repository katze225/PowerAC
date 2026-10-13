package me.katze.powerac.ai;

import java.util.Arrays;

public final class RotationScorer {

    private final AggregationConfig config;

    public RotationScorer(AggregationConfig config) {
        this.config = config;
    }

    public AggregationConfig getConfig() {
        return config;
    }

    public float score(float[] windowProbabilities) {
        int count = windowProbabilities.length;
        if (count == 0) {
            return 0f;
        }

        float mean = 0f;
        for (float value : windowProbabilities) {
            mean += value;
        }
        mean /= count;

        float[] sorted = Arrays.copyOf(windowProbabilities, count);
        Arrays.sort(sorted);
        float median = sorted[(count - 1) / 2];

        int topCount = Math.max(1, (int) (count * config.topFraction));
        float topSum = 0f;
        for (int i = count - topCount; i < count; i++) {
            topSum += sorted[i];
        }
        float topScore = topSum / topCount;

        int suspiciousCount = 0;
        float spreadSum = 0f;
        for (float value : windowProbabilities) {
            if (value > config.suspicious) {
                suspiciousCount++;
            }
            spreadSum += Math.abs(value - median);
        }
        float suspiciousRatio = (float) suspiciousCount / count;
        float spread = spreadSum / count;

        float baseScore = config.meanWeight * mean + config.medianWeight * median;
        float finalScore = suspiciousRatio >= config.ratioGate
            ? baseScore + config.topBoost * (topScore - baseScore)
            : baseScore;
        finalScore -= config.spreadPenalty * spread;
        return Math.max(0f, Math.min(1f, finalScore));
    }

    public boolean isDetected(float score) {
        return score >= config.decision;
    }
}
