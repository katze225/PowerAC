package me.katze.powerac.ai;

public final class AggregationConfig {

    public final float decision;
    public final float suspicious;
    public final float ratioGate;
    public final float topFraction;
    public final float meanWeight;
    public final float medianWeight;
    public final float topBoost;
    public final float spreadPenalty;

    public AggregationConfig(
        float decision,
        float suspicious,
        float ratioGate,
        float topFraction,
        float meanWeight,
        float medianWeight,
        float topBoost,
        float spreadPenalty
    ) {
        this.decision = decision;
        this.suspicious = suspicious;
        this.ratioGate = ratioGate;
        this.topFraction = topFraction;
        this.meanWeight = meanWeight;
        this.medianWeight = medianWeight;
        this.topBoost = topBoost;
        this.spreadPenalty = spreadPenalty;
    }

    public static AggregationConfig defaults() {
        return new AggregationConfig(
            0.79f,
            0.80f,
            0.25f,
            0.10f,
            0.90f,
            0.10f,
            0.07f,
            0.04f
        );
    }
}
