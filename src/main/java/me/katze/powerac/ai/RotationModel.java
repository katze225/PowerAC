package me.katze.powerac.ai;

public final class RotationModel {

    private final RotationNet net;
    private final AggregationConfig aggregation;
    private final RotationScorer scorer;

    public RotationModel(RotationNet net, AggregationConfig aggregation) {
        this.net = net;
        this.aggregation = aggregation;
        this.scorer = new RotationScorer(aggregation);
    }

    public RotationNet getNet() {
        return net;
    }

    public AggregationConfig getAggregation() {
        return aggregation;
    }

    public RotationScorer getScorer() {
        return scorer;
    }

    public boolean isReady() {
        return net != null;
    }
}
