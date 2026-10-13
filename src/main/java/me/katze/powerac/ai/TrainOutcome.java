package me.katze.powerac.ai;

public final class TrainOutcome {

    public final boolean success;
    public final boolean waitingForData;
    public final float loss;
    public final int legitCount;
    public final int cheaterCount;
    public final String message;

    private TrainOutcome(
        boolean success,
        boolean waitingForData,
        float loss,
        int legitCount,
        int cheaterCount,
        String message
    ) {
        this.success = success;
        this.waitingForData = waitingForData;
        this.loss = loss;
        this.legitCount = legitCount;
        this.cheaterCount = cheaterCount;
        this.message = message;
    }

    static TrainOutcome success(float loss, int legitCount, int cheaterCount) {
        return new TrainOutcome(true, false, loss, legitCount, cheaterCount, "trained");
    }

    static TrainOutcome waiting(int legitCount, int cheaterCount) {
        return new TrainOutcome(false, true, -1f, legitCount, cheaterCount, "waiting for both classes");
    }

    static TrainOutcome failure(String message, int legitCount, int cheaterCount) {
        return new TrainOutcome(false, false, -1f, legitCount, cheaterCount, message);
    }
}
