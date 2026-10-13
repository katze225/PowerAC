package me.katze.powerac.ai;

public final class ModelSettings {

    public final boolean autoLoad;
    public final int maxWindowsPerLabel;
    public final int minWindowsPerClass;
    public final int epochs;
    public final int batchSize;
    public final float learningRate;
    public final float weightDecay;
    public final float gradClip;

    public ModelSettings(
        boolean autoLoad,
        int maxWindowsPerLabel,
        int minWindowsPerClass,
        int epochs,
        int batchSize,
        float learningRate,
        float weightDecay,
        float gradClip
    ) {
        this.autoLoad = autoLoad;
        this.maxWindowsPerLabel = Math.max(1, maxWindowsPerLabel);
        this.minWindowsPerClass = Math.max(1, minWindowsPerClass);
        this.epochs = Math.max(1, epochs);
        this.batchSize = Math.max(1, batchSize);
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.gradClip = gradClip;
    }

    public static ModelSettings defaults() {
        return new ModelSettings(true, 4000, 60, 8, 128, 3e-4f, 1e-2f, 1.0f);
    }
}
