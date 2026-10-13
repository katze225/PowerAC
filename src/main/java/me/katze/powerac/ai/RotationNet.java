package me.katze.powerac.ai;

import java.util.List;

public final class RotationNet {

    public static final String FORMAT = "powerac-rotation-cnn-v1";

    private final int inputDim;
    private final int windowSize;
    private final int hiddenDim;
    private long totalSamples;
    private final float[] featureMean;
    private final float[] featureStd;

    private final Conv1d conv1;
    private final Conv1d conv2;
    private final Conv1d conv3;
    private final GroupNorm1d norm1;
    private final GroupNorm1d norm2;
    private final GroupNorm1d norm3;
    private final Linear linear1;
    private final Linear linear2;
    private final Linear linear3;

    public RotationNet(
        int inputDim,
        int windowSize,
        int hiddenDim,
        long totalSamples,
        float[] featureMean,
        float[] featureStd,
        Conv1d conv1,
        Conv1d conv2,
        Conv1d conv3,
        GroupNorm1d norm1,
        GroupNorm1d norm2,
        GroupNorm1d norm3,
        Linear linear1,
        Linear linear2,
        Linear linear3
    ) {
        this.inputDim = inputDim;
        this.windowSize = windowSize;
        this.hiddenDim = hiddenDim;
        this.totalSamples = totalSamples;
        this.featureMean = featureMean;
        this.featureStd = featureStd;
        this.conv1 = conv1;
        this.conv2 = conv2;
        this.conv3 = conv3;
        this.norm1 = norm1;
        this.norm2 = norm2;
        this.norm3 = norm3;
        this.linear1 = linear1;
        this.linear2 = linear2;
        this.linear3 = linear3;
    }

    public int getInputDim() {
        return inputDim;
    }

    public int getWindowSize() {
        return windowSize;
    }

    public int getHiddenDim() {
        return hiddenDim;
    }

    public long getTotalSamples() {
        return totalSamples;
    }

    public void setTotalSamples(long totalSamples) {
        this.totalSamples = totalSamples;
    }

    public float[] getFeatureMean() {
        return featureMean;
    }

    public float[] getFeatureStd() {
        return featureStd;
    }

    public Conv1d getConv1() {
        return conv1;
    }

    public Conv1d getConv2() {
        return conv2;
    }

    public Conv1d getConv3() {
        return conv3;
    }

    public GroupNorm1d getNorm1() {
        return norm1;
    }

    public GroupNorm1d getNorm2() {
        return norm2;
    }

    public GroupNorm1d getNorm3() {
        return norm3;
    }

    public Linear getLinear1() {
        return linear1;
    }

    public Linear getLinear2() {
        return linear2;
    }

    public Linear getLinear3() {
        return linear3;
    }

    public RotationNet copy() {
        return new RotationNet(
            inputDim,
            windowSize,
            hiddenDim,
            totalSamples,
            featureMean.clone(),
            featureStd.clone(),
            conv1.copy(),
            conv2.copy(),
            conv3.copy(),
            norm1.copy(),
            norm2.copy(),
            norm3.copy(),
            linear1.copy(),
            linear2.copy(),
            linear3.copy()
        );
    }

    public float predict(float[] window) {
        return sigmoid(forwardLogit(window));
    }

    public float[] predictBatch(List<float[]> windows) {
        float[] probabilities = new float[windows.size()];
        for (int i = 0; i < windows.size(); i++) {
            probabilities[i] = predict(windows.get(i));
        }
        return probabilities;
    }

    public float forwardLogit(float[] window) {
        float[] activations = new float[inputDim * windowSize];
        for (int t = 0; t < windowSize; t++) {
            int rowOffset = t * inputDim;
            for (int c = 0; c < inputDim; c++) {
                activations[c * windowSize + t] =
                    (window[rowOffset + c] - featureMean[c]) / featureStd[c];
            }
        }

        activations = conv1.forward(activations, windowSize);
        norm1.forwardInPlace(activations, windowSize);
        Gelu.applyInPlace(activations);

        activations = conv2.forward(activations, windowSize);
        norm2.forwardInPlace(activations, windowSize);
        Gelu.applyInPlace(activations);

        activations = conv3.forward(activations, windowSize);
        norm3.forwardInPlace(activations, windowSize);
        Gelu.applyInPlace(activations);

        float[] pooled = new float[conv3.outChannels];
        for (int c = 0; c < conv3.outChannels; c++) {
            int offset = c * windowSize;
            float sum = 0f;
            for (int t = 0; t < windowSize; t++) {
                sum += activations[offset + t];
            }
            pooled[c] = sum / windowSize;
        }

        float[] hidden = linear1.forward(pooled);
        Gelu.applyInPlace(hidden);
        hidden = linear2.forward(hidden);
        Gelu.applyInPlace(hidden);
        return linear3.forward(hidden)[0];
    }

    public static float sigmoid(float value) {
        if (value >= 0f) {
            double z = Math.exp(-(double) value);
            return (float) (1d / (1d + z));
        }
        double z = Math.exp((double) value);
        return (float) (z / (1d + z));
    }
}
