package me.katze.powerac.ai;

public final class GroupNorm1d {

    public static final float DEFAULT_EPSILON = 1e-5f;

    public final int groups;
    public final int channels;
    public final float epsilon;
    public final float[] weight;
    public final float[] bias;

    public GroupNorm1d(
        int groups,
        int channels,
        float epsilon,
        float[] weight,
        float[] bias
    ) {
        this.groups = groups;
        this.channels = channels;
        this.epsilon = epsilon;
        this.weight = weight;
        this.bias = bias;
    }

    public GroupNorm1d copy() {
        return new GroupNorm1d(groups, channels, epsilon, weight.clone(), bias.clone());
    }

    public int parameterCount() {
        return weight.length + bias.length;
    }

    public void forwardInPlace(float[] values, int time) {
        int channelsPerGroup = channels / groups;
        int elementsPerGroup = channelsPerGroup * time;
        for (int g = 0; g < groups; g++) {
            int groupOffset = g * elementsPerGroup;
            double sum = 0d;
            for (int i = 0; i < elementsPerGroup; i++) {
                sum += values[groupOffset + i];
            }
            double mean = sum / elementsPerGroup;
            double variance = 0d;
            for (int i = 0; i < elementsPerGroup; i++) {
                double diff = values[groupOffset + i] - mean;
                variance += diff * diff;
            }
            variance /= elementsPerGroup;
            float inverse = (float) (1d / Math.sqrt(variance + epsilon));
            for (int c = 0; c < channelsPerGroup; c++) {
                int channel = g * channelsPerGroup + c;
                float channelWeight = weight[channel];
                float channelBias = bias[channel];
                int channelOffset = channel * time;
                for (int t = 0; t < time; t++) {
                    int index = channelOffset + t;
                    values[index] =
                        (float) ((values[index] - mean) * inverse) * channelWeight +
                        channelBias;
                }
            }
        }
    }
}
