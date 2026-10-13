package me.katze.powerac.ai;

public final class Conv1d {

    public final int inChannels;
    public final int outChannels;
    public final int kernel;
    public final int padding;
    public final float[] weight;
    public final float[] bias;

    public Conv1d(
        int inChannels,
        int outChannels,
        int kernel,
        int padding,
        float[] weight,
        float[] bias
    ) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernel = kernel;
        this.padding = padding;
        this.weight = weight;
        this.bias = bias;
    }

    public Conv1d copy() {
        return new Conv1d(inChannels, outChannels, kernel, padding, weight.clone(), bias.clone());
    }

    public int parameterCount() {
        return weight.length + bias.length;
    }

    public float[] forward(float[] input, int time) {
        int width = time + 2 * padding;
        float[] padded = new float[inChannels * width];
        for (int c = 0; c < inChannels; c++) {
            System.arraycopy(input, c * time, padded, c * width + padding, time);
        }

        float[] output = new float[outChannels * time];
        for (int o = 0; o < outChannels; o++) {
            int outOffset = o * time;
            float biasValue = bias[o];
            for (int t = 0; t < time; t++) {
                output[outOffset + t] = biasValue;
            }
            int weightOffset = o * inChannels * kernel;
            for (int c = 0; c < inChannels; c++) {
                int paddedOffset = c * width;
                int weightBase = weightOffset + c * kernel;
                for (int k = 0; k < kernel; k++) {
                    float w = weight[weightBase + k];
                    int sourceOffset = paddedOffset + k;
                    for (int t = 0; t < time; t++) {
                        output[outOffset + t] += w * padded[sourceOffset + t];
                    }
                }
            }
        }
        return output;
    }
}
