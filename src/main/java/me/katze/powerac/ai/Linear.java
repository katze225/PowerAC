package me.katze.powerac.ai;

public final class Linear {

    public final int inFeatures;
    public final int outFeatures;
    public final float[] weight;
    public final float[] bias;

    public Linear(int inFeatures, int outFeatures, float[] weight, float[] bias) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weight = weight;
        this.bias = bias;
    }

    public Linear copy() {
        return new Linear(inFeatures, outFeatures, weight.clone(), bias.clone());
    }

    public int parameterCount() {
        return weight.length + bias.length;
    }

    public float[] forward(float[] input) {
        float[] output = new float[outFeatures];
        for (int o = 0; o < outFeatures; o++) {
            int offset = o * inFeatures;
            float acc = bias[o];
            for (int i = 0; i < inFeatures; i++) {
                acc += weight[offset + i] * input[i];
            }
            output[o] = acc;
        }
        return output;
    }
}
