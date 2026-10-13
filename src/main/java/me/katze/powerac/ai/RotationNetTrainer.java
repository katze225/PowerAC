package me.katze.powerac.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class RotationNetTrainer {

    private static final float SQRT_2 = (float) Math.sqrt(2d);
    private static final float INV_SQRT_2PI = (float) (1d / Math.sqrt(2d * Math.PI));

    private final RotationNet net;
    private final float learningRate;
    private final float weightDecay;
    private final float beta1 = 0.9f;
    private final float beta2 = 0.999f;
    private final float epsilon = 1e-8f;
    private final float gradClip;
    private final Random random;

    private final int time;
    private final int inputDim;

    private final Conv1d conv1;
    private final Conv1d conv2;
    private final Conv1d conv3;
    private final GroupNorm1d norm1;
    private final GroupNorm1d norm2;
    private final GroupNorm1d norm3;
    private final Linear linear1;
    private final Linear linear2;
    private final Linear linear3;

    private final List<Param> params = new ArrayList<Param>();
    private final Param conv1W;
    private final Param conv1B;
    private final Param norm1W;
    private final Param norm1B;
    private final Param conv2W;
    private final Param conv2B;
    private final Param norm2W;
    private final Param norm2B;
    private final Param conv3W;
    private final Param conv3B;
    private final Param norm3W;
    private final Param norm3B;
    private final Param linear1W;
    private final Param linear1B;
    private final Param linear2W;
    private final Param linear2B;
    private final Param linear3W;
    private final Param linear3B;

    private int step = 0;

    public RotationNetTrainer(
        RotationNet net,
        float learningRate,
        float weightDecay,
        float gradClip,
        long seed
    ) {
        this.net = net;
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.gradClip = gradClip;
        this.random = new Random(seed);
        this.time = net.getWindowSize();
        this.inputDim = net.getInputDim();
        this.conv1 = net.getConv1();
        this.conv2 = net.getConv2();
        this.conv3 = net.getConv3();
        this.norm1 = net.getNorm1();
        this.norm2 = net.getNorm2();
        this.norm3 = net.getNorm3();
        this.linear1 = net.getLinear1();
        this.linear2 = net.getLinear2();
        this.linear3 = net.getLinear3();
        this.conv1W = param(conv1.weight);
        this.conv1B = param(conv1.bias);
        this.norm1W = param(norm1.weight);
        this.norm1B = param(norm1.bias);
        this.conv2W = param(conv2.weight);
        this.conv2B = param(conv2.bias);
        this.norm2W = param(norm2.weight);
        this.norm2B = param(norm2.bias);
        this.conv3W = param(conv3.weight);
        this.conv3B = param(conv3.bias);
        this.norm3W = param(norm3.weight);
        this.norm3B = param(norm3.bias);
        this.linear1W = param(linear1.weight);
        this.linear1B = param(linear1.bias);
        this.linear2W = param(linear2.weight);
        this.linear2B = param(linear2.bias);
        this.linear3W = param(linear3.weight);
        this.linear3B = param(linear3.bias);
    }

    private Param param(float[] value) {
        Param param = new Param(value);
        params.add(param);
        return param;
    }

    public RotationNet getNet() {
        return net;
    }

    java.util.List<Param> parameters() {
        return params;
    }

    public float train(
        List<float[]> windows,
        float[] labels,
        int epochs,
        int batchSize,
        float positiveWeight
    ) {
        int count = windows.size();
        if (count == 0) {
            return 0f;
        }
        Integer[] order = new Integer[count];
        for (int i = 0; i < count; i++) {
            order[i] = i;
        }
        float lastLoss = 0f;
        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(order);
            double epochLoss = 0d;
            for (int start = 0; start < count; start += batchSize) {
                int end = Math.min(start + batchSize, count);
                zeroGradients();
                for (int i = start; i < end; i++) {
                    int index = order[i];
                    epochLoss += backwardSample(windows.get(index), labels[index], positiveWeight);
                }
                applyGradients(end - start);
            }
            lastLoss = (float) (epochLoss / count);
        }
        return lastLoss;
    }

    private void shuffle(Integer[] order) {
        for (int i = order.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            Integer tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }

    void zeroGradients() {
        for (Param param : params) {
            float[] grad = param.grad;
            for (int i = 0; i < grad.length; i++) {
                grad[i] = 0f;
            }
        }
    }

    private void applyGradients(int batchCount) {
        float scale = 1f / batchCount;
        double normSquared = 0d;
        for (Param param : params) {
            float[] grad = param.grad;
            for (int i = 0; i < grad.length; i++) {
                double g = grad[i] * scale;
                normSquared += g * g;
            }
        }
        double norm = Math.sqrt(normSquared);
        float clipScale = norm > gradClip && norm > 0d ? (float) (gradClip / norm) : 1f;
        float totalScale = scale * clipScale;

        step++;
        float biasCorrection1 = (float) (1d - Math.pow(beta1, step));
        float biasCorrection2 = (float) (1d - Math.pow(beta2, step));
        for (Param param : params) {
            float[] value = param.value;
            float[] grad = param.grad;
            float[] m = param.m;
            float[] v = param.v;
            for (int i = 0; i < value.length; i++) {
                float g = grad[i] * totalScale;
                m[i] = beta1 * m[i] + (1f - beta1) * g;
                v[i] = beta2 * v[i] + (1f - beta2) * g * g;
                float mHat = m[i] / biasCorrection1;
                float vHat = v[i] / biasCorrection2;
                float update = learningRate * (mHat / ((float) Math.sqrt(vHat) + epsilon));
                value[i] -= learningRate * weightDecay * value[i] + update;
            }
        }
    }

    float backwardSample(float[] window, float label, float positiveWeight) {

        float[] x = new float[inputDim * time];
        float[] featureMean = net.getFeatureMean();
        float[] featureStd = net.getFeatureStd();
        for (int t = 0; t < time; t++) {
            for (int c = 0; c < inputDim; c++) {
                x[c * time + t] = (window[t * inputDim + c] - featureMean[c]) / featureStd[c];
            }
        }

        float[] conv1Out = conv1.forward(x, time);
        float[] mean1 = groupNormStats(norm1, conv1Out);
        float[] inv1 = groupNormInverse(norm1, conv1Out, mean1);
        float[] norm1Out = groupNormApply(norm1, conv1Out, mean1, inv1);
        float[] gelu1 = applyGelu(norm1Out);

        float[] conv2Out = conv2.forward(gelu1, time);
        float[] mean2 = groupNormStats(norm2, conv2Out);
        float[] inv2 = groupNormInverse(norm2, conv2Out, mean2);
        float[] norm2Out = groupNormApply(norm2, conv2Out, mean2, inv2);
        float[] gelu2 = applyGelu(norm2Out);

        float[] conv3Out = conv3.forward(gelu2, time);
        float[] mean3 = groupNormStats(norm3, conv3Out);
        float[] inv3 = groupNormInverse(norm3, conv3Out, mean3);
        float[] norm3Out = groupNormApply(norm3, conv3Out, mean3, inv3);
        float[] gelu3 = applyGelu(norm3Out);

        float[] pooled = new float[conv3.outChannels];
        for (int c = 0; c < pooled.length; c++) {
            float sum = 0f;
            for (int t = 0; t < time; t++) {
                sum += gelu3[c * time + t];
            }
            pooled[c] = sum / time;
        }

        float[] hidden1 = linear1.forward(pooled);
        float[] geluHidden1 = applyGelu(hidden1);
        float[] hidden2 = linear2.forward(geluHidden1);
        float[] geluHidden2 = applyGelu(hidden2);
        float logit = linear3.forward(geluHidden2)[0];

        float sigmoid = RotationNet.sigmoid(logit);
        float loss = (float) (
            positiveWeight * label * softplus(-logit) + (1f - label) * softplus(logit)
        );
        float dLogit = positiveWeight * label * (sigmoid - 1f) + (1f - label) * sigmoid;

        float[] dGeluHidden2 = linearBackward(linear3, linear3W, linear3B, geluHidden2, new float[] { dLogit });
        float[] dHidden2 = geluBackward(hidden2, dGeluHidden2);
        float[] dGeluHidden1 = linearBackward(linear2, linear2W, linear2B, geluHidden1, dHidden2);
        float[] dHidden1 = geluBackward(hidden1, dGeluHidden1);
        float[] dPooled = linearBackward(linear1, linear1W, linear1B, pooled, dHidden1);

        float[] dGelu3 = new float[gelu3.length];
        for (int c = 0; c < pooled.length; c++) {
            float share = dPooled[c] / time;
            for (int t = 0; t < time; t++) {
                dGelu3[c * time + t] = share;
            }
        }

        float[] dNorm3 = groupNormBackward(norm3, norm3W, norm3B, conv3Out, mean3, inv3, geluBackward(gelu3, dGelu3));
        float[] dGelu2 = convBackward(conv3, conv3W, conv3B, gelu2, dNorm3);
        float[] dNorm2 = groupNormBackward(norm2, norm2W, norm2B, conv2Out, mean2, inv2, geluBackward(gelu2, dGelu2));
        float[] dGelu1 = convBackward(conv2, conv2W, conv2B, gelu1, dNorm2);
        float[] dNorm1 = groupNormBackward(norm1, norm1W, norm1B, conv1Out, mean1, inv1, geluBackward(gelu1, dGelu1));
        convBackward(conv1, conv1W, conv1B, x, dNorm1);

        return loss;
    }

    private static float softplus(float value) {
        if (value > 20f) {
            return value;
        }
        if (value < -20f) {
            return (float) Math.exp(value);
        }
        return (float) Math.log1p(Math.exp(value));
    }

    private static float[] applyGelu(float[] values) {
        float[] result = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = Gelu.apply(values[i]);
        }
        return result;
    }

    private static float[] geluBackward(float[] input, float[] dOutput) {
        float[] result = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            float v = input[i];
            float phi = 0.5f * (1f + Gelu.erf(v / SQRT_2));
            float pdf = INV_SQRT_2PI * (float) Math.exp(-0.5d * v * v);
            result[i] = dOutput[i] * (phi + v * pdf);
        }
        return result;
    }

    private float[] groupNormStats(GroupNorm1d norm, float[] input) {
        int elementsPerGroup = (norm.channels / norm.groups) * time;
        float[] mean = new float[norm.groups];
        for (int g = 0; g < norm.groups; g++) {
            int offset = g * elementsPerGroup;
            double sum = 0d;
            for (int i = 0; i < elementsPerGroup; i++) {
                sum += input[offset + i];
            }
            mean[g] = (float) (sum / elementsPerGroup);
        }
        return mean;
    }

    private float[] groupNormInverse(GroupNorm1d norm, float[] input, float[] mean) {
        int elementsPerGroup = (norm.channels / norm.groups) * time;
        float[] inverse = new float[norm.groups];
        for (int g = 0; g < norm.groups; g++) {
            int offset = g * elementsPerGroup;
            double variance = 0d;
            for (int i = 0; i < elementsPerGroup; i++) {
                double diff = input[offset + i] - mean[g];
                variance += diff * diff;
            }
            variance /= elementsPerGroup;
            inverse[g] = (float) (1d / Math.sqrt(variance + norm.epsilon));
        }
        return inverse;
    }

    private static float[] groupNormApply(
        GroupNorm1d norm,
        float[] input,
        float[] mean,
        float[] inverse
    ) {
        int channelsPerGroup = norm.channels / norm.groups;
        int time = input.length / norm.channels;
        float[] output = new float[input.length];
        for (int g = 0; g < norm.groups; g++) {
            for (int c = 0; c < channelsPerGroup; c++) {
                int channel = g * channelsPerGroup + c;
                float weight = norm.weight[channel];
                float bias = norm.bias[channel];
                int offset = channel * time;
                for (int t = 0; t < time; t++) {
                    output[offset + t] =
                        (input[offset + t] - mean[g]) * inverse[g] * weight + bias;
                }
            }
        }
        return output;
    }

    private float[] groupNormBackward(
        GroupNorm1d norm,
        Param weightParam,
        Param biasParam,
        float[] input,
        float[] mean,
        float[] inverse,
        float[] dOutput
    ) {
        int channelsPerGroup = norm.channels / norm.groups;
        int elementsPerGroup = channelsPerGroup * time;
        float[] dInput = new float[input.length];
        for (int g = 0; g < norm.groups; g++) {
            int offset = g * elementsPerGroup;
            double sumDxhat = 0d;
            double sumDxhatXhat = 0d;
            for (int i = 0; i < elementsPerGroup; i++) {
                int index = offset + i;
                int channel = offset / time + i / time;
                float xhat = (input[index] - mean[g]) * inverse[g];
                float dxhat = dOutput[index] * norm.weight[channel];
                sumDxhat += dxhat;
                sumDxhatXhat += dxhat * xhat;
            }
            float meanDxhat = (float) (sumDxhat / elementsPerGroup);
            float meanDxhatXhat = (float) (sumDxhatXhat / elementsPerGroup);
            for (int i = 0; i < elementsPerGroup; i++) {
                int index = offset + i;
                int channel = offset / time + i / time;
                float xhat = (input[index] - mean[g]) * inverse[g];
                float dxhat = dOutput[index] * norm.weight[channel];
                dInput[index] = inverse[g] * (dxhat - meanDxhat - xhat * meanDxhatXhat);
                weightParam.grad[channel] += dOutput[index] * xhat;
                biasParam.grad[channel] += dOutput[index];
            }
        }
        return dInput;
    }

    private float[] linearBackward(
        Linear layer,
        Param weightParam,
        Param biasParam,
        float[] input,
        float[] dOutput
    ) {
        float[] dInput = new float[layer.inFeatures];
        for (int o = 0; o < layer.outFeatures; o++) {
            float d = dOutput[o];
            int offset = o * layer.inFeatures;
            for (int i = 0; i < layer.inFeatures; i++) {
                dInput[i] += layer.weight[offset + i] * d;
            }
        }
        for (int o = 0; o < layer.outFeatures; o++) {
            float d = dOutput[o];
            int offset = o * layer.inFeatures;
            for (int i = 0; i < layer.inFeatures; i++) {
                weightParam.grad[offset + i] += d * input[i];
            }
            biasParam.grad[o] += d;
        }
        return dInput;
    }

    private float[] convBackward(
        Conv1d layer,
        Param weightParam,
        Param biasParam,
        float[] input,
        float[] dOutput
    ) {
        int inputTime = input.length / layer.inChannels;
        int width = inputTime + 2 * layer.padding;
        float[] padded = new float[layer.inChannels * width];
        for (int c = 0; c < layer.inChannels; c++) {
            System.arraycopy(input, c * inputTime, padded, c * width + layer.padding, inputTime);
        }
        float[] dPadded = new float[layer.inChannels * width];
        for (int o = 0; o < layer.outChannels; o++) {
            int outOffset = o * inputTime;
            int weightOffset = o * layer.inChannels * layer.kernel;
            float dBias = 0f;
            for (int t = 0; t < inputTime; t++) {
                dBias += dOutput[outOffset + t];
            }
            biasParam.grad[o] += dBias;
            for (int c = 0; c < layer.inChannels; c++) {
                int paddedOffset = c * width;
                int weightBase = weightOffset + c * layer.kernel;
                for (int k = 0; k < layer.kernel; k++) {
                    float w = layer.weight[weightBase + k];
                    float gradSum = 0f;
                    for (int t = 0; t < inputTime; t++) {
                        float dy = dOutput[outOffset + t];
                        gradSum += dy * padded[paddedOffset + t + k];
                        dPadded[paddedOffset + t + k] += w * dy;
                    }
                    weightParam.grad[weightBase + k] += gradSum;
                }
            }
        }
        float[] dInput = new float[layer.inChannels * inputTime];
        for (int c = 0; c < layer.inChannels; c++) {
            System.arraycopy(dPadded, c * width + layer.padding, dInput, c * inputTime, inputTime);
        }
        return dInput;
    }

    static final class Param {

        final float[] value;
        final float[] grad;
        final float[] m;
        final float[] v;

        Param(float[] value) {
            this.value = value;
            this.grad = new float[value.length];
            this.m = new float[value.length];
            this.v = new float[value.length];
        }
    }
}
