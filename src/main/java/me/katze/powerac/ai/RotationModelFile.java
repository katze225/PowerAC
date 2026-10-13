package me.katze.powerac.ai;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;

public final class RotationModelFile {

    private RotationModelFile() {}

    public static void write(RotationModel model, Writer writer) throws IOException {
        RotationNet net = model.getNet();
        AggregationConfig aggregation = model.getAggregation();
        JsonWriter json = new JsonWriter(writer);
        json.setLenient(true);
        json.beginObject();
        json.name("format").value(RotationNet.FORMAT);
        json.name("input_dim").value(net.getInputDim());
        json.name("window_size").value(net.getWindowSize());
        json.name("hidden_dim").value(net.getHiddenDim());
        json.name("total_samples").value(net.getTotalSamples());
        writeArray(json, "feature_mean", net.getFeatureMean());
        writeArray(json, "feature_std", net.getFeatureStd());
        json.name("aggregation").beginObject();
        json.name("decision").value(aggregation.decision);
        json.name("suspicious").value(aggregation.suspicious);
        json.name("ratio_gate").value(aggregation.ratioGate);
        json.name("top_fraction").value(aggregation.topFraction);
        json.name("mean_weight").value(aggregation.meanWeight);
        json.name("median_weight").value(aggregation.medianWeight);
        json.name("top_boost").value(aggregation.topBoost);
        json.name("spread_penalty").value(aggregation.spreadPenalty);
        json.endObject();
        json.name("layers").beginArray();
        writeConv1d(json, net.getConv1());
        writeGroupNorm(json, net.getNorm1());
        writeConv1d(json, net.getConv2());
        writeGroupNorm(json, net.getNorm2());
        writeConv1d(json, net.getConv3());
        writeGroupNorm(json, net.getNorm3());
        writeLinear(json, net.getLinear1());
        writeLinear(json, net.getLinear2());
        writeLinear(json, net.getLinear3());
        json.endArray();
        json.endObject();
        json.flush();
    }

    private static void writeArray(JsonWriter json, String name, float[] values) throws IOException {
        json.name(name).beginArray();
        for (float value : values) {
            json.value(value);
        }
        json.endArray();
    }

    private static void writeConv1d(JsonWriter json, Conv1d conv) throws IOException {
        json.beginObject();
        json.name("type").value("conv1d");
        json.name("in").value(conv.inChannels);
        json.name("out").value(conv.outChannels);
        json.name("kernel").value(conv.kernel);
        json.name("padding").value(conv.padding);
        json.name("weight").beginArray();
        for (int o = 0; o < conv.outChannels; o++) {
            json.beginArray();
            for (int c = 0; c < conv.inChannels; c++) {
                json.beginArray();
                int base = (o * conv.inChannels + c) * conv.kernel;
                for (int k = 0; k < conv.kernel; k++) {
                    json.value(conv.weight[base + k]);
                }
                json.endArray();
            }
            json.endArray();
        }
        json.endArray();
        writeArray(json, "bias", conv.bias);
        json.endObject();
    }

    private static void writeGroupNorm(JsonWriter json, GroupNorm1d norm) throws IOException {
        json.beginObject();
        json.name("type").value("groupnorm");
        json.name("groups").value(norm.groups);
        json.name("channels").value(norm.channels);
        writeArray(json, "weight", norm.weight);
        writeArray(json, "bias", norm.bias);
        json.endObject();
    }

    private static void writeLinear(JsonWriter json, Linear linear) throws IOException {
        json.beginObject();
        json.name("type").value("linear");
        json.name("in").value(linear.inFeatures);
        json.name("out").value(linear.outFeatures);
        json.name("weight").beginArray();
        for (int o = 0; o < linear.outFeatures; o++) {
            json.beginArray();
            int base = o * linear.inFeatures;
            for (int i = 0; i < linear.inFeatures; i++) {
                json.value(linear.weight[base + i]);
            }
            json.endArray();
        }
        json.endArray();
        writeArray(json, "bias", linear.bias);
        json.endObject();
    }

    public static RotationModel parse(Reader reader) throws IOException {
        JsonReader json = new JsonReader(reader);
        json.setLenient(true);

        int inputDim = 10;
        int windowSize = 64;
        int hiddenDim = 512;
        long totalSamples = 0L;
        float[] featureMean = null;
        float[] featureStd = null;
        AggregationConfig aggregation = AggregationConfig.defaults();

        Conv1d[] convolutions = new Conv1d[3];
        GroupNorm1d[] normalizations = new GroupNorm1d[3];
        Linear[] linears = new Linear[3];
        int convolutionIndex = 0;
        int normalizationIndex = 0;
        int linearIndex = 0;

        json.beginObject();
        while (json.hasNext()) {
            String name = json.nextName();
            if ("input_dim".equals(name)) {
                inputDim = json.nextInt();
            } else if ("window_size".equals(name)) {
                windowSize = json.nextInt();
            } else if ("hidden_dim".equals(name)) {
                hiddenDim = json.nextInt();
            } else if ("total_samples".equals(name)) {
                totalSamples = json.nextLong();
            } else if ("feature_mean".equals(name)) {
                featureMean = readFloats(json, inputDim);
            } else if ("feature_std".equals(name)) {
                featureStd = readFloats(json, inputDim);
            } else if ("aggregation".equals(name)) {
                aggregation = readAggregation(json);
            } else if ("layers".equals(name)) {
                json.beginArray();
                while (json.hasNext()) {
                    json.beginObject();
                    String type = null;
                    int in = 0;
                    int out = 0;
                    int kernel = 3;
                    int padding = 1;
                    int groups = 8;
                    int channels = 0;
                    float[] weight = null;
                    float[] bias = null;
                    while (json.hasNext()) {
                        String field = json.nextName();
                        if ("type".equals(field)) {
                            type = json.nextString();
                        } else if ("in".equals(field)) {
                            in = json.nextInt();
                        } else if ("out".equals(field)) {
                            out = json.nextInt();
                        } else if ("kernel".equals(field)) {
                            kernel = json.nextInt();
                        } else if ("padding".equals(field)) {
                            padding = json.nextInt();
                        } else if ("groups".equals(field)) {
                            groups = json.nextInt();
                        } else if ("channels".equals(field)) {
                            channels = json.nextInt();
                        } else if ("weight".equals(field)) {
                            weight = readFloats(json, weightLength(type, in, out, kernel, channels));
                        } else if ("bias".equals(field)) {
                            bias = readFloats(json, biasLength(type, out, channels));
                        } else {
                            json.skipValue();
                        }
                    }
                    json.endObject();
                    if ("conv1d".equals(type)) {
                        convolutions[convolutionIndex++] =
                            new Conv1d(in, out, kernel, padding, weight, bias);
                    } else if ("groupnorm".equals(type)) {
                        normalizations[normalizationIndex++] =
                            new GroupNorm1d(
                                groups,
                                channels,
                                GroupNorm1d.DEFAULT_EPSILON,
                                weight,
                                bias
                            );
                    } else if ("linear".equals(type)) {
                        linears[linearIndex++] = new Linear(in, out, weight, bias);
                    }
                }
                json.endArray();
            } else {
                json.skipValue();
            }
        }
        json.endObject();
        json.close();

        if (featureMean == null || featureStd == null) {
            throw new IOException("rotation model is missing feature_mean/feature_std");
        }

        RotationNet net = new RotationNet(
            inputDim,
            windowSize,
            hiddenDim,
            totalSamples,
            featureMean,
            featureStd,
            convolutions[0],
            convolutions[1],
            convolutions[2],
            normalizations[0],
            normalizations[1],
            normalizations[2],
            linears[0],
            linears[1],
            linears[2]
        );
        return new RotationModel(net, aggregation);
    }

    private static int weightLength(
        String type,
        int in,
        int out,
        int kernel,
        int channels
    ) {
        if ("conv1d".equals(type)) {
            return out * in * kernel;
        }
        if ("linear".equals(type)) {
            return out * in;
        }
        return channels;
    }

    private static int biasLength(String type, int out, int channels) {
        return "groupnorm".equals(type) ? channels : out;
    }

    private static AggregationConfig readAggregation(JsonReader json) throws IOException {
        AggregationConfig defaults = AggregationConfig.defaults();
        float decision = defaults.decision;
        float suspicious = defaults.suspicious;
        float ratioGate = defaults.ratioGate;
        float topFraction = defaults.topFraction;
        float meanWeight = defaults.meanWeight;
        float medianWeight = defaults.medianWeight;
        float topBoost = defaults.topBoost;
        float spreadPenalty = defaults.spreadPenalty;
        json.beginObject();
        while (json.hasNext()) {
            String name = json.nextName();
            if ("decision".equals(name)) {
                decision = (float) json.nextDouble();
            } else if ("suspicious".equals(name)) {
                suspicious = (float) json.nextDouble();
            } else if ("ratio_gate".equals(name)) {
                ratioGate = (float) json.nextDouble();
            } else if ("top_fraction".equals(name)) {
                topFraction = (float) json.nextDouble();
            } else if ("mean_weight".equals(name)) {
                meanWeight = (float) json.nextDouble();
            } else if ("median_weight".equals(name)) {
                medianWeight = (float) json.nextDouble();
            } else if ("top_boost".equals(name)) {
                topBoost = (float) json.nextDouble();
            } else if ("spread_penalty".equals(name)) {
                spreadPenalty = (float) json.nextDouble();
            } else {
                json.skipValue();
            }
        }
        json.endObject();
        return new AggregationConfig(
            decision,
            suspicious,
            ratioGate,
            topFraction,
            meanWeight,
            medianWeight,
            topBoost,
            spreadPenalty
        );
    }

    private static float[] readFloats(JsonReader json, int expectedLength) throws IOException {
        float[] target = new float[expectedLength];
        int[] position = new int[] { 0 };
        readFloatsRecursive(json, target, position);
        if (position[0] != expectedLength) {
            throw new IOException(
                "expected " + expectedLength + " values but read " + position[0]
            );
        }
        return target;
    }

    private static void readFloatsRecursive(
        JsonReader json,
        float[] target,
        int[] position
    ) throws IOException {
        JsonToken token = json.peek();
        if (token == JsonToken.BEGIN_ARRAY) {
            json.beginArray();
            while (json.hasNext()) {
                readFloatsRecursive(json, target, position);
            }
            json.endArray();
            return;
        }
        if (token == JsonToken.NULL) {
            json.nextNull();
            target[position[0]++] = 0f;
            return;
        }
        target[position[0]++] = (float) json.nextDouble();
    }
}
