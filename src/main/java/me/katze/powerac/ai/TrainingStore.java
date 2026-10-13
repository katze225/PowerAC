package me.katze.powerac.ai;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public final class TrainingStore {

    private final Path path;
    private final int inputDim;
    private final int windowSize;
    private final int maxPerLabel;
    private final Random random = new Random();

    private final List<float[]> legit = new ArrayList<float[]>();
    private final List<float[]> cheater = new ArrayList<float[]>();
    private boolean loaded = false;

    public TrainingStore(Path path, int inputDim, int windowSize, int maxPerLabel) {
        this.path = path;
        this.inputDim = inputDim;
        this.windowSize = windowSize;
        this.maxPerLabel = Math.max(1, maxPerLabel);
    }

    public synchronized void load() {
        if (loaded) {
            return;
        }
        loaded = true;
        if (!Files.exists(path)) {
            return;
        }
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            JsonReader json = new JsonReader(reader);
            json.beginObject();
            while (json.hasNext()) {
                String name = json.nextName();
                List<float[]> target = "1".equals(name)
                    ? cheater
                    : ("0".equals(name) ? legit : null);
                json.beginArray();
                while (json.hasNext()) {
                    float[] window = new float[inputDim * windowSize];
                    if (target != null) {
                        readWindow(json, window);
                        target.add(window);
                    } else {
                        skipValue(json);
                    }
                }
                json.endArray();
                if (target != null && target.size() > maxPerLabel) {
                    target.subList(0, target.size() - maxPerLabel).clear();
                }
            }
            json.endObject();
        } catch (IOException | RuntimeException exception) {
            legit.clear();
            cheater.clear();
        }
    }

    public synchronized void addBatch(int label, List<float[]> windows) {
        load();
        List<float[]> target = label == 1 ? cheater : legit;
        for (float[] window : windows) {
            if (window.length == inputDim * windowSize) {
                target.add(window);
            }
        }
        if (target.size() > maxPerLabel) {
            target.subList(0, target.size() - maxPerLabel).clear();
        }
        persist();
    }

    public synchronized int count(int label) {
        load();
        return (label == 1 ? cheater : legit).size();
    }

    public synchronized int total() {
        load();
        return legit.size() + cheater.size();
    }

    public synchronized boolean hasBothLabels() {
        load();
        return !legit.isEmpty() && !cheater.isEmpty();
    }

    public synchronized Dataset asBalancedDataset(int maxPerLabel) {
        load();
        int count = Math.min(Math.min(legit.size(), cheater.size()), maxPerLabel);
        List<float[]> windows = new ArrayList<float[]>(count * 2);
        float[] labels = new float[count * 2];
        windows.addAll(sample(legit, count));
        windows.addAll(sample(cheater, count));
        for (int i = 0; i < count; i++) {
            labels[i] = 0f;
            labels[count + i] = 1f;
        }
        if (windows.isEmpty()) {
            windows.addAll(legit);
            windows.addAll(cheater);
            labels = new float[windows.size()];
            for (int i = 0; i < legit.size(); i++) {
                labels[i] = 0f;
            }
            for (int i = 0; i < cheater.size(); i++) {
                labels[legit.size() + i] = 1f;
            }
        }
        return new Dataset(windows, labels);
    }

    public synchronized TrainingStore clear() {
        legit.clear();
        cheater.clear();
        loaded = true;
        try {
            Files.deleteIfExists(path);
        } catch (IOException ignored) {
        }
        return this;
    }

    private List<float[]> sample(List<float[]> source, int count) {
        if (count >= source.size()) {
            return new ArrayList<float[]>(source);
        }
        List<float[]> copy = new ArrayList<float[]>(source);
        Collections.shuffle(copy, random);
        return new ArrayList<float[]>(copy.subList(0, count));
    }

    private void persist() {
        Path parent = path.getParent();
        try {
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path temp = path.resolveSibling(path.getFileName() + ".tmp");
            try (BufferedWriter writer = Files.newBufferedWriter(temp, StandardCharsets.UTF_8)) {
                JsonWriter json = new JsonWriter(writer);
                json.beginObject();
                json.name("0").beginArray();
                for (float[] window : legit) {
                    writeWindow(json, window);
                }
                json.endArray();
                json.name("1").beginArray();
                for (float[] window : cheater) {
                    writeWindow(json, window);
                }
                json.endArray();
                json.endObject();
                json.flush();
            }
            moveAtomic(temp, path);
        } catch (IOException ignored) {
        }
    }

    private void writeWindow(JsonWriter json, float[] window) throws IOException {
        json.beginArray();
        for (int t = 0; t < windowSize; t++) {
            json.beginArray();
            int offset = t * inputDim;
            for (int f = 0; f < inputDim; f++) {
                json.value(window[offset + f]);
            }
            json.endArray();
        }
        json.endArray();
    }

    private void readWindow(JsonReader json, float[] window) throws IOException {
        int position = 0;
        json.beginArray();
        while (json.hasNext()) {
            readRow(json, window, position);
            position += inputDim;
        }
        json.endArray();
    }

    private void readRow(JsonReader json, float[] window, int offset) throws IOException {
        json.beginArray();
        int i = 0;
        while (json.hasNext() && i < inputDim) {
            window[offset + i] = (float) json.nextDouble();
            i++;
        }
        while (json.hasNext()) {
            json.skipValue();
        }
        json.endArray();
    }

    private static void skipValue(JsonReader json) throws IOException {
        JsonToken token = json.peek();
        if (token == JsonToken.BEGIN_ARRAY) {
            json.beginArray();
            while (json.hasNext()) {
                skipValue(json);
            }
            json.endArray();
        } else if (token == JsonToken.BEGIN_OBJECT) {
            json.beginObject();
            while (json.hasNext()) {
                json.nextName();
                skipValue(json);
            }
            json.endObject();
        } else if (token == JsonToken.NULL) {
            json.nextNull();
        } else {
            json.skipValue();
        }
    }

    static void moveAtomic(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException exception) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    public static final class Dataset {

        public final List<float[]> windows;
        public final float[] labels;

        Dataset(List<float[]> windows, float[] labels) {
            this.windows = windows;
            this.labels = labels;
        }
    }
}
