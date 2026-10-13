package me.katze.powerac.ai;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import me.katze.powerac.PowerAC;
import me.katze.powerac.object.AIRotationData;

public final class RotationModelManager {

    public static final int INPUT_DIM = 10;
    public static final int WINDOW_SIZE = 64;
    public static final int WINDOW_STRIDE = 8;

    private static final String MODEL_RESOURCE = "rotation.json";

    private final PowerAC plugin;
    private volatile ModelSettings settings;

    private final File modelFile;
    private final File trainingFile;

    private volatile RotationModel model;
    private volatile TrainingStore trainingStore;
    private volatile boolean training;
    private volatile float lastLoss = -1f;

    private final Object trainLock = new Object();

    public RotationModelManager(PowerAC plugin, ModelSettings settings) {
        this.plugin = plugin;
        this.settings = settings;
        this.modelFile = new File(plugin.getDataFolder(), MODEL_RESOURCE);
        this.trainingFile = new File(plugin.getDataFolder(), "rotation_replay.json");
        this.trainingStore =
            new TrainingStore(trainingFile.toPath(), INPUT_DIM, WINDOW_SIZE, settings.maxWindowsPerLabel);
    }

    public void applySettings(ModelSettings settings) {
        this.settings = settings;
        this.trainingStore =
            new TrainingStore(trainingFile.toPath(), INPUT_DIM, WINDOW_SIZE, settings.maxWindowsPerLabel);
    }

    public ModelSettings getSettings() {
        return settings;
    }

    public synchronized void reload() {
        model = null;
        if (!modelFile.exists()) {
            try {
                plugin.saveResource(MODEL_RESOURCE, false);
            } catch (IllegalArgumentException exception) {
                plugin
                    .getLogger()
                    .warning("No bundled rotation model found; waiting for local training.");
            }
        }
        if (!modelFile.exists()) {
            return;
        }
        try (BufferedReader reader = Files.newBufferedReader(modelFile.toPath(), StandardCharsets.UTF_8)) {
            model = RotationModelFile.parse(reader);
            plugin
                .getLogger()
                .info(
                    "Rotation model loaded (hidden=" +
                    model.getNet().getHiddenDim() +
                    ", samples=" +
                    model.getNet().getTotalSamples() +
                    ")"
                );
        } catch (IOException | RuntimeException exception) {
            model = null;
            plugin.getLogger().severe("Failed to load rotation model: " + exception.getMessage());
        }
    }

    public boolean isReady() {
        return model != null;
    }

    public boolean isTraining() {
        return training;
    }

    public float getLastLoss() {
        return lastLoss;
    }

    public int getLegitCount() {
        return trainingStore.count(0);
    }

    public int getCheaterCount() {
        return trainingStore.count(1);
    }

    public int getHiddenDim() {
        RotationModel current = model;
        return current == null ? 0 : current.getNet().getHiddenDim();
    }

    public long getTotalSamples() {
        RotationModel current = model;
        return current == null ? 0L : current.getNet().getTotalSamples();
    }

    public File getModelFile() {
        return modelFile;
    }

    public DetectionResult score(List<AIRotationData> samples) {
        RotationModel current = model;
        if (current == null) {
            return DetectionResult.unavailable("rotation model is not loaded");
        }
        List<float[]> windows = toWindows(samples);
        if (windows.isEmpty()) {
            return DetectionResult.unavailable("not enough rotation data");
        }
        float[] probabilities = current.getNet().predictBatch(windows);
        float score = current.getScorer().score(probabilities);
        boolean detected = current.getScorer().isDetected(score);
        return new DetectionResult(
            score,
            detected,
            score,
            windows.size(),
            detected ? "AI rotation detection" : null
        );
    }

    public TrainOutcome train(List<AIRotationData> samples, int label) {
        if (training) {
            return TrainOutcome.failure("training already in progress", getLegitCount(), getCheaterCount());
        }
        synchronized (trainLock) {
            training = true;
            try {
                RotationModel current = model;
                if (current == null) {
                    return TrainOutcome.failure("rotation model is not loaded", 0, 0);
                }
                List<float[]> windows = toWindows(samples);
                if (windows.isEmpty()) {
                    return TrainOutcome.failure("not enough rotation data in batch", getLegitCount(), getCheaterCount());
                }
                trainingStore.addBatch(label, windows);
                int legitCount = trainingStore.count(0);
                int cheaterCount = trainingStore.count(1);
                ModelSettings currentSettings = settings;
                if (
                    legitCount < currentSettings.minWindowsPerClass ||
                    cheaterCount < currentSettings.minWindowsPerClass
                ) {
                    return TrainOutcome.waiting(legitCount, cheaterCount);
                }

                TrainingStore.Dataset dataset = trainingStore.asBalancedDataset(
                    currentSettings.maxWindowsPerLabel
                );
                RotationNet netCopy = current.getNet().copy();
                netCopy.setTotalSamples(trainingStore.total());
                float positiveWeight = (float) legitCount / Math.max(1, cheaterCount);

                RotationNetTrainer trainer = new RotationNetTrainer(
                    netCopy,
                    currentSettings.learningRate,
                    currentSettings.weightDecay,
                    currentSettings.gradClip,
                    System.nanoTime()
                );
                float loss = trainer.train(
                    dataset.windows,
                    dataset.labels,
                    currentSettings.epochs,
                    currentSettings.batchSize,
                    positiveWeight
                );

                RotationModel trained = new RotationModel(netCopy, current.getAggregation());
                save(trained);
                model = trained;
                lastLoss = loss;
                plugin
                    .getLogger()
                    .info(
                        "Rotation model trained: loss=" +
                        loss +
                        ", legit=" +
                        legitCount +
                        ", cheater=" +
                        cheaterCount
                    );
                return TrainOutcome.success(loss, legitCount, cheaterCount);
            } catch (RuntimeException | IOException exception) {
                plugin.getLogger().severe("Rotation training failed: " + exception.getMessage());
                return TrainOutcome.failure(exception.getMessage(), getLegitCount(), getCheaterCount());
            } finally {
                training = false;
            }
        }
    }

    public void clear() {
        synchronized (trainLock) {
            trainingStore.clear();
            lastLoss = -1f;
            model = null;
            try {
                Files.deleteIfExists(modelFile.toPath());
            } catch (IOException ignored) {
            }
            reload();
        }
    }

    public List<float[]> toWindows(List<AIRotationData> samples) {
        return RotationWindows.toWindows(samples);
    }

    private void save(RotationModel trained) throws IOException {
        Path target = modelFile.toPath();
        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temp = target.resolveSibling(modelFile.getName() + ".tmp");
        try (BufferedWriter writer = Files.newBufferedWriter(temp, StandardCharsets.UTF_8)) {
            RotationModelFile.write(trained, writer);
        }
        TrainingStore.moveAtomic(temp, target);
    }
}
