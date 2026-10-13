package me.katze.powerac.module.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import me.katze.powerac.ai.DetectionResult;
import me.katze.powerac.ai.RotationModelManager;
import me.katze.powerac.ai.TrainOutcome;
import me.katze.powerac.module.Module;
import me.katze.powerac.module.ModuleInfo;
import me.katze.powerac.object.AIRotationData;
import me.katze.powerac.player.PowerPlayer;
import me.katze.powerac.utility.StringUtility;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

@ModuleInfo("Rotation")
public final class RotationModule extends Module {

    private static final int TRAIN_SAMPLES_REQUIRED = 300;

    private final Object lock = new Object();
    private List<AIRotationData> delayedBatch;
    private final List<AIRotationData> trainingSamples = new ArrayList<>();
    private int trainingLabel = -1;
    private CommandSender trainingSender;
    private boolean continuousTraining;
    private boolean inFlight;
    private boolean queued;
    private boolean sendScheduled;
    private boolean trainingActive;
    private boolean trainingInFlight;
    private int trainingSentBatches;

    public RotationModule(
        me.katze.powerac.PowerAC plugin,
        me.katze.powerac.manager.PlayerManager playerManager,
        PowerPlayer player
    ) {
        super(plugin, playerManager, player);
    }

    public void enqueueAiBatch(List<AIRotationData> batch) {
        if (batch == null || batch.isEmpty()) {
            return;
        }
        if (collectTrainingSamples(batch)) {
            return;
        }
        boolean shouldSchedule = false;
        synchronized (lock) {
            if (inFlight || queued) {
                delayedBatch = batch;
                return;
            }
            delayedBatch = batch;
            queued = true;
            shouldSchedule = true;
        }
        if (shouldSchedule) {
            scheduleNextSend();
        }
    }

    public boolean startTrainingSession(CommandSender sender, int label, boolean continuous) {
        if (label != 0 && label != 1) {
            return false;
        }
        RotationModelManager manager = getPlugin().getRotationModelManager();
        if (manager == null) {
            sender.sendMessage(
                StringUtility.getString(
                    getPlugin()
                        .getConfigManager()
                        .getTrainMessage("error", "&cTraining error: {details}")
                        .replace("{details}", "model manager is not available")
                )
            );
            return true;
        }

        boolean started;
        synchronized (lock) {
            if (trainingActive || trainingInFlight) {
                started = false;
            } else {
                if (getPlayer().getRotationTracker() != null) {
                    getPlayer().getRotationTracker().reset();
                }
                trainingSamples.clear();
                delayedBatch = null;
                queued = false;
                trainingLabel = label;
                trainingSentBatches = 0;
                trainingSender = sender;
                continuousTraining = continuous;
                trainingActive = true;
                started = true;
            }
        }

        if (!started) {
            sender.sendMessage(
                StringUtility.getString(
                    getPlugin()
                        .getConfigManager()
                        .getTrainMessage(
                            "already-running",
                            "&cA training session is already running."
                        )
                )
            );
            return true;
        }

        sender.sendMessage(
            StringUtility.getString(
                getPlugin()
                    .getConfigManager()
                    .getTrainMessage(
                        "collecting-started",
                        "&aCollecting {samples} samples started{mode}."
                    )
                    .replace("{samples}", Integer.toString(TRAIN_SAMPLES_REQUIRED))
                    .replace("{mode}", continuous ? " in loop mode" : "")
            )
        );
        return true;
    }

    public boolean stopTrainingSession() {
        synchronized (lock) {
            if (!trainingActive && !trainingInFlight) {
                return false;
            }
            trainingSamples.clear();
            delayedBatch = null;
            queued = false;
            trainingActive = false;
            trainingInFlight = false;
            continuousTraining = false;
            trainingLabel = -1;
            trainingSentBatches = 0;
            trainingSender = null;
            return true;
        }
    }

    public void clearTrainingData(CommandSender sender) {
        RotationModelManager manager = getPlugin().getRotationModelManager();
        if (manager != null) {
            manager.clear();
        }
        sender.sendMessage(
            StringUtility.getString(
                getPlugin()
                    .getConfigManager()
                    .getTrainMessage(
                        "cleared",
                        "&aYour trained model data has been cleared."
                    )
            )
        );
    }

    public boolean isTrainingSessionRunning() {
        synchronized (lock) {
            return trainingActive || trainingInFlight;
        }
    }

    private void scheduleNextSend() {
        synchronized (lock) {
            if (sendScheduled) {
                return;
            }
            sendScheduled = true;
        }
        getPlugin().getTaskScheduler().runAsync(this::processQueue);
    }

    private void processQueue() {
        List<AIRotationData> toSend;
        synchronized (lock) {
            sendScheduled = false;
            if (!queued || delayedBatch == null) {
                return;
            }
            toSend = delayedBatch;
            delayedBatch = null;
            queued = false;
            inFlight = true;
        }
        scoreBatch(toSend);
    }

    private void scoreBatch(List<AIRotationData> batch) {
        RotationModelManager manager = getPlugin().getRotationModelManager();
        final DetectionResult result = manager == null
            ? DetectionResult.unavailable("model manager is not available")
            : manager.score(batch);

        getPlugin()
            .getTaskScheduler()
            .runPlayer(getPlayer().getUuid(), () -> onDetectionResult(result));
        onRequestFinished();
    }

    private void onRequestFinished() {
        boolean shouldSchedule = false;
        synchronized (lock) {
            inFlight = false;
            if (delayedBatch != null && !queued) {
                queued = true;
                shouldSchedule = true;
            }
        }
        if (shouldSchedule) {
            scheduleNextSend();
        }
    }

    private void onDetectionResult(DetectionResult result) {
        if (result == null) {
            return;
        }
        if (result.probability >= 0f) {
            getPlayerManager()
                .publishMonitorUpdate(
                    getPlayer().getUuid(),
                    getPlayer().getName(),
                    result.probability,
                    result.detected,
                    true
                );
        }
        if (!result.detected) {
            return;
        }

        double probability = result.probability;
        String probabilityText = probability < 0D
            ? "?"
            : String.format(Locale.US, "%.3f", probability);
        String reason = result.reason == null ? "" : result.reason;
        String details = reason.trim().isEmpty()
            ? "AI rotation detection (local), p: " + probabilityText
            : "AI rotation detection (local), p: " + probabilityText + ", " + reason;
        getPlayer().getViolationManager().flag(this, probability, details, true);
    }

    private boolean collectTrainingSamples(List<AIRotationData> batch) {
        List<AIRotationData> toSend = null;
        synchronized (lock) {
            if (!trainingActive || trainingInFlight) {
                return false;
            }
            trainingSamples.addAll(batch);
            if (trainingSamples.size() >= TRAIN_SAMPLES_REQUIRED) {
                toSend = new ArrayList<>(
                    trainingSamples.subList(0, TRAIN_SAMPLES_REQUIRED)
                );
                trainingSamples.clear();
                trainingInFlight = true;
                trainingActive = false;
            }
        }
        if (toSend != null) {
            final List<AIRotationData> payload = toSend;
            getPlugin().getTaskScheduler().runAsync(() -> runTraining(payload));
            return true;
        }
        return trainingActive;
    }

    private void runTraining(List<AIRotationData> batch) {
        int label;
        CommandSender sender;
        boolean restart;
        synchronized (lock) {
            label = trainingLabel;
            sender = trainingSender;
            restart = continuousTraining;
        }

        RotationModelManager manager = getPlugin().getRotationModelManager();
        TrainOutcome outcome = manager == null
            ? null
            : manager.train(batch, label);

        synchronized (lock) {
            trainingInFlight = false;
            if (
                outcome != null &&
                (outcome.success || outcome.waitingForData) &&
                outcome.legitCount + outcome.cheaterCount > 0
            ) {
                trainingSentBatches += 1;
            }
            if (restart && outcome != null && outcome.success) {
                trainingActive = true;
                if (getPlayer().getRotationTracker() != null) {
                    getPlayer().getRotationTracker().reset();
                }
            } else {
                trainingActive = false;
                trainingLabel = -1;
                trainingSender = null;
                continuousTraining = false;
            }
        }

        if (sender != null) {
            runOnSender(sender, () -> notifyTrainingOutcome(sender, outcome, restart));
        }
    }

    private void notifyTrainingOutcome(
        CommandSender sender,
        TrainOutcome outcome,
        boolean restart
    ) {
        if (outcome == null) {
            sender.sendMessage(
                StringUtility.getString(
                    getPlugin()
                        .getConfigManager()
                        .getTrainMessage("error", "&cTraining error: {details}")
                        .replace("{details}", "model manager is not available")
                )
            );
            return;
        }

        int sentCount;
        synchronized (lock) {
            sentCount = trainingSentBatches;
        }

        if (outcome.success) {
            String path = restart ? "sent-continuous" : "trained";
            String fallback = restart
                ? "&aModel trained. Collecting next batch... Batches sent: &f{count}"
                : "&aModel trained. loss: &f{loss}&a, legit: &f{legit}&a, cheater: &f{cheater}&a. Batches: &f{count}";
            sender.sendMessage(
                StringUtility.getString(
                    getPlugin()
                        .getConfigManager()
                        .getTrainMessage(path, fallback)
                        .replace("{count}", Integer.toString(sentCount))
                        .replace("{loss}", String.format(Locale.US, "%.4f", outcome.loss))
                        .replace("{legit}", Integer.toString(outcome.legitCount))
                        .replace("{cheater}", Integer.toString(outcome.cheaterCount))
                )
            );
            return;
        }

        if (outcome.waitingForData) {
            sender.sendMessage(
                StringUtility.getString(
                    getPlugin()
                        .getConfigManager()
                        .getTrainMessage(
                            "waiting",
                            "&eData stored (legit: &f{legit}&e, cheater: &f{cheater}&e). Need more of the other class, then retrain."
                        )
                        .replace("{legit}", Integer.toString(outcome.legitCount))
                        .replace("{cheater}", Integer.toString(outcome.cheaterCount))
                )
            );
            return;
        }

        sender.sendMessage(
            StringUtility.getString(
                getPlugin()
                    .getConfigManager()
                    .getTrainMessage("error", "&cTraining error: {details}")
                    .replace("{details}", outcome.message == null ? "unknown" : outcome.message)
            )
        );
    }

    private void runOnSender(CommandSender sender, Runnable task) {
        if (sender instanceof Player) {
            getPlugin().getTaskScheduler().runPlayer((Player) sender, task);
            return;
        }
        getPlugin().getTaskScheduler().runGlobal(task);
    }
}
