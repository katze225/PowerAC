package me.katze.powerac.ai;

import java.util.ArrayList;
import java.util.List;
import me.katze.powerac.object.AIRotationData;

public final class RotationWindows {

    private RotationWindows() {}

    public static List<float[]> toWindows(List<AIRotationData> samples) {
        List<float[]> windows = new ArrayList<float[]>();
        if (samples == null) {
            return windows;
        }
        int total = samples.size();
        for (
            int start = 0;
            start + RotationModelManager.WINDOW_SIZE <= total;
            start += RotationModelManager.WINDOW_STRIDE
        ) {
            float[] window = new float[RotationModelManager.WINDOW_SIZE * RotationModelManager.INPUT_DIM];
            for (int t = 0; t < RotationModelManager.WINDOW_SIZE; t++) {
                writeFeatures(window, t * RotationModelManager.INPUT_DIM, samples.get(start + t));
            }
            windows.add(window);
        }
        return windows;
    }

    public static void writeFeatures(float[] window, int offset, AIRotationData data) {
        window[offset] = data.getDeltaYaw();
        window[offset + 1] = data.getDeltaPitch();
        window[offset + 2] = data.getAccelYaw();
        window[offset + 3] = data.getAccelPitch();
        window[offset + 4] = data.getJerkYaw();
        window[offset + 5] = data.getJerkPitch();
        window[offset + 6] = data.getGcdErrorYaw();
        window[offset + 7] = data.getGcdErrorPitch();
        window[offset + 8] = data.getDeltaTime();
        window[offset + 9] = data.getAttackFlag();
    }
}
