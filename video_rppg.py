import matplotlib.pyplot as plt
import numpy as np
from pyVHR.analysis.pipeline import Pipeline

# Benchmark
# ===================
pipe = Pipeline()
time, BPM, uncertainty = pipe.run_on_video(
    "data/video.mp4", roi_approach="patches", roi_method="faceparsing"
)

plt.figure()
plt.plot(time, BPM)
plt.fill_between(time, BPM - uncertainty, BPM + uncertainty, alpha=0.2)
plt.show()
# ===================

video, sampling_rate = nk.read_video("data/video.mp4")

signal = nk.ecg_simulate(duration=int(len(video) / sampling_rate), sampling_rate=200)
signal = nk.signal_resample(signal, desired_length=len(video), sampling_rate=200)
nk.video_plot(video, frames=3, signals=signal)
nk.video_plot(video, frames=3, signals=[signal, signal])
nk.video_plot([video, video], frames=3, signals=[signal, signal])


# Plane-Orthogonal-to-Skin (POS)
# ==============================
# 4. Spatial averaging
C = np.mean(frames, axis=(1, 2))  # (len, color)
S = np.zeros((len(frames), 2))

for n, Ci in enumerate(C):
    if n > sampling_rate:
        # 5. Temporal normalization
        Cn = Ci / np.mean(C[n - sampling_rate : n + 1], axis=0)
    else:
        Cn = Ci / np.mean(C[0 : n + 1], axis=0)

    # 6. Projection
    projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
    S[n, :] = np.matmul(projection_matrix, Cn)

h = S[:, 0] + (np.std(S[:, 0]) / np.std(S[:, 1])) * S[:, 1]

nk.signal_plot(
    [
        h,
        nk.signal_filter(h, sampling_rate=sampling_rate, lowcut=1, highcut=1.8),
        np.mean(C, axis=1),
        nk.signal_filter(np.mean(C, axis=1), sampling_rate=sampling_rate, lowcut=1, highcut=1.8),
    ],
    sampling_rate=sampling_rate,
    standardize=True,
)
