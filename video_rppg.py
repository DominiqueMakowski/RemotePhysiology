import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for file in glob.glob("data/*.csv"):
    print(file)


# ===================
# Ground truth
data = pd.read_csv(file)

# for i in range(3):
#     start = i * 60000
#     nk.signal_plot(
#         [
#             data["pyVHR"][start : start + 60000],
#             data["ECG_Rate"][start : start + 60000],
#             data["PPG_Rate"][start : start + 60000],
#         ],
#         sampling_rate=1000,
#     )


video, sampling_rate = nk.read_video(file.replace(".csv", ".mp4"))

# Crop
vid = video[0 : sampling_rate * 3, :, :, :]
dat = data[0 : 1000 * 3]

# Save example
np.savez("example", vid)
dat.to_csv("example.csv", index=False)

# Faces
faces = nk.video_face(vid)
ppg = nk.video_ppg(vid)


nk.video_plot(
    [vid, faces],
    frames=5,
    signals=[
        data["PPG_Clean"],
        nk.signal_filter(
            nk.signal_resample(ppg, desired_length=len(dat)),
            sampling_rate=1000,
            lowcut=1,
            highcut=1.5,
        ),
    ],
)

# nk.video_plot([video, faces], frames=5, signals=[data["pyVHR"], data["ECG_Rate"]])
