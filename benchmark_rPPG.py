import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

for file in glob.glob("data/*A.csv"):
    print(file)
    # Ground truth
    data = pd.read_csv(file)

    # Filter
    data["pyVHR"] = nk.signal_filter(data["pyVHR"], sampling_rate=1000, lowcut=0.65, highcut=4)
    data["nkPPG"] = nk.signal_filter(data["nkVHR"], sampling_rate=1000, lowcut=0.65, highcut=4)
    data["nkVHR"] = nk.signal_rate(
        nk.ppg_findpeaks(data["nkPPG"])["PPG_Peaks"], sampling_rate=1000, desired_length=len(data)
    )

    # ===================
    # Visualize
    # ===================
    name = file.replace("data\\", "").replace(".csv", "")
    video, sampling_rate = nk.read_video(file.replace(".csv", ".mp4"))

    # Crop
    vid = video[0 : sampling_rate * 45, :, :, :]
    dat = data[0 : 1000 * 10]

    nk.video_plot(
        vid,
        frames=5,
        signals=[
            dat["PPG_Clean"],
            dat["nkPPG"],
            dat["PPG_Rate"],
            dat["nkVHR"],
            dat["pyVHR"],
        ],
    )
    fig = plt.gcf()
    ax = fig.axes
    ax[1].set_ylabel("PPG")
    ax[2].set_ylabel("rPPG")
    ax[3].set_ylabel("Heart Rate (True)")
    ax[4].set_ylabel("Heart Rate (nkVHR)")
    ax[5].set_ylabel("Heart Rate (pyVHR)")
    plt.legend()
    fig.set_size_inches(10, 10)
    plt.savefig(f"figures/{name}_video.png")

    # ===================
    # Correlation
    # ===================

    fig, ax = plt.subplot_mosaic(
        [["upper", "upper"], ["lower left", "lower right"]], figsize=(12, 12)
    )
    corr = data[["ECG_Rate", "PPG_Rate", "nkVHR", "pyVHR"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax["upper"])
    sns.regplot(
        x=data["PPG_Rate"],
        y=data["pyVHR"],
        color="red",
        scatter_kws={"alpha": 0.1, "color": "grey"},
        ax=ax["lower left"],
    )
    sns.regplot(
        x=data["PPG_Rate"],
        y=data["nkVHR"],
        color="red",
        scatter_kws={"alpha": 0.1, "color": "grey"},
        ax=ax["lower right"],
    )
    plt.tight_layout()
    plt.savefig(f"figures/{name}_correlation.png")


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


# # Save example
# np.savez("example", vid)
# dat.to_csv("example.csv", index=False)

# # Faces
# faces = nk.video_face(vid)
# ppg = nk.video_ppg(vid)


# nk.video_plot(
#     [vid, faces],
#     frames=5,
#     signals=[
#         data["PPG_Clean"],
#         nk.signal_filter(
#             nk.signal_resample(ppg, desired_length=len(dat)),
#             sampling_rate=1000,
#             lowcut=1,
#             highcut=1.5,
#         ),
#     ],
# )

# nk.video_plot([video, faces], frames=5, signals=[data["pyVHR"], data["ECG_Rate"]])
