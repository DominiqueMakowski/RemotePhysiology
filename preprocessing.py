import base64
import glob

import neurokit2 as nk
import pandas as pd
import pyVHR.analysis.pipeline

# 1. Convert json video data to mp4
# ==================================
for file in glob.glob("data/*.json"):
    print(file)
    data = pd.read_json(file)

    video = base64.b64decode(data["video"][0])

    with open(file.replace(".json", ".mp4"), "wb") as f:
        f.write(video)
        f.close()

# 2. Preprocess physio (stored as .csv)
# =====================================
for file in glob.glob("data/*.txt"):
    print(file)
    data, info = nk.read_bitalino(file)
    sampling_rate = info["sampling_rate"]

    # Get onset and end
    lux = data["LUX"].values
    events = nk.events_find(lux, threshold="auto", threshold_keep="below", duration_min=10000)

    print(f"- duration: {events['duration'][0] / sampling_rate / 60}")
    if events["duration"][0] < 300000:
        print(f"Skipping for {file}")
        continue

    # Process
    ecg, _ = nk.ecg_process(data["ECGB"], sampling_rate=sampling_rate)
    ppg, _ = nk.ppg_process(data["PULS"], sampling_rate=sampling_rate)
    rsp, _ = nk.rsp_process(data["RESP"], sampling_rate=sampling_rate)

    dat = pd.concat(
        [
            ecg[["ECG_Clean", "ECG_Rate"]],
            ppg[["PPG_Clean", "PPG_Rate"]],
            rsp[["RSP_Clean", "RSP_Rate"]],
        ],
        axis=1,
    )

    # Truncate
    dat = dat.iloc[events["onset"][0] : events["onset"][0] + events["duration"][0]]

    # 3. Benchmark using pyVHR
    # ======================
    pipe = pyVHR.analysis.pipeline.Pipeline()
    time, BPM, uncertainty = pipe.run_on_video(
        file.replace(".txt", ".mp4"), roi_approach="patches", roi_method="faceparsing"
    )

    dat["pyVHR"] = nk.signal_resample(BPM, desired_length=len(dat))

    # Save
    dat.to_csv(file.replace(".txt", ".csv"), index=False)
