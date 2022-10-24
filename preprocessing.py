import base64
import glob

import neurokit2 as nk
import pandas as pd

# 1. Convert json video data to mp4
# ==================================
for file in glob.glob("data/*.json"):
    print(file)
    data = pd.read_json(file)

    video = base64.b64decode(data["video"][0])

    with open(file.replace(".json", ".mp4"), "wb") as f:
        f.write(video)
        f.close()

# 2. Preprocess physio
# ==================================
for file in glob.glob("data/*.txt"):
    print(file)
    data, info = nk.read_bitalino(file)
    sampling_rate = info["sampling_rate"]

    lux = data["LUX"]
    nk.signal_plot(lux, sampling_rate=sampling_rate)
    events = nk.events_find(lux, threshold="auto", threshold_keep="below", duration_min=10000)

    print(f"duration: {events['duration'][0] / sampling_rate / 60}")
    data = data.iloc[events["onset"][0] : events["onset"][0] + events["duration"][0]]
