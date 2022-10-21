import base64
import glob

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
