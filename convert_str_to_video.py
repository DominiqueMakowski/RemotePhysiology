import base64

import pandas as pd

# Convert json to mp4
# ===================
data = pd.read_json("data.json")

video = base64.b64decode(data["record_video_data"][1])

with open("video.mp4", "wb") as file:
    file.write(video)
    file.close()
