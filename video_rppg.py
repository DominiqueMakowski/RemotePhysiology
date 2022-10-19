import matplotlib.pyplot as plt
from pyVHR.analysis.pipeline import Pipeline

pipe = Pipeline()
time, BPM, uncertainty = pipe.run_on_video(
    "data/video.mp4", roi_approach="patches", roi_method="faceparsing"
)

plt.figure()
plt.plot(time, BPM)
plt.fill_between(time, BPM - uncertainty, BPM + uncertainty, alpha=0.2)
plt.show()
