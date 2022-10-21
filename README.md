# Open Remote Physiology Dataset (remophysio)

An exploration of tools to extract physiological features (heart rate, blinks, ...) from webcam recordings, that can be used in online experiments.

This is currently a work in progress. Successful and reliable features could be integrated in [NeuroKit](https://github.com/neuropsychology/NeuroKit).

## Methods

### Procedure

Researchers who agreed to share their data underwent an 8-min resting state paradigm (using the [**Replicable Resting-state Task**](https://github.com/RealityBending/RestingState) in jsPsych). Faces were recorded using the laptop's webcam, and physiological signals (ECG, PPG, RSP) were recorded using BITalino.

### Preprocessing




**Record a new webcam video.** This can be done by using this minimal [**jsPsych**](https://www.jspsych.org/7.3/extensions/record-video/) experiment: https://dominiquemakowski.github.io/HeartOnline/. It is also possible to run a more complete [**Resting State**](https://github.com/RealityBending/RestingState) paradigm (in which one can enable cam recording). The data will be stored in a **.json** file, and must be converted to an **mp4** using the [convert_str_to_video.py](convert_str_to_video.py) script. The video file must be saved locally.




