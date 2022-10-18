import base64
import sys
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack
import scipy.signal

# Convert json to mp4
# ===================
data = pd.read_json("data.json")

video = base64.b64decode(data["record_video_data"][1])

with open("video.mp4", "wb") as file:
    file.write(video)
    file.close()

# Convert mp4 to array
# ====================
def _build_laplacian_pyramid(img, levels=3):
    pyramid = img.copy()
    for level in range(levels):
        pyramid = cv2.pyrDown(pyramid)

    upsampled = cv2.pyrUp(pyramid)
    (height, width, depth) = upsampled.shape
    pyramid = cv2.resize(pyramid, (height, width))

    return cv2.subtract(pyramid, upsampled)


def extract_face(path, classifier="haarcascade_frontalface_alt0.xml", blur=3):
    """
    extract_faces _summary_

    Parameters
    ----------
    path : str
        The path of a cascade classifier.
    """

    # TODO: is there a way to read that from online?
    # E.g., from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/
    faceCascade = cv2.CascadeClassifier(classifier)

    capture = cv2.VideoCapture(path)
    sampling_rate = int(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    check = True

    while capture.isOpened():
        success, img = capture.read()
        if not success:
            break

        if check:
            face_detector = faceCascade.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1.3, 5
            )
            if len(face_detector) > 0:
                (x, y, w, h) = face_detector[0]
                check = False

        if not check:
            img = cv2.resize(img[y : y + h, x : x + w], (500, 500)) * (1.0 / 255)
            frames.append(_build_laplacian_pyramid(img, levels=blur))

    capture.release()

    return np.array(frames), sampling_rate


def fft_filter(video, fps, freq_min=1, freq_max=1.8):
    fft = scipy.fftpack.fft(video, axis=0)
    iff = scipy.fftpack.ifft(fft, axis=0)

    return iff, fft, frequencies


frames, fps = extract_faces("video.mp4")
plt.imshow(frames[100, :, :, 0])

iff, fft, frequencies = fft_filter(frames, fps)
heart_rate, signal = find_heart_rate(fft, frequencies)
signal = np.mean(np.mean(np.mean(iff, axis=1), axis=1), axis=1)
raw = nk.signal_filter(
    np.mean(np.mean(np.mean(frames, axis=1), axis=1), axis=1),
    sampling_rate=fps,
    lowcut=1,
    highcut=1.8,
)
nk.signal_plot(
    [raw, signal.real, signal.imag], sampling_rate=fps, title="Heart Rate", standardize=True
)

nk.ppg_findpeaks(signal, sampling_rate=fps)
nk.signal_rate([27, 61, 85, 108, 138], sampling_rate=fps)

plt.plot(t, signal.real, "b-", t, signal.imag, "r--")
