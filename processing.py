import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import scipy.signal
import sklearn.decomposition


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


def webcam_heartrate(frames, sampling_rate=30):
    raw = np.mean(np.mean(np.mean(frames, axis=1), axis=1), axis=1)

    # Clean
    raw = nk.signal_filter(
        raw,
        sampling_rate=sampling_rate,
        lowcut=1,
        highcut=1.8,
    )

    iff = scipy.fftpack.ifft(scipy.fftpack.fft(frames, axis=0), axis=0)
    iff = np.mean(np.mean(np.mean(iff, axis=1), axis=1), axis=1)
    return raw, iff


def temporal_ideal_filter(tensor, sampling_rate=30, low=0.4, high=2, axis=0):
    fft = scipy.fftpack.fft(tensor, axis=axis)
    frequencies = scipy.fftpack.fftfreq(tensor.shape[0], d=1.0 / sampling_rate)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = scipy.fftpack.ifft(fft, axis=axis)
    return np.abs(iff)


def webcam_heartrate(frames, sampling_rate=30):
    """full signal processing pipeline"""
    # Average all pixels in the different colors
    mixed_signals = np.mean(frames, axis=(1, 2))  # (len, color)
    raw = np.mean(mixed_signals, axis=1)

    # Filter signals
    def preprocess(signal, sampling_rate=sampling_rate):
        return nk.signal_filter(
            signal,
            sampling_rate=sampling_rate,
            lowcut=1,
            highcut=2,
        )

    normalized = np.apply_along_axis(preprocess, 0, mixed_signals)  # detrend and normalize

    nk.signal_plot(
        [normalized[:, 0], normalized[:, 1], normalized[:, 2]], sampling_rate=sampling_rate
    )

    ica = sklearn.decomposition.FastICA(n_components=1, max_iter=1000)
    ica_transformed = ica.fit_transform(normalized)
    nk.signal_plot(ica_transformed[:, 0], sampling_rate=sampling_rate)
    nk.signal_plot(
        [ica_transformed[:, 0], ica_transformed[:, 1], ica_transformed[:, 2]],
        sampling_rate=sampling_rate,
    )

    return ica_transformed


frames, sampling_rate = extract_face("video.mp4")
plt.imshow(frames[100, :, :, 0])
filtered_tensor = temporal_ideal_filter(frames, sampling_rate=sampling_rate)
plt.imshow(filtered_tensor[100, :, :, 0])

raw, iff = webcam_heartrate(frames, sampling_rate)


nk.signal_plot([raw, iff.real], sampling_rate=sampling_rate, title="Heart Rate", standardize=True)
