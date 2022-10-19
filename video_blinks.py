import cv2
import matplotlib.pyplot as plt
import menpo.io
import menpo.landmark
import menpodetect
import neurokit2 as nk
import numpy as np
import scipy.fftpack
import scipy.signal
import sklearn.decomposition
from menpodetect import load_dlib_frontal_face_detector

frames, sampling_rate = nk.read_video("data/video.mp4")


plt.imshow(frames[0, 2, :, :])  # (frame, RGB-channel, height, width)

img = menpo.image.Image(frames[0, :, :, :], copy=True)

img_bw = img.as_greyscale()

# Face detection
faces = menpodetect.load_opencv_frontal_face_detector()(img_bw)
print("{} detected faces.".format(len(faces)))

# Eyes detection
eyes = menpodetect.load_opencv_eye_detector()(img_bw)
print("{} detected eyes.".format(len(eyes)))

img_bw.view()
faces[0].view(line_width=1, render_markers=False, line_colour="g")
eyes[0].view(line_width=1, render_markers=False, line_colour="r")
eyes[1].view(line_width=1, render_markers=False, line_colour="r")


# initial bbox
initial_bbox = bboxes[0]

# fit image
result = fitter.fit_from_bb(
    image, initial_bbox, max_iters=[15, 5], gt_shape=image.landmarks["PTS"].lms
)

# print result
print(result)

# View
img.view_landmarks(group="face_0", with_labels=True)


import menpowidgets

menpowidgets.visualize_images(img)

# https://github.com/opencv/opencv/blob/master/data/haarcascades/


def extract_eyes(file, classifier="utils/haarcascade_eye.xml", blur=3):
    """
    extract_faces _summary_

    Parameters
    ----------
    classifier : str
        The path of a cascade classifier. https://github.com/opencv/opencv/blob/master/data/haarcascades/
    """

    # TODO: is there a way to read that from online?
    # E.g., from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/
    Model = cv2.CascadeClassifier(classifier)

    capture = cv2.VideoCapture(file)
    sampling_rate = int(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    face_detected = False

    while capture.isOpened():
        success, img = capture.read()
        if not success:
            break

        if face_detected is False:
            face_detector = faceCascade.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1.3, 5
            )
            if len(face_detector) > 0:
                (x, y, w, h) = face_detector[0]
                face_detected = True

        if face_detected:
            img = cv2.resize(img[y : y + h, x : x + w], (500, 500)) * (1.0 / 255)
            frames.append(_build_laplacian_pyramid(img, levels=blur))

    capture.release()

    return np.array(frames), sampling_rate


# def webcam_heartrate(frames, sampling_rate=30):
#     raw = np.mean(np.mean(np.mean(frames, axis=1), axis=1), axis=1)

#     # Clean
#     raw = nk.signal_filter(
#         raw,
#         sampling_rate=sampling_rate,
#         lowcut=1,
#         highcut=1.8,
#     )

#     iff = scipy.fftpack.ifft(scipy.fftpack.fft(frames, axis=0), axis=0)
#     iff = np.mean(np.mean(np.mean(iff, axis=1), axis=1), axis=1)
#     return raw, iff


# def temporal_ideal_filter(tensor, sampling_rate=30, low=0.4, high=2, axis=0):
#     fft = scipy.fftpack.fft(tensor, axis=axis)
#     frequencies = scipy.fftpack.fftfreq(tensor.shape[0], d=1.0 / sampling_rate)
#     bound_low = (np.abs(frequencies - low)).argmin()
#     bound_high = (np.abs(frequencies - high)).argmin()
#     fft[:bound_low] = 0
#     fft[bound_high:-bound_high] = 0
#     fft[-bound_low:] = 0
#     iff = scipy.fftpack.ifft(fft, axis=axis)
#     return np.abs(iff)


# def webcam_heartrate(frames, sampling_rate=30):
#     """full signal processing pipeline"""
#     # Average all pixels in the different colors
#     mixed_signals = np.mean(frames, axis=(1, 2))  # (len, color)
#     raw = np.mean(mixed_signals, axis=1)

#     # Filter signals
#     def preprocess(signal, sampling_rate=sampling_rate):
#         return nk.signal_filter(
#             signal,
#             sampling_rate=sampling_rate,
#             lowcut=1,
#             highcut=2,
#         )

#     normalized = np.apply_along_axis(preprocess, 0, mixed_signals)  # detrend and normalize

#     nk.signal_plot(
#         [normalized[:, 0], normalized[:, 1], normalized[:, 2]], sampling_rate=sampling_rate
#     )

#     ica = sklearn.decomposition.FastICA(n_components=1, max_iter=1000)
#     ica_transformed = ica.fit_transform(normalized)
#     nk.signal_plot(ica_transformed[:, 0], sampling_rate=sampling_rate)
#     nk.signal_plot(
#         [ica_transformed[:, 0], ica_transformed[:, 1], ica_transformed[:, 2]],
#         sampling_rate=sampling_rate,
#     )

#     return ica_transformed


frames, sampling_rate = extract_face(file="data/video.mp4", blur=0)
plt.imshow(frames[0, :, :, 0])

# Plane-Orthogonal-to-Skin (POS)
# ==============================
# 4. Spatial averaging
C = np.mean(frames, axis=(1, 2))  # (len, color)
S = np.zeros((len(frames), 2))

for n, Ci in enumerate(C):
    if n > sampling_rate:
        # 5. Temporal normalization
        Cn = Ci / np.mean(C[n - sampling_rate : n + 1], axis=0)
    else:
        Cn = Ci / np.mean(C[0 : n + 1], axis=0)

    # 6. Projection
    projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
    S[n, :] = np.matmul(projection_matrix, Cn)

h = S[:, 0] + (np.std(S[:, 0]) / np.std(S[:, 1])) * S[:, 1]

nk.signal_plot(
    [
        h,
        nk.signal_filter(h, sampling_rate=sampling_rate, lowcut=1, highcut=1.8),
        np.mean(C, axis=1),
        nk.signal_filter(np.mean(C, axis=1), sampling_rate=sampling_rate, lowcut=1, highcut=1.8),
    ],
    sampling_rate=sampling_rate,
    standardize=True,
)

# filtered_tensor = temporal_ideal_filter(frames, sampling_rate=sampling_rate)
# plt.imshow(filtered_tensor[100, :, :, 0])

# raw, iff = webcam_heartrate(frames, sampling_rate)


# nk.signal_plot([raw, iff.real], sampling_rate=sampling_rate, title="Heart Rate", standardize=True)