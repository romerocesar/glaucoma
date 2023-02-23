import os
import glob
import numpy as np
import cv2 as cv

# Ideal high/low-pass filter: not apply due to ringing artifacts

def frequency_filter_smoothing(src, machine, D0):
    rows, cols = src.shape[:2]

    # Create filter matrix
    H = np.zeros((rows, cols, 2))

    # Discrete Fourier transformation
    F = cv.dft(np.float32(src), flags=cv.DFT_COMPLEX_OUTPUT)

    # Centrelization
    F_shift = np.fft.fftshift(F)

    for i in range(rows):
        for j in range(cols):
            # Euclidean Metrix (from current pixel to centre)
            D = np.sqrt(
                np.power((i - int(rows / 2)), 2) + 
                np.power((j - int(cols / 2)), 2))
            
            # Second Butterworth filter
            if machine == "butterworth":
                H[i, j] = 1 / (1 + np.power((D / D0), 4))
            
            # Gaussian filter
            elif machine == "gaussian":
                H[i, j] = np.exp((-4 * np.power(D, 2)) / (2 * np.power(D0, 2)))
            
            else:
                pass
    
    G_shift = F_shift * H

    # Counter centrelization
    G = np.fft.ifftshift(G_shift)

    # Counter discrete Fourier transformation
    dst = cv.idft(G)

    # Regularization
    dst = cv.magnitude(dst[:, :, 0], dst[:, :, 1])
    min, max = np.amin(dst, (0, 1)), np.amax(dst, (0, 1))
    dst = cv.normalize(dst, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    return dst


def frequency_filter_sharpening(src, machine, D0):
    rows, cols = src.shape[:2]

    # Create filter matrix
    H = np.zeros((rows, cols, 2))

    # Discrete Fourier transformation
    F = cv.dft(np.float32(src), flags=cv.DFT_COMPLEX_OUTPUT)

    # Centrelization
    F_shift = np.fft.fftshift(F)

    for i in range(rows):
        for j in range(cols):
            # Euclidean Metrix (from current pixel to centre)
            D = np.sqrt(
                np.power((i - int(rows / 2)), 2) + 
                np.power((j - int(cols / 2)), 2))
            
            # Second Butterworth filter
            if machine == "butterworth":
                H[i, j] = 1 / (1 + np.power((D0 / D), 4))
            
            # Gaussian filter
            elif machine == "gaussian":
                H[i, j] = 1 - np.exp((-4 * np.power(D, 2)) / (2 * np.power(D0, 2)))
            
            else:
                pass
    
    # High-pass emphasis function
    H = 0.5 + 3 * H
    G_shift = F_shift * H

    # Counter centrelization
    G = np.fft.ifftshift(G_shift)

    # Counter discrete Fourier transformation
    dst = cv.idft(G)

    # Regularization
    dst = cv.magnitude(dst[:, :, 0], dst[:, :, 1])
    min, max = np.amin(dst, (0, 1)), np.amax(dst, (0, 1))
    dst = cv.normalize(dst, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    return dst
