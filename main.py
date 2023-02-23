import os
import glob
import numpy as np
import cv2 as cv


def projective_transformation(src):
    rows, cols = src.shape[:2]

    # gray processing (gray image)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Image binarization
    ret, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    # Detect contour
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Detect outermost contour (there exists contour created by the hole in the image)
    contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)[0]

    # Create rectengle of outermost contour
    rect = cv.minAreaRect(contours_sorted)

    # Create four vertices of the rectengle
    pts = cv.boxPoints(rect)

    # Create four vertices after transformation
    pts_after = np.float32([[0, 0], [rows, 0], [rows, cols], [0, cols]])

    # transformation matrix
    M = cv.getPerspectiveTransform(pts, pts_after)

    # prejective transformation
    dst = cv.warpPerspective(src, M, (rows, cols))

    return dst

def spatial_filter_smoothing(src, kernel):
    # Opening / Closing
    dst = cv.morphologyEx(src, cv.MORPH_CLOSE, kernel)

    # # Median filter
    # dst = cv.medianBlur(src, 7)

    # # Gaussian filter
    # dst = cv.GaussianBlur(dst, (5, 5), 2, 2, borderType=cv.BORDER_DEFAULT)

    # Bilateral filter
    dst = cv.bilateralFilter(dst, 11, 75, 75, borderType=cv.BORDER_REPLICATE)

    return dst


# Sobel filter: not apply due to changes of original colors

def spatial_filter_sharpening(src):
    # Laplacian filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv.filter2D(src, -1, kernel=kernel)
    return dst


def homomorphic_filter(src, D0):
    rows, cols = src.shape[:2]

    # Create filter matrix
    H = np.zeros((rows, cols, 2))

    # Fast Fourier transformation
    F = np.fft.fft2(np.float64(src))

    # Centrelization
    F_shift = np.fft.fftshift(F)

    # Seperate function of illumination and reflection
    I, R = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))

    # Euclidean Metrix
    D = np.sqrt(np.power(I, 2) + np.power(R, 2))

    # Gaussian filter
    H = 1 - np.exp((-4 * np.power(D, 2)) / (2 * np.power(D0, 2)))

    # High-pass emphasis function
    H = 0.5 + 3 * H
    G_shift = F_shift * H

    # Counter centrelization
    G = np.fft.ifftshift(G_shift)

    # Counter fast Fourier transformation
    dst = np.fft.ifft2(G)

    # Regularization
    dst = np.real(dst)
    min, max = np.amin(dst, (0, 1)), np.amax(dst, (0, 1))
    dst = cv.normalize(dst, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    return dst

def inpainting(src):
    # gray processing (gray image)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # # Detect hole (circle)
    circle = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=15, maxRadius=30)
    circle = np.uint8(circle)

    # Create mask of hole (hole = white, other = black)
    mask = np.zeros((256, 256), np.uint8)
    mask.fill(0)
    cv.circle(mask, (circle[0, 0][0], circle[0, 0][1]), circle[0, 0][2]+5, (255, 255, 255), -1)

    # Inpainting
    dst = cv.inpaint(src, mask, inpaintRadius=100, flags=cv.INPAINT_TELEA)

    return dst



if not os.path.exists("./Results"):
    os.makedirs("./Results")

img_path = []
img_path.extend(glob.glob(os.path.join("./test_images/", "*.jpg")))

for path in img_path:
    img = cv.imread(path)

    img = projective_transformation(img)

    (b, g, r) = cv.split(img)

    kernel = np.ones((3, 3), np.uint8)
    b = spatial_filter_smoothing(b, kernel=kernel)
    g = spatial_filter_smoothing(g, kernel=kernel)
    kernel = np.ones((5, 5), np.uint8)
    r = spatial_filter_smoothing(r, kernel=kernel)

    img = cv.merge((b, g, r))

    img = spatial_filter_sharpening(img)

    (b, g, r) = cv.split(img)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    # r = clahe.apply(r)

    # b = homomorphic_filter(b, D0=1)
    g = homomorphic_filter(g, D0=1)
    r = homomorphic_filter(r, D0=1)

    img = cv.merge((b, g, r))
    print(path)
    img = inpainting(img)

    cv.imwrite(path.replace("test_images", "Results", 1), img)
