'''prepare glaucoma images to improve predictions

usage:
  glaucoma [-i input] [-o output]
  glaucoma -h

OPTIONS:
  -h --help             show this help message.
  -i --input=INPUT      input directory [default: test_images]
  -o --output=OUTPUT    output directory [default: Results]
'''
from pathlib import Path
import logging
import sys

import docopt
import cv2 as cv
import numpy as np


logger = logging.getLogger('glaucoma')
logging.basicConfig(level=logging.DEBUG)


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


def median_blur(img):
    return cv.medianBlur(img, 7)


def gaussian_filter(img):
    return cv.GaussianBlur(img, (5, 5), 2, 2, borderType=cv.BORDER_DEFAULT)


def morph_close(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)


def morph_open(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)


def bilateral_filter(img):
    return cv.bilateralFilter(img, 11, 75, 75, borderType=cv.BORDER_REPLICATE)


def erode(img):
    b, g, r = cv.split(img)
    kernel = np.ones((2, 2), np.uint8)
    # b = cv.erode(b, kernel, iterations=1)
    # g = cv.erode(g, kernel, iterations=1)
    r = cv.erode(r, kernel, iterations=1)
    return cv.merge((b, g, r))


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
    circle = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30,
                             minRadius=5, maxRadius=25)
    circle = np.uint8(circle)

    # Create mask of hole (hole = white, other = black)
    mask = np.zeros((256, 256), np.uint8)
    mask.fill(0)
    cv.circle(img=mask, center=(circle[0, 0][0], circle[0, 0][1]), radius=circle[0, 0][2]+5,
              color=(255, 255, 255),
              thickness=-1)  # filled

    # Inpainting
    dst = cv.inpaint(src, mask, inpaintRadius=100, flags=cv.INPAINT_TELEA)

    return dst


def main():

    args = docopt.docopt(__doc__)

    input_path = Path(args['--input'])
    output_path = Path(args['--output'])
    output_path.mkdir(exist_ok=True)
    logger.info(input_path.absolute())

    for path in input_path.glob("*.jpg"):
        logger.debug(f'processing {path=}')
        fname = path.as_posix()
        img = cv.imread(fname)

        # img = projective_transformation(img)

        img = erode(img)

        # (b, g, r) = cv.split(img)
        # kernel = np.ones((3, 3), np.uint8)
        # b = spatial_filter_smoothing(b, kernel=kernel)
        # g = spatial_filter_smoothing(g, kernel=kernel)
        # kernel = np.ones((5, 5), np.uint8)
        # r = spatial_filter_smoothing(r, kernel=kernel)
        # img = cv.merge((b, g, r))

        # img = spatial_filter_sharpening(img)

        # (b, g, r) = cv.split(img)
        # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # b = clahe.apply(b)
        # g = clahe.apply(g)
        # # r = clahe.apply(r)
        # # b = homomorphic_filter(b, D0=1)
        # g = homomorphic_filter(g, D0=1)
        # r = homomorphic_filter(r, D0=1)
        # img = cv.merge((b, g, r))

        # try:
        #     img = inpainting(img)
        # except TypeError:
        #     logger.error(f'failed to find circle in {fname=}')
        #     continue

        fname = output_path.joinpath(path.name).as_posix()
        cv.imwrite(fname, img)
        logger.info(f'wrote image {fname=}')


if __name__ == '__main__':
    main()
