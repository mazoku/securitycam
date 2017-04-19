# import the necessary packages
import numpy as np
import cv2
from itertools import chain


class HSVHistogram:
    def __init__(self, bins):
        # store the number of bins for the histogram
        self.bins = bins
        self.hist_sizes = [180, 16, 16]
        self.hist_ranges = [180, 256, 256]
        self.channels = [0, 1]

        self.sizes = None
        self.ranges = None
        self.update_hist_params()

    def update_hist_params(self):
        self.sizes = [self.hist_sizes[i] for i in self.channels]
        self.ranges = list(chain.from_iterable([(0, self.hist_ranges[i]) for i in self.channels]))

    def describe(self, image, use_segments=False):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        if use_segments:
            # grab the dimensions and compute the center of the image
            (h, w) = image.shape[:2]
            (cX, cY) = (int(w * 0.5), int(h * 0.5))

            # divide the image into four rectangles/segments (top-left,
            # top-right, bottom-right, bottom-left)
            segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h)]

            # construct an elliptical mask representing the center of the
            # image
            (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
            ellipMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

            # loop over the segments
            for (startX, endX, startY, endY) in segments:
                # construct a mask for each corner of the image, subtracting
                # the elliptical center from it
                cornerMask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
                cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

            # extract a color histogram from the elliptical region and
            # update the feature vector
            # hist = self.histogram(image, ellipMask)
            hist = self.gen_histogram(image, ellipMask)
            features.extend(hist)
        else:
            # hist = self.histogram(image)
            hist = self.gen_histogram(image)
            features.extend(hist)

        # return the feature vector
        return np.array(features)

    def histogram(self, image, mask=None):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel; then
        # normalize the histogram
        try:
            hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        except:
            pass
        cv2.normalize(hist, hist).flatten()

        # return the histogram
        return hist

    def gen_histogram(self, image, mask=None):
        if mask is None:
            mask = 255 * np.ones(image.shape[:2], dtype=np.uint8)

        im_cs = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([im_cs], self.channels, mask, self.sizes, self.ranges)
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).flatten()

        return hist