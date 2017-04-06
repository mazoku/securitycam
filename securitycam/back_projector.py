from __future__ import division

import numpy as np
import cv2
import imutils
from collections import deque
from itertools import chain
import sys
import os

from select_roi import SelectROI


class BackProjector:
    def __init__(self, space='hsv', hist_sizes=None, hist_ranges=None, channels=-1):
        self.rgb_frame = None
        self.heat_map = None  # heatmap of back projection
        self.model_im = None  # image of model
        self.model_hist = None  # hist of model
        self.space = space
        self.backprojection = None  # backprojection

        if space == 'hsv':
            self.space_code = cv2.COLOR_BGR2HSV
            if hist_sizes is None:
                self.hist_sizes = [180, 16, 16]
            if hist_ranges is None:
                self.hist_ranges = [180, 256, 256]
        elif space == 'lab':
            self.space_code = cv2.COLOR_BGR2Lab
            if hist_sizes is None:
                self.hist_sizes = [16, 16, 16]
            if hist_ranges is None:
                self.hist_ranges = [256, 256, 256]
        elif space == 'rgb':
            self.space_code = cv2.COLOR_BGR2RGB
            if hist_sizes is None:
                self.hist_sizes = [8, 8, 8]
            if hist_ranges is None:
                self.hist_ranges = [256, 256, 256]

        self.channels = channels
        if self.channels == -1:
            self.channels = range(3)
        elif not isinstance(self.channels, list):
            self.channels = [self.channels]

        self.sizes = [self.hist_sizes[i] for i in self.channels]
        self.ranges = list(chain.from_iterable([(0, self.hist_ranges[i]) for i in self.channels]))

    @property
    def model_im(self):
        return self.__model_im

    @model_im.setter
    def model_im(self, x):
        self.__model_im = x
        self.calc_model_hist(x)

    def calc_model_hist(self, im=None):
        if im is None:
            im = self.model_im
        model_cs = cv2.cvtColor(im, self.space_code)
        model_hist = cv2.calcHist([model_cs], self.channels, None, self.sizes, self.ranges)
        cv2.normalize(model_hist, model_hist, 0, 255, cv2.NORM_MINMAX)
        self.model_hist = model_hist

    def calc_model_from_protos(self, protos_path):
        root, __, files = os.walk(protos_path).next()
        img_paths = [os.path.join(root, x) for x in files]

        model_hist = np.zeros([self.hist_sizes[i] for i in self.channels])
        for i, img_path in enumerate(img_paths):
            im = cv2.imread(img_path)
            im = cv2.resize(im, (200, 300))
            im = cv2.GaussianBlur(im, (3, 3), 0)
            im_cs = cv2.cvtColor(im, self.space_code)

            h = cv2.calcHist([im_cs], self.channels, None, self.sizes, self.ranges)
            model_hist += h
        model_hist /= len(img_paths)
        self.model_hist = model_hist

    def postprocess(self, convolution=True, morphology=True):
        postp = self.backprojection.copy()
        # postprocessing - convolution
        if convolution:
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            postp = cv2.filter2D(postp, -1, disc)

        # postprocessing - morphology
        if morphology:
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            _, postp = cv2.threshold(postp, 180, 255, cv2.THRESH_BINARY)
            postp = cv2.erode(postp, disc, iterations=3)
            postp = cv2.dilate(postp, disc, iterations=3)

            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            postp = cv2.dilate(postp, disc, iterations=3)
            postp = cv2.erode(postp, disc, iterations=3)
            # postp = cv2.morphologyEx(postp, cv2.MORPH_CLOSE, disc)

        return postp

    def calc_heatmap(self, frame=None, convolution=True, morphology=True):
        if frame is not None:
            self.calc_backprojection(frame)
        else:
            self.calc_backprojection(self.rgb_frame)

        self.heat_map = self.postprocess(convolution=convolution, morphology=morphology)

    def calc_backprojection(self, frame):
        # save and convert frame
        self.rgb_frame = frame
        cs_im = cv2.cvtColor(self.rgb_frame, self.space_code)

        # bluring image
        cs_im = cv2.GaussianBlur(cs_im, (3, 3), 0)

        # calculating backprojection
        self.backprojection = cv2.calcBackProject([cs_im], self.channels, self.model_hist, self.ranges, 1)

    def track(self, track_window, frame=None, calc_heatmap=False):
        if calc_heatmap:
            if frame is not None:
                self.calc_heatmap(frame)
            else:
                self.calc_heatmap(self.rgb_frame)
        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        ret, track_window = cv2.CamShift(self.heat_map, track_window, term_crit)
        return ret, track_window


if __name__ == '__main__':
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    video_capture = cv2.VideoCapture(data_path)

    # selecting model
    for i in range(150):
        ret, frame = video_capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    roi_selector = SelectROI()
    roi_selector.pt1 = (222, 283)
    roi_selector.pt2 = (249, 330)
    roi_selector.pt1 = (351, 31)
    roi_selector.pt2 = (410, 130)
    roi_rect = (roi_selector.pt1[0], roi_selector.pt1[1],
                roi_selector.pt2[0] - roi_selector.pt1[0],
                roi_selector.pt2[1] - roi_selector.pt1[1])
    # roi_selector.select(frame)
    # roi_rect = roi_selector.roi_rect
    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

    im_model_vis = frame.copy()
    cv2.rectangle(im_model_vis, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
    cv2.imshow('model', im_model_vis)

    # backprojector -------------------------------------
    bp = BackProjector(space='hsv', channels=[0, 1])
    bp.model_im = img_roi
    bp.calc_model_hist(bp.model_im)
    track_window = roi_rect
    while True:
        ret, frame = video_capture.read()
        if not ret:
            sys.exit(0)
        # frame = imutils.resize(frame, width=800)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        bp.calc_heatmap(frame, convolution=True, morphology=False)

        # # CamShift tracker
        # ret, track_window = bp.track(track_window)
        # x, y, w, h = track_window
        # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if bp.heat_map is None:
            im_vis = np.hstack((frame, np.zeros_like(frame)))
        else:
            im_vis = np.hstack((frame, cv2.cvtColor(bp.heat_map, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('backprojection heatmap', im_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # -----------
    cv2.destroyAllWindows()
    video_capture.release()