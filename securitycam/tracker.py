from __future__ import division

import cv2
import numpy as np
from back_projector import BackProjector
from select_roi import SelectROI
import sys

class Tracker:
    def __init__(self, track_window=None):
        self.center = None
        self.track_window = track_window
        self.frame = None
        self.track_space = None
        self.ret = False
        self.prev_track_window = None

    @property
    def track_window(self):
        return self.track_window

    @track_window.setter
    def track_window(self, window):
        self.track_window = window
        if window is None:
            self.center = None
        else:
            self.center = (window[0] + window[2] / 2, window[1] + window[1] / 2)

    def track(self, frame, track_space=None, track_window=None):
        self.frame = frame

        if track_space is None:
            self.track_space = frame.copy()
        else:
            self.track_space = track_space

        if track_window is not None:
            self.track_window = track_window

        self.prev_track_window = self.track_window[:]

        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.ret, track_window = cv2.CamShift(self.track_space, self.track_window, term_crit)
        if self.ret:
            self.track_window = track_window
        else:
            self.track_window = None


if __name__ == '__main__':
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    video_capture = cv2.VideoCapture(data_path)

    # selecting model
    for i in range(150):
        ret, frame = video_capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    roi_selector = SelectROI()
    # roi_selector.select(frame)
    # roi_rect = roi_selector.roi_rect
    # roi_selector.pt1 = (222, 283)
    # roi_selector.pt2 = (249, 330)
    roi_selector.pt1 = (351, 31)
    roi_selector.pt2 = (410, 130)
    roi_rect = (roi_selector.pt1[0], roi_selector.pt1[1],
                roi_selector.pt2[0] - roi_selector.pt1[0],
                roi_selector.pt2[1] - roi_selector.pt1[1])
    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

    bp = BackProjector(space='hsv', channels=[0, 1])
    bp.model_im = img_roi
    bp.calc_model_hist(bp.model_im)

    tracker = Tracker()
    tracker.track_window = roi_rect
    while True:
        ret, frame = video_capture.read()
        if not ret:
            sys.exit(0)
        # frame = imutils.resize(frame, width=800)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        bp.calc_heatmap(frame, convolution=True, morphology=False)
        tracker.track(frame, bp.heat_map)

        frame_vis = frame.copy()
        if tracker.ret:
            x, y, w, h = tracker.track_window
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im_vis = np.hstack((frame_vis, cv2.cvtColor(bp.heat_map, cv2.COLOR_GRAY2BGR)))

        cv2.imshow('CamShift tracker', im_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break