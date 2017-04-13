from __future__ import division

import cv2
import numpy as np
from back_projector import BackProjector
from select_roi import SelectROI
import sys


class Tracker(object):
    def __init__(self, window=None):
        self.center = None
        self._track_window = window
        self.frame = None
        self.track_space = None
        self.ret = False
        self.prev_track_window = None
        self.found = False
        self.score = 0

    @property
    def track_window(self):
        return self._track_window

    @track_window.setter
    def track_window(self, window):
        # self.prev_track_window = self._track_window[:]
        self._track_window = window
        if window is None:
            self.center = None
        else:
            self.center = (window[0] + window[2] / 2, window[1] + window[1] / 2)

    # def get_track_window(self):
    #     print 'track window getter'
    #     return self._track_window
    #
    # def set_track_window(self, window):
    #     print 'track window setter'
    #     self._track_window = window
    #     if window is None:
    #         self.center = None
    #     else:
    #         self.center = (window[0] + window[2] / 2, window[1] + window[1] / 2)
    #
    # track_window = property(get_track_window, set_track_window)

    def trackbox_score(self, track_space):
        mask_box = np.zeros(track_space.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask_box, self.track_box, 1, thickness=-1)
        mask_space = track_space > 0

        values_space = track_space[np.nonzero(mask_box & mask_space)]
        if values_space.any():
            self.score = values_space.mean()
        else:
            self.score = 0

        # values_box = track_space[np.nonzero(mask_box)]
        # score_box = values_box.mean()

        # print 'box: {:.2f}, space:{:.2f}'.format(prob_box, prob_space)

        # cv2.imshow('masks', np.hstack((255 * mask_box, 255 * (mask_space & mask_box))))
        # cv2.waitKey(0)

    def track(self, frame, track_space=None, track_window=None):
        self.frame = frame

        if track_space is None:
            self.track_space = frame.copy()
        else:
            self.track_space = track_space

        if track_window is not None:
            self.track_window = track_window
            # self.center = (self.track_window[0] + self.track_window[2] / 2, self.track_window[1] + self.track_window[1] / 2)

        if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
            self.prev_track_window = self.track_window[:]

            # Setup the termination criteria, either 10 iteration or move by at least 1 pt
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            self.track_box, self.track_window = cv2.CamShift(self.track_space, self.track_window, term_crit)
            # track_box = elipse: ((center_x, center_y), (width, height), angle)

            # print 'box: {}, window:{}'.format(self.track_box, self.track_window)
            if self.track_window[2] == 0 or self.track_window[3] == 0:
                self.found = False
            else:
                self.found = True
                self.trackbox_score(track_space)
        else:
            self.found = False

            # if track_window[2] == 0 or track_window[3] == 0:
            #     # self.ret = True
            #     print 'in'
            #     track_window = (0, 0, 50, 50)
            #     self.found = False
            # if track_box:
            #     self.track_window = track_window
            #     # self.center = (self.track_window[0] + self.track_window[2] / 2, self.track_window[1] + self.track_window[1] / 2)
            # else:
            #     self.track_window = None


if __name__ == '__main__':
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0220.mp4'
    video_capture = cv2.VideoCapture(data_path)
    # output_fname = '/home/tomas/temp/cv_seminar/backproj_tracker_in.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # selecting model
    for i in range(150):
        ret, frame = video_capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # video writer initialization
    # video_writer = cv2.VideoWriter(output_fname, fourcc, 30.0, (2 * frame.shape[1], frame.shape[0]), True)

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
    roi_selector.select(frame)
    roi_rect = roi_selector.roi_rect

    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

    bp = BackProjector(space='hsv', channels=[0, 1])
    bp.model_im = img_roi
    bp.calc_model_hist(bp.model_im)

    tracker = Tracker()
    tracker.track_window = roi_rect

    # bp.calc_heatmap(frame, convolution=True, morphology=False)
    # cv2.rectangle(frame, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 2)
    # im_vis = np.hstack((frame, cv2.cvtColor(bp.heat_map, cv2.COLOR_GRAY2BGR)))
    # for i in range(10):
    #     video_writer.write(im_vis)
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

        # video_writer.write(im_vis)
        cv2.imshow('CamShift tracker', im_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break