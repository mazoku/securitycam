from __future__ import division

import numpy as np
import cv2
import imutils
from collections import deque


class MotionDetector:
    def __init__(self):
        self.pos = None
        self.curr_frame = None
        # self.first_frame = None
        self.max_hist_len = 10
        self.frame_hist = deque()
        self.frame_size = None

    def process_frame(self, img):
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(img, width=500)
        self.curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.curr_frame = cv2.GaussianBlur(self.curr_frame, (21, 21), 0)

        # compute the weighted difference between the current frame and the frame history
        frameDelta = cv2.absdiff(self.prev_frame, self.curr_frame)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.frame_hist.append(self.curr_frame)
        if len(self.frame_hist) > self.max_hist_len:
            self.frame_hist.popleft()

    def calc_diff(self):
        hist_len = len(self.frame_hist)
        deltas = []
        hist_delta = np.zeros(self.curr_frame.shape, dtype=np.float)
        for i, f in enumerate(self.frame_hist):
            frame_delta = cv2.absdiff(f, self.curr_frame) * i / hist_len
            hist_delta += frame_delta
            deltas.append(frame_delta)
        return hist_delta, deltas

    def show_hist(self, show_now=True):
        img_vis = np.hstack(self.frame_hist)
        cv2.imshow('frame hist', img_vis)
        if show_now:
            cv2.waitKey(0)
            cv2.destroyAllWindows()