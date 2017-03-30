from __future__ import division

import numpy as np
import cv2
import imutils
from collections import deque


class MotionDetector:
    def __init__(self, max_hist_len=5):
        self.pos = None
        self.curr_frame = None
        # self.first_frame = None
        self.max_hist_len = max_hist_len
        self.frame_hist = deque()
        self.frame_size = None

    def process_frame(self, frame):
        # resize the frame, convert it to grayscale, and blur it
        # frame = imutils.resize(frame, width=500)
        self.curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.curr_frame = cv2.GaussianBlur(self.curr_frame, (21, 21), 0)

        # # compute the weighted difference between the current frame and the frame history
        # frameDelta = cv2.absdiff(self.prev_frame, self.curr_frame)
        # thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        #
        # # dilate the thresholded image to fill in holes, then find contours on thresholded image
        # thresh = cv2.dilate(thresh, None, iterations=2)
        # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.frame_hist:
            hist_delta, deltas = self.calc_diff()
            self.show_hist(self.frame_hist, deltas, show_now=True)

        self.frame_hist.append(self.curr_frame)
        if len(self.frame_hist) > self.max_hist_len:
            self.frame_hist.popleft()

    def calc_diff(self):
        hist_len = len(self.frame_hist)
        deltas = []
        hist_delta = np.zeros(self.curr_frame.shape)#, dtype=np.float)
        for i, f in enumerate(self.frame_hist):
            frame_delta = cv2.absdiff(f, self.curr_frame)# * i / hist_len
            thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, np.ones((5, 5)), iterations=2)
            hist_delta += thresh
            deltas.append(thresh)
        return hist_delta, deltas

    def show_hist(self, hist, hist2=None, show_now=True):
        img_vis = np.hstack([imutils.resize(x, width=400) for x in hist])
        if hist2 is not None:
            img_vis2 = np.hstack([imutils.resize(x, width=400) for x in hist2])
            img_vis = np.vstack((img_vis, img_vis2))
        cv2.imshow('frame hist', img_vis)
        if show_now:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    video_capture = cv2.VideoCapture(data_path)

    md = MotionDetector()
    for i in range(8):
        ret, frame = video_capture.read()
        md.process_frame(frame)