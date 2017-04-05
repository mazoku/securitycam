from __future__ import division

import numpy as np
import cv2
import imutils
from collections import deque
import sys


class MotionDetector:
    def __init__(self, max_hist_len=5):
        self.pos = None  # position of significant motion
        self.rgb_frame = None
        self.curr_frame = None  # last grabbed frame
        self.curr_res = None  # result of BGD subtractor
        self.heat_map = None  # motion heatmap
        # self.first_frame = None
        self.max_hist_len = max_hist_len
        self.frame_hist = deque()
        self.frame_size = None
        self.weights = self.calc_weights()

    def calc_weights(self):
        c = [1 / x for x in range(self.max_hist_len, 0, -1)]
        return c

    def process_frame(self, frame, show_hist=False, show_res=True, show_now=True):
        # resize the frame, convert it to grayscale, and blur it
        # frame = imutils.resize(frame, width=500)
        self.rgb_frame = frame
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
            self.curr_res, deltas = self.calc_diff()
            if show_res:
                self.show_result(show_now=show_now and not show_hist)
            if show_hist:
                self.show_history(self.frame_hist, deltas, show_now=show_now)
            if show_now and (show_res or show_hist):
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    sys.exit(0)

        self.frame_hist.append(self.curr_frame)
        if len(self.frame_hist) > self.max_hist_len:
            self.frame_hist.popleft()

    def calc_heatmap(self, update=False, frame=None, show=False, show_now=True):
        if update and frame is not None:
            self.curr_frame = frame
            self.process_frame(frame)
        if self.curr_res is None:
            return None
        motion_img = self.curr_res.copy()

        # find the contours and take the 5 biggest
        motion_t = 0.5 * 255
        th = cv2.threshold((255 * motion_img).astype(np.uint8), motion_t, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea)[-6:]

        hm = self.curr_frame.copy()
        mask = np.zeros_like(hm)
        for c in cnts:
            cv2.drawContours(mask, [c], -1, 1, -1)
        hm *= mask

        if show:
            img_vis = self.rgb_frame.copy()
            for c in cnts:
                cv2.drawContours(img_vis, [c], -1, (0, 0, 255), 1)
            img_vis = np.hstack((img_vis, cv2.cvtColor((255 * self.curr_res).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                 cv2.cvtColor((255 * hm).astype(np.uint8), cv2.COLOR_GRAY2BGR)))
            cv2.imshow('contours', img_vis)
            if show_now:
                cv2.waitKey(0)
        self.heat_map = hm

    def calc_diff(self):
        # hist_len = len(self.frame_hist)
        deltas = []
        hist_delta = np.zeros(self.curr_frame.shape)#, dtype=np.float)
        for i, f in enumerate(self.frame_hist):
            frame_delta = cv2.absdiff(f, self.curr_frame)# * i / hist_len
            thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, np.ones((5, 5)), iterations=2)
            thresh = thresh.astype(np.float) * self.weights[i]
            hist_delta += thresh
            deltas.append(thresh)
        hist_delta /= hist_delta.max()
        return hist_delta, deltas

    def show_result(self, show_now=True):
        imgs = [self.curr_frame, (255 * self.curr_res).astype(np.uint8)]
        imgs = [imutils.resize(i, width=800) for i in imgs]
        img_vis = np.hstack(imgs)

        cv2.imshow('result', img_vis)
        # if show_now:
        #     cv2.waitKey(50)
            # cv2.destroyAllWindows()

    def show_history(self, hist_fr, hist_mc=None, show_now=True):
        n_imgs = len(hist_fr) + 1
        img_w = 800
        hist1 = list(hist_fr)
        hist1.append(self.curr_frame)
        img_vis = np.hstack([imutils.resize(x, width=img_w) for x in hist1])
        if hist_mc is not None:
            hist2 = list(hist_mc)
            hist2.append((255 * self.curr_res).astype(np.uint8                       ))
            img_vis2 = np.hstack([imutils.resize(x, width=img_w) for x in hist2])
            img_vis = np.vstack((img_vis, img_vis2))
        cv2.line(img_vis, ((n_imgs - 1) * img_w, 0), ((n_imgs - 1) * img_w, img_vis.shape[0]), (0, 0, 255), 2)
        cv2.imshow('frame hist', img_vis)
        # if show_now:
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


if __name__ == '__main__':
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    video_capture = cv2.VideoCapture(data_path)

    # my adaptive -------------------------------------
    md = MotionDetector()
    while True:
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=800)
        md.process_frame(frame, show_res=False)
        md.calc_heatmap()

        if md.heat_map is None:
            im_vis = np.hstack((frame, np.zeros_like(frame)))
        else:
            im_vis = np.hstack((frame, cv2.cvtColor((255 * md.heat_map).astype(np.uint8), cv2.COLOR_GRAY2BGR)))
        cv2.imshow('motion heatmap', im_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # MOG -------------------------------------
    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=500)
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    # fgbg = cv2.createBackgroundSubtractorKNN(history=20, detectShadows=False, dist2Threshold=1000)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
    # while (1):
    #     ret, frame = video_capture.read()
    #     frame = imutils.resize(frame, width=640)
    #     fgmask = fgbg.apply(frame)
    #     # fgmask = cv2.erode(fgmask, np.ones((3, 3)))
    #     # fgmask = cv2.dilate(fgmask, np.ones((5, 5)))
    #     cv2.imshow('frame', fgmask)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break

    # -----------
    cv2.destroyAllWindows()
    video_capture.release()