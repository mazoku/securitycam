from __future__ import division
import cv2
import numpy as np
import imutils
from imutils import paths
from collections import defaultdict
import pickle
import gzip
import os

from back_projector import BackProjector
from tracker import Tracker
from descriptor import Descriptor
from motion_detector import MotionDetector


class Object(object):
    def __init__(self, name, motion_detector=None):
        self.name = name  # object label
        self.protos = []  # list of image prototypes
        self.masks = []
        self.features = []
        self.mean_features = None
        self.median_features = None
        self.heatmap = None  # backprojection and motion detection heatmap

        self.back_projector = BackProjector(space='hsv', channels=[0, 1])
        self.tracker = Tracker()
        self.descriptor = Descriptor('hsvhist')
        if motion_detector is None:
            self.motion_detector = MotionDetector()

    def calc_heatmap(self):
        hm1 = self.back_projector.heat_map
        if self.motion_detector.heat_map is None:
            self.heatmap = hm1.astype(np.uint8)
        else:
            hm2 = self.motion_detector.heat_map
            hm = ((hm1 + hm2) / 2).astype(np.uint8)
            self.heatmap = hm

    def analyse_heatmap(self):
        # threshold
        _, hm = cv2.threshold(self.heatmap, 100, 255, cv2.THRESH_BINARY)

        # find 4 biggest contours
        cnts = cv2.findContours(hm.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not cnts:
            return None, (0, 0), self.max_dist + 1
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        # calculate distances between contours and last tracking window
        dists, centers = self.calc_cnts_dists(cnts, hm, show=False, show_now=True)

        # find contour closest to the tracking window
        try:
            min_idx = np.argmin(dists)
        except:
            pass
        winner_c = cnts[min_idx]
        winner_cent = centers[min_idx]
        winner_d = dists[min_idx]

        return winner_c, winner_cent, winner_d

    def calc_cnts_dists(self, cnts, hm, show=False, show_now=True):
        dists = []
        centers = []
        if show:
            # im_vis = np.zeros((hm.shape[0], hm.shape[1], 3))
            im_vis = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
        for c in cnts:
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                tmp = np.squeeze(c)
                if tmp.ndim == 1:
                    cX = tmp[0]
                    cY = tmp[1]
                elif tmp.ndim > 1:
                    m = tmp.mean(axis=0).astype(np.int32)
                    cX = m[0]
                    cY = m[1]
                else:
                    cX = 0
                    cY = 0
            centers.append((cX, cY))
            dists.append(np.linalg.norm((cX - self.tracker.center[0], cY - self.tracker.center[1])))
            # if show:
            #     cv2.drawContours(im_vis, [c], -1, (0, 0, 255))
            #     cv2.circle(im_vis, (cX, cY), 5, (255, 0, 255), -1)
        if show:
            for cnt, cent, d in zip(cnts, centers, dists):
                cv2.drawContours(im_vis, [cnt], -1, (0, 0, 255))
                cv2.circle(im_vis, (cent[0], cent[1]), 5, (255, 0, 255), -1)
                cv2.putText(im_vis, '{:.1f}'.format(d), (cent[0], cent[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.imshow('cnts analysis', im_vis)
            if show_now:
                cv2.waitKey(0)

        return dists, centers

    def control_smoothness(self):
        # TODO: tohle by melo byt spise v trackeru
        # TODO: kontrola z heatmapy? Nemela by byt nejprve jen zmena pozice track_window?
        # analyze heatmap
        contour, center, dist = self.analyse_heatmap()

        # if distance between tracking window and detected contour is to big ...
        if dist > self.max_dist:
            # ... increase number of consequent non-smooth frames
            self.n_nonsmooth += 1
            print 'Difference #{}'.format(self.n_nonsmooth)
            # if number of consequent non-smooth frames is to big -> reinitialize
            if self.n_nonsmooth > self.max_n_nonsmooth:
                print 'Difference is for to long - reinitializing.'
                (x, y, w, h) = cv2.boundingRect(contour)
                self.tracker.track_window = (x, y, w, h)
                self.tracker.center = (x + w / 2, y + h / 2)
                print 'Reseting difference counter'
                self.n_nonsmooth = 0
        else:
            # if distance smaller than max_dist, reset the counter of non-smooth frames
            if self.n_nonsmooth > 0:
                print 'Reseting difference counter'
            self.n_nonsmooth = 0

    def track(self, frame):
        self.tracker.track(frame)

    def detect(self):
        pass

    def save(self):
        pass

    @staticmethod
    def load():
        pass

    def calc_model(self):
        pass

    def create_from_protos(self):
        pass

    def update(self, frame):
        # back project and motion detection
        self.back_projector.calc_heatmap(frame, convolution=True, morphology=False)
        self.motion_detector.calc_heatmap(frame=frame, update=True, show=False)

        # calculating heatmap
        self.calc_heatmap()

        # update tracker
        # self.tracker.track(frame, self.back_projector.heat_map, track_window=self.tracker.track_window)
        self.tracker.track(frame, self.heatmap)

        # control the smoothness of object position
        self.control_smoothness()