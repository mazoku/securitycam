from __future__ import division
import cv2
import numpy as np
import imutils
from imutils import paths
from collections import defaultdict
import pickle
import gzip
import os
from itertools import chain

from back_projector import BackProjector
from tracker import Tracker
from descriptor import Descriptor
from motion_detector import MotionDetector


class Object(object):
    def __init__(self, name, motion_detector=None, space='hsv', hist_sizes=None, hist_ranges=None, channels=[0, 1]):
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

        # flags
        self.show_heatmaps_F = False  # F means it's a flag (no more name collisions with methods)

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

    def calc_model(self, img=None, mask=None, calc_char=True, show=False, show_now=True):
        if img is None:
            img = self.protos[0]
        else:
            self.protos.append(img)
        # if mask is None and calc_char:
        #         score_im, mask = self.char_pixels(img, im)

        model_cs = cv2.cvtColor(img, self.space_code)
        model_hist = cv2.calcHist([model_cs], self.channels, mask, self.sizes, self.ranges)
        cv2.normalize(model_hist, model_hist, 0, 255, cv2.NORM_MINMAX)

        if show:
            if mask is None:
                mask = 255 * np.ones(img.shape[:2], dtype=np.uint8)
            cv2.imshow('model', np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
            if show_now:
                cv2.waitKey(0)

        self.model_hist = model_hist
        self.back_projector.model_hist = self.model_hist

    def load_protos(self, protos_path):
        """
        Load prototypes from disk.
        :param protos_path: Path to where the prototypes residues on the disk.
        :return:
        """
        # get all the files in given directory
        root, __, files = os.walk(protos_path).next()
        img_paths = [os.path.join(root, x) for x in files]

        # load the image and append it to the protos list
        protos = []
        for img_path in enumerate(img_paths):
            protos.append(cv2.imread(img_path))
        self.protos = protos

    def create_from_protos(self, protos_path=None):
        """
        Calculates histogram of the model from prototypes given by the path or already loaded.
        :param protos_path: Path to where the prototypes residues on the disk.
        :return:
        """
        # load prototypes from file if the path is given
        if protos_path is not None:
            self.load_protos(protos_path)

        # create the model structure
        model_hist = np.zeros([self.hist_sizes[i] for i in self.channels])

        # process each prototype and compute its histogram
        for im in self.protos:
            im = cv2.resize(im, (200, 300))
            im = cv2.GaussianBlur(im, (3, 3), 0)
            im_cs = cv2.cvtColor(im, self.space_code)
            h = cv2.calcHist([im_cs], self.channels, None, self.sizes, self.ranges)
            model_hist += h
        model_hist /= len(self.protos)

        # update the model attribute
        self.model_hist = model_hist

    def show_heatmaps(self, frame):
        win_height = 300
        im1 = imutils.resize(frame, height=win_height)
        im2 = cv2.cvtColor(imutils.resize(self.heatmap, height=win_height), cv2.COLOR_GRAY2BGR)
        im3 = cv2.cvtColor(imutils.resize(self.back_projector.heat_map, height=win_height), cv2.COLOR_GRAY2BGR)
        im4 = cv2.cvtColor(imutils.resize(self.motion_detector.heat_map, height=win_height), cv2.COLOR_GRAY2BGR)

        # creating mosaic
        row1 = np.hstack((im1, im2))
        row2 = np.hstack((im3, im4))
        im_vis = np.vstack((row1, row2))

        cv2.imshow('{}: frame | obj_heatmap | backproj_heatmap | motion_heatmap'.format(self.name), im_vis)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or key == 27:
            self.show_heatmaps_F = False
        else:
            self.show_heatmaps_F = True

    def update(self, frame):
        # back project and motion detection
        self.back_projector.calc_heatmap(frame, convolution=True, morphology=False)
        self.motion_detector.calc_heatmap(frame=frame, update=True, show=False)

        # calculating heatmap
        self.calc_heatmap()

        # visualize heatmaps
        if self.show_heatmaps_F:
            self.show_heatmaps(frame)

        # update tracker
        # self.tracker.track(frame, self.back_projector.heat_map, track_window=self.tracker.track_window)
        # self.tracker.track(frame, self.heatmap)

        # control the smoothness of object position
        # self.control_smoothness()