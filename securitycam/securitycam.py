from __future__ import division

import cv2
import numpy as np
import matplotlib.pyplot as plt
from classifier import Classifier
from descriptor import Descriptor
from back_projector import BackProjector
from motion_detector import MotionDetector
from tracker import Tracker
from select_roi import SelectROI

import os


class SecurityCam:
    def __init__(self):
        self.stream = None

        # self.tracker = cv2.Tracker_create(tracker)
        self.descriptor = Descriptor('hsvhist')
        self.classifier = Classifier(desc=self.descriptor)
        self.classifier.train_from_protos(train_size=-1, return_test=False)
        self.backprojector = BackProjector(space='hsv', channels=[0, 1])
        self.motiondetector = MotionDetector()
        self.tracker = Tracker()
        self.heatmap = None  # final heatmap derived from motion detection and backprojection
        self.max_dist = 250
        self.n_nonsmooth = 0  # number of consecutive differences between tracker and detector
        self.max_n_nonsmooth = 5
        # self.track_window = None
        # bp.calc_model_hist(bp.model_im)
        # track_window = roi_rect

    def mark_track_window(self, frame):
        roi_selector = SelectROI()
        roi_selector.select(frame)
        roi_rect = roi_selector.roi_rect
        self.tracker.track_window = roi_rect

    def roi2image(self, im, roi):
        im_roi = im[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        return im_roi

    def process_frame(self, frame):
        track_im = self.roi2image(frame, self.tracker.track_window)
        if not track_im.any():
            track_im = self.roi2image(frame, (0, 0, 50, 50))
            is_artif = True
        else:
            is_artif = False

        self.backprojector.calc_heatmap(frame, convolution=True, morphology=False)
        self.tracker.track(frame, self.backprojector.heat_map, track_window=self.tracker.track_window)
        self.motiondetector.calc_heatmap(frame=frame, update=True, show=False)

        if self.tracker.track_window[2] == 50:
            pass

        # row2 = np.hstack((self.backprojector.heat_map, self.motiondetector.heat_map))
        # row2 = cv2.cvtColor(row2, cv2.COLOR_GRAY2BGR)
        # row1 =  np.hstack((frame, cv2.cvtColor((((self.backprojector.heat_map + self.motiondetector.heat_map) / 2).astype(np.uint8)), cv2.COLOR_GRAY2BGR)))
        # im_vis = np.vstack((row1, row2))
        # cv2.imshow('heatmaps', im_vis)
        # cv2.waitKey(1)

        heat_contour, heat_cent, heat_dist = self.analyse_heatmap(frame)
        self.control_smoothness(heat_cent, heat_contour, heat_dist)

        # im_vis = frame.copy()
        # x, y, w, h = self.tracker.prev_track_window
        # cv2.rectangle(im_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # x, y, w, h = self.tracker.track_window
        # cv2.rectangle(im_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('tracker history', im_vis)
        # cv2.waitKey(0)

        if not is_artif:
            #TODO: jak se lisi vysledky, kdyz use_mean=True
            feats = self.descriptor.describe(track_im, use_mean=False)[0].flatten()
            label = self.classifier.predict(feats.reshape(1, -1))
            probs = self.classifier.predict_proba(feats.reshape(1, -1))
            # print probs
            prob = probs.max()
            # classes = self.classifier.model.classes_
            # probs = self.classifier.predict_proba(feats.reshape(1, -1))[0, :]
            # res = [(c, p) for c, p in zip(classes, probs)]
            # print res
        else:
            label = None
            prob = 0
        return label, prob

    def control_smoothness(self, cent_detector, cont_detector, dist):
        if dist > self.max_dist:
            self.n_nonsmooth += 1
            print 'Difference #{}'.format(self.n_nonsmooth)
            if self.n_nonsmooth > self.max_n_nonsmooth:
                print 'Difference is for to long - reinitializing.'
                (x, y, w, h) = cv2.boundingRect(cont_detector)
                self.tracker.track_window = (x, y, w, h)
                self.tracker.center = (x + w / 2, y + h / 2)
                print 'Reseting difference counter'
                self.n_nonsmooth = 0
        else:
            if self.n_nonsmooth > 0:
                print 'Reseting difference counter'
            self.n_nonsmooth = 0

    def analyse_heatmap(self, frame):
        hm1 = self.backprojector.heat_map
        hm2 = self.motiondetector.heat_map
        hm = ((hm1 + hm2) / 2).astype(np.uint8)
        _, hm = cv2.threshold(hm, 100, 255, cv2.THRESH_BINARY)

        # plt.figure()
        # plt.subplot(131), plt.imshow(hm1, 'gray')
        # plt.subplot(132), plt.imshow(hm2, 'gray')
        # plt.subplot(133), plt.imshow(hm, 'gray')
        # plt.show()

        cnts = cv2.findContours(hm.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not cnts:
            return None, (0, 0), self.max_dist + 1
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        dists, centers = self.calc_cnts_dists(cnts, hm, show=False, show_now=True)

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


# -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0220.mp4'
    # data_path = '/home/tomas/Data/videa/ada1.mp4'
    # data_path = '/home/tomas/Data/videa/ada2.mp4'
    video_capture = cv2.VideoCapture(data_path)
    save_output = False

    # selecting model
    for i in range(150):
        ret, frame = video_capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    roi_selector = SelectROI()
    # roi_selector.select(frame)
    # roi_rect = roi_selector.roi_rect
    # roi_selector.pt1 = (222, 283)
    # roi_selector.pt2 = (249, 330)
    # roi_selector.pt1 = (351, 31)
    # roi_selector.pt2 = (410, 130)
    roi_selector.pt1 = (464, 374) # DJI_0220, f150
    roi_selector.pt2 = (494, 431)
    roi_rect = (roi_selector.pt1[0], roi_selector.pt1[1],
                roi_selector.pt2[0] - roi_selector.pt1[0],
                roi_selector.pt2[1] - roi_selector.pt1[1])

    # roi_selector.select(frame)
    # roi_rect = roi_selector.roi_rect
    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]
    target_label = 'matous'

    # visualizing model
    im_model_vis = frame.copy()
    cv2.rectangle(im_model_vis, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
    cv2.imshow('model', im_model_vis)

    # initializing SecurityCam object -------------------------------
    seccam = SecurityCam()
    seccam.tracker.track_window = roi_rect
    seccam.backprojector.model_im = img_roi
    seccam.backprojector.calc_model_hist()

    # EXAMPLE - REINIT -----------------------------------------------
    # for i in range(450):
    #     ret, frame = video_capture.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outdir = '/home/tomas/Data/sitmp/output'
        video_writer = cv2.VideoWriter(os.path.join(outdir, 'output.avi'), fourcc, 20.0, (frame.shape[1], frame.shape[0]), True)#frame.shape[:2])
    frame_num = 0
    # processing video / camera stream
    while ret:
        frame_num += 1
        if seccam.tracker.track_window is None:
            pass
        label, prob = seccam.process_frame(frame)
        print label, prob
        im_vis = frame.copy()
        # if label is not None and prob > 0.4:
        if label == target_label and prob > 0.6:
            x, y, w, h = seccam.tracker.track_window
            cv2.rectangle(im_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(im_vis, '{}'.format(label[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow('security cam', im_vis)

        if save_output:
            video_writer.write(im_vis)
            if frame_num % 20 == 0:
                fname = os.path.join(outdir, 'frame_{:04d}.png'.format(frame_num))
                cv2.imwrite(fname, im_vis)
        # reading new frame
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        else:
            msg = 'Did not get a frame - end of video file or camera error.'
        # cv2.waitKey(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            msg = 'Terminated by user.'
            break

    print msg
    video_capture.release()
    video_writer.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()