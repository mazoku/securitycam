from __future__ import division

import cv2
from classifier import Classifier
from descriptor import Descriptor
from back_projector import BackProjector
from motion_detector import MotionDetector
from tracker import Tracker
from select_roi import SelectROI


class SecurityCam:
    def __init__(self):
        self.stream = None

        # self.tracker = cv2.Tracker_create(tracker)
        self.descriptor = Descriptor('hsvhist')
        self.classifier = Classifier(desc=self.descriptor)
        self.classifier.train_from_protos(train_size=-1, return_test=False)
        self.backprojector = BackProjector(space='hsv', channels=[0, 1])
        self.tracker = Tracker()
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

        self.backprojector.calc_heatmap(frame, convolution=True, morphology=False)
        self.tracker.track(frame, self.backprojector.heat_map, track_window=self.tracker.track_window)

        # im_vis = frame.copy()
        # x, y, w, h = self.tracker.prev_track_window
        # cv2.rectangle(im_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # x, y, w, h = self.tracker.track_window
        # cv2.rectangle(im_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('tracker history', im_vis)
        # cv2.waitKey(0)

        #TODO: jak se lisi vysledky, kdyz use_mean=True
        feats = self.descriptor.describe(track_im, use_mean=False)[0].flatten()
        label = self.classifier.predict(feats.reshape(1, -1))
        classes = self.classifier.model.classes_
        probs = self.classifier.predict_proba(feats.reshape(1, -1))[0, :]
        res = [(c, p) for c, p in zip(classes, probs)]
        print res
        return label


if __name__ == '__main__':
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0220.mp4'
    video_capture = cv2.VideoCapture(data_path)

    # selecting model
    for i in range(150):
        ret, frame = video_capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    roi_selector = SelectROI()
    roi_selector.select(frame)
    roi_rect = roi_selector.roi_rect
    # roi_selector.pt1 = (222, 283)
    # roi_selector.pt2 = (249, 330)
    # roi_selector.pt1 = (351, 31)
    # roi_selector.pt2 = (410, 130)
    # roi_rect = (roi_selector.pt1[0], roi_selector.pt1[1],
    #             roi_selector.pt2[0] - roi_selector.pt1[0],
    #             roi_selector.pt2[1] - roi_selector.pt1[1])
    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

    # visualizing model
    im_model_vis = frame.copy()
    cv2.rectangle(im_model_vis, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
    cv2.imshow('model', im_model_vis)

    # initializing SecurityCam object -------------------------------
    seccam = SecurityCam()
    seccam.tracker.track_window = roi_rect
    seccam.backprojector.model_im = img_roi
    seccam.backprojector.calc_model_hist()

    # processing video / camera stream
    while ret:
        label = seccam.process_frame(frame)
        im_vis = frame.copy()
        x, y, w, h = seccam.tracker.track_window
        cv2.rectangle(im_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(im_vis, '{}'.format(label[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow('security cam', im_vis)
        # reading new frame
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        else:
            msg = 'Did not get a frame - end of video file or camera error.'
        # cv2.waitKey(0)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            msg = 'Terminated by user.'
            break

    print msg
    cv2.waitKey(0)
    cv2.destroyAllWindows()