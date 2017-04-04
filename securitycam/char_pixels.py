from __future__ import division

import cv2
import numpy as np
from select_roi import SelectROI

from itertools import chain


def back_projection(img, roi, channels=-1, space='hsv'):
    img_roi = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    if space == 'hsv':
        space_code = cv2.COLOR_BGR2HSV
        hist_sizes = [180, 256, 256]
    elif space == 'lab':
        space_code = cv2.COLOR_BGR2Lab
        hist_sizes = [16, 16, 16]
    elif space == 'rgb':
        space_code = cv2.COLOR_BGR2RGB
        hist_sizes = [8, 8, 8]

    if channels == -1:
        channels = range(3)
    elif not isinstance(channels, list):
        channels = [channels]

    sizes = [hist_sizes[i] for i in channels]
    ranges = list(chain.from_iterable([(0, hist_sizes[i]) for i in channels]))

    hsv_roi = cv2.cvtColor(img_roi, space_code)
    hsv_im = cv2.cvtColor(img, space_code)

    # Find the histograms using calcHist. Can be done with np.histogram2d also
    hist_roi = cv2.calcHist([hsv_roi], channels, None, sizes, ranges)
    hist_im = cv2.calcHist([hsv_im], channels, None, sizes, ranges)

    R = hist_roi / hist_im
    h, s, v = cv2.split(hsv_im)
    B = R[h.ravel(), s.ravel()]
    B = np.minimum(B, 1)
    B = B.reshape(hsv_im.shape[:2])

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(B, -1, disc, B)
    B = np.uint8(B)
    cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)

    ret, thresh = cv2.threshold(B, 50, 255, 0)
    return ret, thresh


def char_pixels(img, roi, channels=-1, space='hsv'):
    img_roi = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    if space == 'hsv':
        space_code = cv2.COLOR_BGR2HSV
        hist_sizes = [180, 256, 256]
    elif space == 'lab':
        space_code = cv2.COLOR_BGR2Lab
        hist_sizes = [16, 16, 16]
    elif space == 'rgb':
        space_code = cv2.COLOR_BGR2RGB
        hist_sizes = [8, 8, 8]

    if channels == -1:
        channels = range(3)
    elif not isinstance(channels, list):
        channels = [channels]

    hsv_roi = cv2.cvtColor(img_roi, space_code)
    # roi_hist = cv2.calcHist([hsv_roi], channels, None, [180] * len(channels), [0, 180] * len(channels))
    sizes = [hist_sizes[i] for i in channels]
    ranges = list(chain.from_iterable([(0, hist_sizes[i]) for i in channels]))
    roi_hist = cv2.calcHist([hsv_roi], channels, None, sizes, ranges)
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    hsv_img = cv2.cvtColor(img, space_code)
    dst = cv2.calcBackProject([hsv_img], channels, roi_hist, ranges, 1)

    return dst

# ---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0220.mp4'
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0221.mp4'
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    # data_path = '/home/tomas/Data/videa/ada1.mp4'
    # data_path = '/home/tomas/Data/videa/ada2.mp4'
    # data_path = '/home/tomas/Data/videa/katka1.mp4'
    # data_path = '/home/tomas/Data/videa/katka2.mp4'

    video_capture = cv2.VideoCapture(data_path)

    # take first frame of the video
    # 50 ... zacina bezet horizontalne
    # 180 ... bezi v prave casti obrazovky
    for i in range(1):
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # setup initial location of window
    roi_selector = SelectROI()
    # roi_selector.select(frame)
    # roi_rect = roi_selector.roi_rect
    roi_selector.pt1 = (222, 283)
    roi_selector.pt2 = (249, 330)
    roi_rect = (222, 283, 27, 47)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        # dst = char_pixels(frame, roi_rect, channels=[1, 2], space='hsv')
        ret, dst = back_projection(frame, roi_rect, channels=[0, 1])

        # visualization
        im1 = frame.copy()
        cv2.rectangle(im1, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
        im2 = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im2, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
        im_vis = np.hstack((im1, im2))
        cv2.imshow('back proj', im_vis)

        # k = cv2.waitKey(30) & 0xFF
        #
        # if k == 27:  # Esc
        #     break
        cv2.waitKey(0)

    cv2.destroyAllWindows()


    # while True:
    #     ret, frame = video_capture.read()
