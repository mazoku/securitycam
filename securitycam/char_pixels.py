from __future__ import division

import cv2
import numpy as np
from select_roi import SelectROI
import matplotlib.pyplot as plt

from itertools import chain


def back_projection(img, roi=None, img_roi=None, channels=-1, space='hsv'):
    if roi is None and img_roi is None:
        raise AttributeError('One of the roi, roi_img must be not None.')
    if roi is not None:
        img_roi = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    # img_roi = cv2.GaussianBlur(img_roi, (3, 3), 0)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img_roi = cv2.medianBlur(img_roi, 3)
    # img = cv2.medianBlur(img, 3)
    # img_roi = cv2.bilateralFilter(img_roi, 5, 41, 21)
    # img = cv2.bilateralFilter(img, 5, 41, 21)

    if space == 'hsv':
        space_code = cv2.COLOR_BGR2HSV
        hist_sizes = [16, 16, 16]
        hist_ranges = [180, 256, 256]
    elif space == 'lab':
        space_code = cv2.COLOR_BGR2Lab
        hist_sizes = [256, 256, 256]
        hist_ranges = [256, 256, 256]
    elif space == 'rgb':
        space_code = cv2.COLOR_BGR2RGB
        hist_sizes = [256, 256, 256]
        hist_ranges = [256, 256, 256]

    if channels == -1:
        channels = range(3)
    elif not isinstance(channels, list):
        channels = [channels]

    sizes = [hist_sizes[i] for i in channels]
    # ranges = list(chain.from_iterable([(0, hist_sizes[i]) for i in channels]))
    ranges = list(chain.from_iterable([(0, hist_ranges[i]) for i in channels]))

    cs_roi = cv2.cvtColor(img_roi, space_code)
    cs_im = cv2.cvtColor(img, space_code)

    # Find the histograms using calcHist. Can be done with np.histogram2d also
    hist_roi = cv2.calcHist([cs_roi], channels, None, sizes, ranges)
    hist_im = cv2.calcHist([cs_im], channels, None, sizes, ranges)
    # plot hist
    # chans = cv2.split(cs_im)
    # plt.figure()
    # for i, c in enumerate(chans):
    #     plt.subplot(3, 1, i + 1)
    #     # c = cv2.normalize(c, None, 0, hist_ranges[i], cv2.NORM_MINMAX)
    #     hist = cv2.calcHist([c], [0], None, [hist_sizes[i]], [0, hist_ranges[i]])
    #     minv, maxv, minloc, maxloc = cv2.minMaxLoc(hist)
    #     # hist = np.bincount(c.ravel(), minlength=hist_sizes[i])
    #     plt.plot(hist)
    #     plt.xlim([0, hist_sizes[i]])
    # plt.show()

    R = hist_roi / hist_im
    # c1, c2, c3 = cv2.split(cs_im)
    # B = R[c1.ravel(), c2.ravel()]
    # cs_im = cv2.merge(chans_img)
    chans = cv2.split(cs_im)
    # chans = [cv2.normalize(chans[i], None, 0, hist_sizes[i] - 1, cv2.NORM_MINMAX) for i in range(3)]
    B = R[tuple(chans[i].ravel() for i in channels)]
    B = np.minimum(B, 1)
    B = B.reshape(cs_im.shape[:2])

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(B, -1, disc, B)
    B = np.uint8(B)
    cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)

    ret, thresh = cv2.threshold(B, 50, 255, 0)
    return ret, thresh


def char_pixels(img, roi=None, img_roi=None, channels=-1, space='hsv'):
    if roi is None and img_roi is None:
        raise AttributeError('One of the roi, roi_img must be not None.')
    if roi is not None:
        img_roi = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    if space == 'hsv':
        space_code = cv2.COLOR_BGR2HSV
        hist_sizes = [180, 256, 256]
        hist_ranges = [180, 256, 256]
    elif space == 'lab':
        space_code = cv2.COLOR_BGR2Lab
        hist_sizes = [16, 16, 16]
        hist_ranges = [256, 256, 256]
    elif space == 'rgb':
        space_code = cv2.COLOR_BGR2RGB
        hist_sizes = [8, 8, 8]
        hist_ranges = [256, 256, 256]

    if channels == -1:
        channels = range(3)
    elif not isinstance(channels, list):
        channels = [channels]

    sizes = [hist_sizes[i] for i in channels]
    ranges = list(chain.from_iterable([(0, hist_ranges[i]) for i in channels]))

    cs_roi = cv2.cvtColor(img_roi, space_code)
    cs_im = cv2.cvtColor(img, space_code)

    roi_hist = cv2.calcHist([cs_roi], channels, None, sizes, ranges)
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # hsv_img = cv2.cvtColor(img, space_code)
    dst = cv2.calcBackProject([cs_im], channels, roi_hist, ranges, 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

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
    roi_selector.select(frame)
    roi_rect = roi_selector.roi_rect
    # roi_selector.pt1 = (222, 283)
    # roi_selector.pt2 = (249, 330)
    # roi_rect = (222, 283, 27, 47)

    ret, frame = video_capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

    # roi_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(roi_hsv)
    # im_vis = np.hstack((h, s, v))
    # cv2.imshow('HSC channels', im_vis)
    # cv2.waitKey(0)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        dst = char_pixels(frame, img_roi=img_roi, channels=[0, 1], space='hsv')
        # ret, dst = back_projection(frame, img_roi=img_roi, channels=[0, 1], space='hsv')

        # visualization
        im1 = frame.copy()
        cv2.rectangle(im1, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
        im2 = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im2, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
        im_vis = np.hstack((im1, im2))
        cv2.imshow('back proj', im_vis)
        # cv2.waitKey(0)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc
            break
        # cv2.waitKey(1)

    cv2.destroyAllWindows()


    # while True:
    #     ret, frame = video_capture.read()
