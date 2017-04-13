from __future__ import division

import cv2
import numpy as np
from select_roi import SelectROI
import matplotlib.pyplot as plt

from itertools import chain
import os


def char_pixels(img, roi=None, img_roi=None, model_hist=None, channels=-1, space='hsv'):
    if roi is None and img_roi is None and model_hist is None:
        raise AttributeError('One of the roi, roi_img or model_hist must be not None.')
    if roi is not None:
        img_roi = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    if space == 'hsv':
        space_code = cv2.COLOR_BGR2HSV
        hist_sizes = [180, 16, 16]
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

    # preparing model
    if model_hist is None:
        cs_roi = cv2.cvtColor(img_roi, space_code)
        roi_hist = cv2.calcHist([cs_roi], channels, None, sizes, ranges)
    else:
        roi_hist = model_hist
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # converting input
    cs_im = cv2.cvtColor(img, space_code)
    cs_im = cv2.GaussianBlur(cs_im, (3, 3), 0)

    # roi_hist = cv2.calcHist([cs_roi], channels, None, sizes, ranges)
    # cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # hsv_img = cv2.cvtColor(img, space_code)
    dst = cv2.calcBackProject([cs_im], channels, roi_hist, ranges, 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)
    _, t = cv2.threshold(dst, 180, 255, cv2.THRESH_BINARY)
    t = cv2.erode(t, disc, iterations=3)
    t = cv2.dilate(t, disc, iterations=3)

    # return dst
    return t


def model_from_protos(protos_dir, channels, space='hsv'):
    if space == 'hsv':
        space_code = cv2.COLOR_BGR2HSV
        hist_sizes = [180, 16, 16]
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

    root, __, files = os.walk(protos_dir).next()
    img_paths = [os.path.join(root, x) for x in files]

    # loop over the input dataset of images
    # model_hist = [np.zeros(hist_sizes[i]) for i in range(3)]
    model_hist = np.zeros([hist_sizes[i] for i in channels])
    for i, img_path in enumerate(img_paths):
        im = cv2.imread(img_path)
        im = cv2.resize(im, (200, 300))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        im_cs = cv2.cvtColor(im, space_code)

        h = cv2.calcHist([im_cs], channels, None, sizes,ranges)
        model_hist += h
        # for i in range(h.shape[2]):
        #     cv2.imshow('value channel', h[:, :, i])
        #     cv2.waitKey(0)
        # cv2.imshow('input', im_cs)
        # cv2.waitKey(0)

        # for j in range(3):
        #     hj = cv2.calcHist([im_cs], [j], None, [hist_sizes[j]], [0, hist_ranges[j]])[:, 0]
        #     model_hist[j] += hj

    # model_hist = [x / len(img_paths) for x in model_hist]
    model_hist /= len(img_paths)
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(3, 1, i + 1), plt.plot(model_hist[i]), plt.xlim([0, hist_sizes[i]])
    # plt.show()

    return model_hist


# ---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0220.mp4'
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0221.mp4'
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    data_path = '/home/tomas/Data/videa/ada1.mp4'
    # data_path = '/home/tomas/Data/videa/ada2.mp4'
    # data_path = '/home/tomas/Data/videa/katka1.mp4'
    # data_path = '/home/tomas/Data/videa/katka2.mp4'

    video_capture = cv2.VideoCapture(data_path)

    output_fname = '/home/tomas/temp/cv_seminar/hsv_ada.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # creating model
    # model_hist = model_from_protos('../data/matous/protos/', channels=[0, 1])

    # take first frame of the video
    # 50 ... zacina bezet horizontalne
    # 180 ... bezi v prave casti obrazovky
    for i in range(150):
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # video writer initialization
    video_writer = cv2.VideoWriter(output_fname, fourcc, 30.0, (2 * frame.shape[1], 2 * frame.shape[0]), True)

    # setup initial location of window
    roi_selector = SelectROI()
    # roi_selector.select(frame)
    # roi_rect = roi_selector.roi_rect
    roi_selector.pt1 = (222, 283)
    roi_selector.pt2 = (249, 330)
    roi_selector.pt1 = (351, 31)
    roi_selector.pt2 = (410, 130)
    roi_rect = (roi_selector.pt1[0], roi_selector.pt1[1],
                roi_selector.pt2[0] - roi_selector.pt1[0],
                roi_selector.pt2[1] - roi_selector.pt1[1])

    # ret, frame = video_capture.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

    # roi_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(roi_hsv)
    # im_vis = np.hstack((h, s, v))
    # cv2.imshow('HSC channels', im_vis)
    # cv2.waitKey(0)

    im_model = frame.copy()
    cv2.rectangle(im_model, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
    cv2.imshow('model', im_model)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # colorspace channel visualization
        c1, c2, c3 = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        # # c1, c2, c3 = cv2.split(frame)  # BGR
        row1 = np.hstack((frame, cv2.cvtColor(c1, cv2.COLOR_GRAY2BGR)))
        row2 = np.hstack((cv2.cvtColor(c2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(c3, cv2.COLOR_GRAY2BGR)))
        im_vis = np.vstack((row1, row2))
        video_writer.write(im_vis)
        cv2.imshow('channels', im_vis)
        # cv2.waitKey(20)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc
            break
        # cv2.waitKey(1)

    cv2.destroyAllWindows()


    # while True:
    #     ret, frame = video_capture.read()
