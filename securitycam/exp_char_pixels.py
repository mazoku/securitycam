from __future__ import division

import cv2
import matplotlib.pyplot as plt
from select_roi import SelectROI


def select_roi(frame):
    roi_selector = SelectROI()
    roi_selector.pt1 = (450, 346)
    roi_selector.pt2 = (512, 508)
    roi_rect = (roi_selector.pt1[0], roi_selector.pt1[1],
                roi_selector.pt2[0] - roi_selector.pt1[0],
                roi_selector.pt2[1] - roi_selector.pt1[1])
    # roi_selector.select(frame)
    # roi_rect = roi_selector.roi_rect
    # print roi_rect

    im_model = frame.copy()
    cv2.rectangle(im_model, roi_selector.pt1, roi_selector.pt2, (0, 255, 0), 1)
    cv2.imshow('model', im_model)

    return roi_rect


def calc_hist(img, sizes=(256, 256, 256), show=False, show_now=True):
    hist = []
    for s, c in zip(sizes, cv2.split(img)):
        h = cv2.calcHist([c], [0], None, [s], [0, s])
        h /= h.sum()
        hist.append(h)

    if show:
        plt.figure()
        plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=3)
        # plt.imshow(img)
        # plt.subplot(411)
        plt.imshow(img)
        for i, (s, h) in enumerate(zip(sizes, hist)):
            # plt.subplot(4, 1, i + 2)
            ax = plt.subplot2grid((3, 6), (i, 3), colspan=3)
            plt.plot(h)
            plt.xlim([0, s])
            ax.yaxis.tick_right()
        if show_now:
            plt.show()

    return hist


def hist_score_pt(image, pt):



def hist_score_im(image, hist):
    pts = image.flatten()
    for p in pts:
        score = 



def char_pixels(frame, model):
    # converting to hsv
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    model_hsv = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)

    # histogram calculation
    # frame_h = []
    # frame_h.append(cv2.calcHist([frame_hsv], [0], None, [180], [0, 180]))
    # frame_h.append(cv2.calcHist([frame_hsv], [1], None, [256], [0, 256]))
    # frame_h.append(cv2.calcHist([frame_hsv], [2], None, [256], [0, 256]))

    # calculate histograms
    hist_frame = calc_hist(frame_hsv, [180, 256, 256], show=True, show_now=False)
    # hist_model = calc_hist(model_hsv, [180, 256, 256], show=True)

    # calculate histogram score
    hist_score_im(model_hsv, hist_frame)


# ---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0220.mp4'
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0221.mp4'
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    # data_path = '/home/tomas/Data/videa/ada1.mp4'
    # data_path = '/home/tomas/Data/videa/ada2.mp4'
    # data_path = '/home/tomas/Data/videa/katka1.mp4'
    # data_path = '/home/tomas/Data/videa/katka2.mp4'

    video_capture = cv2.VideoCapture(data_path)

    for i in range(150):
        ret, frame = video_capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    roi_rect = select_roi(frame)
    img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

    # find characteristic pixels
    char_pixels(frame, img_roi)