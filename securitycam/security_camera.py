from __future__ import division

import cv2
import numpy as np
import imutils


class SecurityCam(object):
    def __init__(self):
        self.stream = None  # video stream to be analyzed

        self.detect_faces = False  # detect faces around objects?
        self.detect_pedestrians = False  # detect pedestrians around objects?
        self.classify = False  # classify objects?

        self.img_scale = 0.5  # scaling factor

        self.save_output = False  # save visualized output to a file?

    def run(self, frame_ind=0):
        """
        Main method for processing the video stream
        :param frame_ind: index of the first frame to be processed
        :return:
        """

        for i in range(frame_ind):
            ret, frame = self.stream.read()


# -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0220.mp4'
    # data_path = '/home/tomas/Data/videa/ada1.mp4'
    # data_path = '/home/tomas/Data/videa/ada2.mp4'


    seccam = SecurityCam()
    seccam.stream = cv2.VideoCapture(data_path)

    # selecting track window
    frame_ind = 600
    frame, roi_rect, img_roi = seccam.mark_track_window(frame_ind)
    target_label = 'matous'
