from __future__ import division

import cv2
import numpy as np
import imutils

from select_roi import SelectROI
from object import Object


class SecurityCam(object):
    def __init__(self):
        # data structures
        self.stream = None  # video stream to be analyzed
        self.objects = []
        self.frame = None
        self.frame_idx = 0

        # flags
        self.detect_faces = False  # detect faces around objects?
        self.detect_pedestrians = False  # detect pedestrians around objects?
        self.classify = False  # classify objects?
        self.save_output = False  # save visualized output to a file?

        # parameters
        self.img_scale = 0.5  # scaling factor

    def mark_new_object(self, frame_idx=0):
        """
        Interactively marking an object in a specific frame.
        :param frame_ind: Index of frame where the object will be marked. The video processing will
        continue from this frame.
        :return: frame - frame where the marking is done
        :return: roi_rect - ROI rectangle: (pt1.x, pt1.y, width, height)
        :return: img_roi - ROI image (cropped from frame)
        """
        # get to the desired frame
        self.frame_idx = frame_idx
        for i in range(self.frame_idx + 1):
            ret, frame = self.stream.read()
        frame = cv2.resize(frame, None, fx=self.img_scale, fy=self.img_scale)

        # mark the object
        roi_selector = SelectROI()
        roi_selector.select(frame)
        roi_rect = roi_selector.roi_rect

        # roi_selector.pt1 = (464, 374) # DJI_0220, f150
        # roi_selector.pt2 = (494, 431)
        # roi_selector.pt1 = (297, 121)  # DJI_0220, f600 - mizi z obrazu
        # roi_selector.pt2 = (416, 319)
        # roi_rect = (roi_selector.pt1[0], roi_selector.pt1[1],
        #             roi_selector.pt2[0] - roi_selector.pt1[0],
        #             roi_selector.pt2[1] - roi_selector.pt1[1])

        img_roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]
        return frame, roi_rect, img_roi

    def detect_new_objects(self):
        """
        Method for detectiong general obejcts - pedestrian detection, moving objects detection etc.
        :return:
        """
        pass

    def create_object(self):
        """
        Create new object from an image ROI.
        :return:
        """

    def update_objects(self):
        """
        Update current objects in the new frame - track, detect, learn, ...
        :return:
        """

    def create_img_vis(self):
        """
        Creating visualization - drawing rectangles around objects, labels above objects etc.
        :return: image augmented for visualization
        """
        img_vis = self.frame.copy()

        return img_vis

    def run(self, frame_ind=0):
        """
        Main method for processing the video stream
        :param frame_ind: index of the first frame to be processed
        :return:
        """

        # get to the desired frame
        for i in range(frame_ind):
            ret, frame = self.stream.read()
        self.frame = cv2.resize(frame, None, fx=self.img_scale, fy=self.img_scale)

        msg = '<Default msg>'
        while True:
            # update current objects
            # --

            # detect new objects
            # --

            # read new frame
            ret, self.frame = self.stream.read()
            if ret:
                self.frame = cv2.resize(self.frame, None, fx=self.img_scale, fy=self.img_scale)
            else:
                msg = 'Did not get a frame - end of video file or camera error.'
                break

            # visualization
            img_vis = self.create_img_vis()
            cv2.imshow('security camera', img_vis)

            # processing user input
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                cv2.waitKey(0)
            elif key == ord('q') or key == 27:
                msg = 'Terminated by user.'
                break

        # finalizing and preparing to exit
        print msg
        self.stream.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    frame, roi_rect, img_roi = seccam.mark_new_object(frame_ind)
    target_label = 'matous'

    matous = Object()
    seccam.objects.append(matous)

    seccam.run()
