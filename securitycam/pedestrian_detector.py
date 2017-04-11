# import the necessary packages
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


class PedestrianDetector:
    def __init__(self):
        # load the face detector
        self.descriptor = cv2.HOGDescriptor()
        self.descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image, winStride=(4, 4), padding=(8, 8), scale=1.05):
        # detect pedestrians in the image
        rects, weights = self.descriptor.detectMultiScale(image, winStride=winStride, padding=padding, scale=scale)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # rects = rects[np.argmax(weights)]
        # return the bounding boxes around the faces in the image
        return pick


if __name__ == '__main__':
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0220.mp4'
    detector = PedestrianDetector()
    video_capture = cv2.VideoCapture(data_path)

    exp_roi = (426, 281, 88, 100)
    while True:
        ret, frame = video_capture.read()

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        win_region = frame[exp_roi[1]:exp_roi[1] + exp_roi[3], exp_roi[0]:exp_roi[0] + exp_roi[2]]
        rects_f = detector.detect(frame)
        rects_w = detector.detect(win_region)

        frame_vis = frame.copy()
        if rects_f is not None:
            for r in rects_f:
                cv2.rectangle(frame_vis, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 1)
            # cv2.rectangle(frame_vis, (reps.left(), reps.top()), (reps.right(), reps.bottom()), (0, 0, 255), 2)
        elif rects_w is not None:
            for r in rects_w:
                r[0] += exp_roi[0]
                r[1] += exp_roi[1]
                cv2.rectangle(frame_vis, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
        cv2.imshow('', frame_vis)

        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break