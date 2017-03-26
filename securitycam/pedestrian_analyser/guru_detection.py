from __future__ import print_function
# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2


def detect(image, descriptor):
    # image = imutils.resize(image, width=min(400, image.shape[1]))
    image = imutils.resize(image, width=min(600, image.shape[1]))

    # detect people in the image
    (rects, weights) = descriptor.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    # (rects, weights) = descriptor.detectMultiScale(image, winStride=(4, 4), padding=(16, 16), scale=1.05)

    # draw the original bounding boxes
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    return image, rects, pick


if __name__ == '__main__':
    # data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0221.mp4'
    data_path = '/home/tomas/Data/videa/katka1.mp4'
    video_capture = cv2.VideoCapture(data_path)

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        ret, frame = video_capture.read()
        if ret:
            image, rects, picks = detect(frame, hog)
            img_vis = image.copy()
            if rects is not None:
                for (xA, yA, xB, yB) in rects:
                    cv2.rectangle(img_vis, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.imshow('detection', img_vis)

            # quit the program on the press of key 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()