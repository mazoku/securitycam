from __future__ import division

import sys
import dlib
import cv2
from skimage import io


def detect(img, face_detector):
    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(img, 1)

    return detected_faces


if __name__ == '__main__':
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    video_capture = cv2.VideoCapture(0)
    # win = dlib.image_window()

    while True:
        ret, frame = video_capture.read()

        # rects = detector.detect(frame)
        faces = detect(frame, face_detector)
        print "I found {} faces".format(len(faces))

        # Open a window on the desktop showing the image
        # win.set_image(frame)
        if faces is not None:
            frame_vis = frame.copy()
            for i, face_rect in enumerate(faces):
                # Draw a box around each face we found
                cv2.rectangle(frame_vis, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
                cv2.imshow('', frame_vis)
        else:
            cv2.imshow('', frame)

        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break