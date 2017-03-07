# import the necessary packages
import cv2


class FaceDetector:
    def __init__(self, faceCascadePath='../../cascades/haarcascade_frontalface_default.xml'):
        # load the face detector
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        # detect faces in the image
        rects = self.faceCascade.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                                  minSize=minSize, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        # return the bounding boxes around the faces in the image
        return rects


if __name__ == '__main__':
    detector = FaceDetector()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        rects = detector.detect(frame)
        if rects is not None:
            frame_vis = frame.copy()
            for r in rects:
                cv2.rectangle(frame_vis, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)
            # cv2.rectangle(frame_vis, (reps.left(), reps.top()), (reps.right(), reps.bottom()), (0, 0, 255), 2)
            cv2.imshow('', frame_vis)
        else:
            cv2.imshow('', frame)

        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break