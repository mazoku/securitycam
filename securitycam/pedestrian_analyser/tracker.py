from __future__ import division

import cv2
import numpy as np


ref_pt = [None, None]
marked = False
marking = False


def mark_by_mouse(event, x, y, flags, param):
    # grab references to the global variables
    global ref_pt, marking, marked
    # title, img = param
    # img_vis = img.copy()

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt[0] = (x, y)
        ref_pt[1] = (x, y)
        marking = True

    if event == cv2.EVENT_MOUSEMOVE and marking:
        ref_pt[1] = (x, y)
        #     cv2.rectangle(img_vis, ref_pt[0], (x, y), (0, 255, 0), 2)
        #     cv2.imshow(title, img)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that the cropping operation is finished
        # ref_pt.append((x, y))
        if x < ref_pt[0][0]:
            if y < ref_pt[0][1]:
                tl = (x, y)
                br = ref_pt[0]
            else:
                tl = (x, ref_pt[0][1])
                br = (ref_pt[0][0], y)
        else:
            if y < ref_pt[0][1]:
                tl = (ref_pt[0][0], y)
                br = (x, ref_pt[0][1])
            else:
                tl = ref_pt[0]
                br = (x, y)
        ref_pt = [tl, br]
        marking = False
        marked = True

        # draw a rectangle around the region of interest
        # cv2.rectangle(img_vis, ref_pt[0], ref_pt[1], (0, 255, 0), 2)
        # cv2.imshow(title, img)


def get_roi(img):
    title = 'specify ROI'
    cv2.namedWindow(title)
    cv2.imshow(title, img)
    cv2.setMouseCallback(title, mark_by_mouse, (title, img))

    while True:
        # display the image and wait for a keypress
        img_vis = img.copy()

        if marking:
            cv2.rectangle(img_vis, ref_pt[0], ref_pt[1], (0, 255, 0), 2)
        cv2.imshow('specify ROI', img_vis)

        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord('q'):
            return None

        if marked:
            cv2.destroyAllWindows()
            return ref_pt[0][0], ref_pt[0][1], ref_pt[1][0] - ref_pt[0][0], ref_pt[1][1] - ref_pt[0][1]


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

    # setup initial location of window
    # roi_rect = get_roi(frame)
    roi_rect = cv2.selectROI(frame, fromCenter=False)
    print 'specified ROI pts: {}'.format(roi_rect)

    init_once = False

    # tracker = cv2.Tracker_create("MIL")
    # tracker = cv2.Tracker_create("TLD")
    # tracker = cv2.Tracker_create("BOOSTING")
    # tracker = cv2.Tracker_create("KCF")
    tracker = cv2.Tracker_create("MEDIANFLOW")

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print 'no image read'
            break

        if not init_once:
            ok = tracker.init(frame, roi_rect)
            init_once = True

        ok, newbox = tracker.update(frame)
        print ok, newbox

        if ok:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        cv2.imshow("tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    video_capture.release()