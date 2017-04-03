from __future__ import division

import cv2
import numpy as np
import imutils


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
    data_path = '/home/tomas/Data/sitmp/Matous_tracking_Z30/DJI_0222.mp4'

    video_capture = cv2.VideoCapture(data_path)

    # take first frame of the video
    # 50 ... zacina bezet horizontalne
    # 180 ... bezi v prave casti obrazovky
    for i in range(100):
        ret, frame = video_capture.read()

    frame = imutils.resize(frame, width=800)

    # setup initial location of window
    roi_rect = get_roi(frame)
    track_window = roi_rect
    print 'specified ROI pts: {}'.format(roi_rect)

    # set up the ROI for tracking
    roi = frame[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    mask = None
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while (1):
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=800)

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            # ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # Draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), 255, 2)
            # img2 = dst
            # cv2.imshow('img2', img2)
            img_vis = np.hstack((frame, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)))
            cv2.imshow('img2', img_vis)

            key = cv2.waitKey(60) & 0xFF
            # if the 'c' key is pressed, break from the loop
            if key == ord('q') or key == 27:
                break

    cv2.destroyAllWindows()
    video_capture.release()


    # while True:
    #     ret, frame = video_capture.read()
    #     if ret:
    #         image, rects, picks = detect(frame, hog)
    #         img_vis = image.copy()
    #         if rects is not None:
    #             for (xA, yA, xB, yB) in rects:
    #                 cv2.rectangle(img_vis, (xA, yA), (xB, yB), (0, 255, 0), 2)
    #         cv2.imshow('detection', img_vis)
    #
    #         # quit the program on the press of key 'q'
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    # # When everything is done, release the capture
    # video_capture.release()
    # cv2.destroyAllWindows()

# -------------------------------------
# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values
# r,h,c,w = 200,20,300,20
# track_window = (c,r,w,h)
#
# # set up the ROI for tracking
# roi = frame[r:r+h, c:c+w]
# hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#
# # Setup the termination criteria, either 10 iteration or move by at least 1 pt
# term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
#
# while(1):
#     ret ,frame = cap.read()
#
#     if ret == True:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#
#         # apply meanshift to get the new location
#         ret, track_window = cv2.meanShift(dst, track_window, term_crit)
#
#         # Draw it on image
#         x,y,w,h = track_window
#         img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
#         cv2.imshow('img2',img2)
#
#         k = cv2.waitKey(60) & 0xff
#         if k == 27:
#             break
#         else:
#             cv2.imwrite(chr(k)+".jpg",img2)
#
#     else: