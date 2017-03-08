import time
import os
import argparse

import openface
import cv2


def get_rep(img, align, args, multiple=False):
    start = time.time()
    if isinstance(img, basestring):
        bgr_img = cv2.imread(img)
    else:
        bgr_img = img.copy()

    rgbImg = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # if args.verbose:
    #     print("  + Original size: {}".format(rgbImg.shape))
    # if args.verbose:
    #     print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        # raise Exception("Unable to find a face: {}".format(img_path))
        print '! Unable to find a face.'
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    # reps = []
    # for bb in bbs:
    #     start = time.time()
    #     alignedFace = align.align(args.imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    #     if alignedFace is None:
    #         # raise Exception("Unable to align image: {}".format(img_path))
    #         print '! Unable to align image.'
    #         return None
    #     if args.verbose:
    #         print("Alignment took {} seconds.".format(time.time() - start))
    #         print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))
    #
    #     start = time.time()
    #     rep = net.forward(alignedFace)
    #     if args.verbose:
    #         print("Neural network forward pass took {} seconds.".format(time.time() - start))
    #     reps.append((bb.center().x, rep))
    # sreps = sorted(reps, key=lambda x: x[0])
    sreps = bb1
    return sreps


if __name__ == '__main__':
    openface_dir = '/home/tomas/openface-master'
    model_dir = os.path.join(openface_dir, 'models')
    # dlibModelDir = os.path.join(modelDir, 'dlib')
    dlib_model_dir = os.path.join(model_dir, 'dlib')
    openface_model_dir = os.path.join(model_dir, 'openface')

    parser = argparse.ArgumentParser()
    parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                        default=os.path.join(openface_model_dir, 'nn4.small2.v1.t7'))
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--captureDevice', type=int, default=0,
                        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)

    args = parser.parse_args()
    args.verbose = True

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    while True:
        ret, frame = video_capture.read()

        # img_path = ''
        # params =
        reps = get_rep(frame, align, args)

        if reps is not None:
            frame_vis = frame.copy()
            cv2.rectangle(frame_vis, (reps.left(), reps.top()), (reps.right(), reps.bottom()), (0, 0, 255), 2)
            cv2.imshow('', frame_vis)
        else:
            cv2.imshow('', frame)

        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()