from __future__ import division
import cv2
import numpy as np
import imutils
from imutils import paths
from collections import defaultdict
import pickle
import gzip
import os


class Object(object):
    def __init__(self, name):
        self.name = name
        self.protos = []
        self.masks = []
        self.features = []
        self.mean_features = None
        self.median_features = None

    def update_obj(self, img, mask, features):
        self.protos.append(img)
        self.masks.append(mask)
        self.features.append(features)

    def calc_mean(self):
        features = np.array(self.features)
        features = np.mean(features, axis=0)
        return features

    def mark_protos(self, proto_dir):
        # img_paths = list(paths.list_images(proto_dir))
        root, __, files = os.walk(proto_dir).next()
        img_paths = [os.path.join(root, x) for x in files]

        # loop over the input dataset of images
        for img_path in img_paths:
            # load the image, crop the image, then update the list of objects
            image = cv2.imread(img_path)
            roi_rect = cv2.selectROI('select object', image, fromCenter=False)
            roi_rect = map(int, roi_rect)
            obj_im = image[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]]

            self.protos.append(obj_im)

    def display_protos(self, show_now=True):
        resized = []
        for img in self.protos:
            im_res = imutils.resize(img, height=400)
            resized.append(im_res)
        cv2.imshow(self.name, np.hstack(resized))
        if show_now:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def write_protos(self, dir):
        # write to a pickle file
        fname = os.path.join(dir, self.name + '_protos.pklz')
        with gzip.open(fname, 'wb') as f:
            pickle.dump(self.protos, f)

        # write to a directory
        dirname = os.path.join(dir, self.name, 'protos')
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for i, p in enumerate(self.protos):
            fname = os.path.join(dirname, '{}_proto_{:02d}.png'.format(self.name, i + 1))
            cv2.imwrite(fname, p)

    def load_protos(self, dir):
        fname = os.path.join(dir, self.name + '_protos.pklz')
        with gzip.open(fname, 'rb') as f:
            self.protos = pickle.load(f)

    def hist_score_im(self, image, hist, threshold=200, show=False, show_now=True):
        # smoothing
        try:
            ims = cv2.bilateralFilter(image, 9, 75, 75)
        except:
            ims = image.copy()

        # calcuating score
        pts = ims.reshape((-1, 3))  # n_pts * 3 (number of hsv channels
        score = np.zeros(pts.shape[0])
        for i, p in enumerate(pts):
            score[i] = self.hist_score_pt(hist, p)
        # to describe uniqueness of pts, we need to invert the score
        score = score.max() - score

        # reshaping and normalization
        score_im = score.reshape(image.shape[:2])
        cv2.normalize(score_im, score_im, 0, 255, norm_type=cv2.NORM_MINMAX)
        score_im = score_im.astype(np.uint8)

        # h = cv2.calcHist([score_im], [0], None, [256], [0, 256])
        # plt.figure()
        # plt.plot(h)
        # plt.xlim([0, 256])
        # plt.show()

        # thresholding
        score_t = 255 * (score_im > threshold).astype(np.uint8)

        # visualization
        if show:
            cv2.imshow('smoothing', np.hstack((image, ims)))
            cv2.imshow('score', np.hstack((image, cv2.cvtColor(score_t, cv2.COLOR_GRAY2BGR))))
            if show_now:
                cv2.waitKey(0)

        return score_im, score_t

    def hist_score_pt(self, hist, pt):
        score = 0
        for h, p in zip(hist, pt):
            score += h[p]
        return score

    def char_pixels(self, frame, model):
        # converting to hsv
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        model_hsv = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)

        # calculate histograms
        hist_frame = self.calc_hist(frame_hsv, [180, 256, 256])
        # hist_model = calc_hist(model_hsv, [180, 256, 256], show=True)

        # calculate histogram score
        score_im, score_t = self.hist_score_im(model_hsv, hist_frame)
        return score_im, score_t

    def calc_model_hist(self, frame, im=None, mask=None, calc_char=True, show=False, show_now=True):
        if im is None:
            im = self.model_im
        if mask is None and calc_char:
                score_im, mask = self.char_pixels(frame, im)

        model_cs = cv2.cvtColor(im, self.space_code)
        model_hist = cv2.calcHist([model_cs], self.channels, mask, self.sizes, self.ranges)
        cv2.normalize(model_hist, model_hist, 0, 255, cv2.NORM_MINMAX)

        if show:
            if mask is None:
                mask = 255 * np.ones(im.shape[:2], dtype=np.uint8)
            cv2.imshow('model', np.hstack((im, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
            if show_now:
                cv2.waitKey(0)

        self.model_hist = model_hist

    def calc_model_from_protos(self, protos_path):
        root, __, files = os.walk(protos_path).next()
        img_paths = [os.path.join(root, x) for x in files]

        model_hist = np.zeros([self.hist_sizes[i] for i in self.channels])
        for i, img_path in enumerate(img_paths):
            im = cv2.imread(img_path)
            im = cv2.resize(im, (200, 300))
            im = cv2.GaussianBlur(im, (3, 3), 0)
            im_cs = cv2.cvtColor(im, self.space_code)

            h = cv2.calcHist([im_cs], self.channels, None, self.sizes, self.ranges)
            model_hist += h
        model_hist /= len(img_paths)
        self.model_hist = model_hist


if __name__ == '__main__':
    # grab the image paths from the dataset directory
    obj_dir = '../data/'

    # extract object names
    # obj_names = []
    # for (path, dirs, files) in os.walk(obj_dir):
    #     obj_names.extend(dirs)
    __, obj_names, __ = os.walk(obj_dir).next()

    # create objects, mark protos and write them into a file
    objects = []
    for name in obj_names:
        object = Object(name)
        object.mark_protos(os.path.join(obj_dir, name))
        object.write_protos(obj_dir)
        objects.append(object)

    # display each object's protos
    for obj in objects:
        obj.display_protos(show_now=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()