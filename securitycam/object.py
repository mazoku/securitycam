from __future__ import division
import cv2
import numpy as np
import imutils
from imutils import paths
from collections import defaultdict
import pickle
import gzip
import os


class Object:
    def __init__(self, name):
        self.name = name
        self.protos = []
        self.features = []

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