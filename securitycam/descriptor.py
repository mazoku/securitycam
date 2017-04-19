from __future__ import division

import numpy as np
import cv2
import imutils

from hsvhistogram import HSVHistogram
from labhistogram import LabHistogram


class Descriptor:
    def __init__(self, type, bins=([8, 8, 8])):
        if type == 'hsvhist':
            self.desc = HSVHistogram(bins)
        elif type == 'labhist':
            self.desc = LabHistogram(bins)

    def describe(self, imgs, use_mean=True):
        if not isinstance(imgs, list):
            imgs = [imgs]
        features = []
        for img in imgs:
            img_features = self.desc.describe(img)
            features.append(img_features)

        if use_mean:
            features = np.array(features)
            features = np.mean(features, axis=0)
            features = [features]

        return features

    def describe_char(self, img):
        pass

    @staticmethod
    def chi2_distance(hist1, hist2, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(hist1, hist2)])

        # return the chi-squared distance
        return d

    def compare_hists(self, hist1, hist2, method='chi2'):
        if method == 'chi2':
            d = self.chi2_distance(hist1, hist2)

        # print 'dist = {:.3f}'.format(d)
        # return the chi-squared distance
        return d


if __name__ == '__main__':
    ada1 = cv2.imread('/home/tomas/projects/securitycam/data/ada/protos/ada_proto_02.png')
    ada2 = cv2.imread('/home/tomas/projects/securitycam/data/ada/protos/ada_proto_05.png')
    mat1 = cv2.imread('/home/tomas/projects/securitycam/data/matous/protos/matous_proto_01.png')
    mat2 = cv2.imread('/home/tomas/projects/securitycam/data/matous/protos/matous_proto_04.png')

    desc = Descriptor('hsvhist')
    # desc = Descriptor('labhist')
    pairs = [(ada1, ada2), (mat1, mat2), (ada1, mat1), (ada2, mat2)]
    results = []
    for im1, im2 in pairs:
        feat1 = desc.describe(im1, use_mean=False)[0]
        feat2 = desc.describe(im2, use_mean=False)[0]
        d = desc.compare_hists(feat1, feat2)

        if im1.shape[0] > im2.shape[0]:
            im_vis = np.hstack((im1, imutils.resize(im2, height=im1.shape[0])))
        else:
            im_vis = np.hstack((imutils.resize(im1, height=im2.shape[0]), im2))
        cv2.putText(im_vis, 'dist = {:.3f}'.format(d), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        results.append(im_vis.copy())

    for i, r in enumerate(results):
        cv2.imshow('res #{}'.format(i+1), r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()