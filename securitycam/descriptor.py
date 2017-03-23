from __future__ import division

import numpy as np

from hsvhistogram import HSVHistogram
from labhistogram import LabHistogram


class Descriptor:
    def __init__(self, type, bins=([8, 8, 8])):
        if type == 'hsvhist':
            self.desc = HSVHistogram(bins)
        elif type == 'labhist':
            self.desc = LabHistogram(bins)

    def describe(self, imgs, mode='mean'):
        features = []
        for img in imgs:
            img_features = self.desc.describe(img)
            features.append(img_features)

        if mode == 'mean':
            features = np.array(features)
            features = np.mean(features, axis=0)
            features = [features]

        return features


if __name__ == '__main__':
