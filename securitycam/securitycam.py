from __future__ import division

import cv2
from classifier import Classifier
from descriptor import Descriptor


class SecurityCam:
    def __init__(self, tracker='KCF', descriptor='hsvhist'):
        self.stream = None
        if tracker not in ('KCF', 'TLD', 'MIL', 'BOOSTING', 'MEDIANFLOW'):
            print 'Warning! Unknown tracker type {}, using default KCF'.format(tracker)
            tracker = 'KCF'
        self.tracker = cv2.Tracker_create(tracker)
        self.classifier = Classifier()

        if descriptor not in ('hsvhist', 'labhist'):
            print 'Warning! Unknown descriptor type {}, using default hsvhist'.format(descriptor)
            descriptor = 'hsvhist'
        self.descriptor = Descriptor(descriptor)




if __name__ == '__main__':
    pass