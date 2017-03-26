from __future__ import division

import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import cPickle
import os
import imutils

from descriptor import Descriptor


class Classifier:
    def __init__(self, data=[], labels=[]):
        self.model = SVC(kernel='linear', C=100, probability=True, random_state=42)
        self.data = data
        self.labels = labels

    def train(self):
        try:
            self.model.fit(self.data, self.labels)
        except:
            pass

    def predict(self, img):
        return self.model.predict(img)

    def save_model(self, path='../data/classifier.pkl'):
        with open(path, 'w') as f:
            f.write(cPickle.dumps(self.model))


if __name__ == '__main__':
    objects = ['matous', 'ada', 'katka']
    train_data = []
    train_features = []
    labels = []

    test_data = []
    test_features = []

    desc = Descriptor('hsvhist')
    for o in objects:
        obj_path = os.path.join('/home/tomas/projects/securitycam/data/{}/protos/'.format(o))
        root, __, files = os.walk(obj_path).next()
        img_paths = [os.path.join(root, x) for x in files]

        # loop over the input dataset of images
        for i, img_path in enumerate(img_paths):
            # load the image, crop the image, then update the list of objects
            image = cv2.imread(img_path)
            feats = desc.describe(image, use_mean=False)[0].flatten()

            if i < 3:
                train_data.append(image)
                train_features.append(feats)
                labels.append(o)
            else:
                test_data.append(image)
                test_features.append(feats)

    # training model
    classif = Classifier(train_features, labels)
    classif.train()

    # testing
    results = []
    for td, tf in zip(test_data, test_features):
        lab = classif.predict(tf)
        results.append((lab, td.copy()))

    # display results
    for i, (l, r) in enumerate(results):
        im_vis = imutils.resize(r, height=640)
        cv2.putText(im_vis, '{}'.format(l[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow('test #{}'.format(i + 1), im_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
