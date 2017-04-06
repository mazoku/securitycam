from __future__ import division

import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import cPickle
import os
import imutils
from imutils.paths import list_images

from descriptor import Descriptor


class Classifier:
    def __init__(self, desc=None, data=[], labels=[]):
        self.model = SVC(kernel='linear', C=100, probability=True, random_state=42)
        self.data = data
        self.labels = labels
        self.desc = desc

    def train(self):
        try:
            self.model.fit(self.data, self.labels)
        except:
            pass

    def predict(self, img):
        return self.model.predict(img)

    def predict_proba(self, img):
        return self.model.predict_proba(img)

    def save_model(self, path='../data/classifier.pkl'):
        with open(path, 'w') as f:
            f.write(cPickle.dumps(self.model))

    def train_from_protos(self, objects=['matous', 'ada', 'katka'], train_size=-1, return_test=False):
        if self.desc is None:
            raise AttributeError('Descriptor was not specified.')
        train_data = []
        train_features = []
        train_labels = []

        test_data = []
        test_features = []
        test_labels = []

        for o in objects:
            obj_path = os.path.join('/home/tomas/projects/securitycam/data/{}/protos/'.format(o))
            root, __, files = os.walk(obj_path).next()
            img_paths = [os.path.join(root, x) for x in files]

            if train_size == -1:
                n_train = len(img_paths)
            else:
                n_train = min(train_size, len(img_paths))

            # loop over the input dataset of images
            for i, img_path in enumerate(img_paths):
                # load the image, crop the image, then update the list of objects
                image = cv2.imread(img_path)
                feats = self.desc.describe(image, use_mean=False)[0].flatten()

                if i < n_train:
                    train_data.append(image)
                    train_features.append(feats)
                    train_labels.append(o)
                elif return_test:
                    test_data.append(image)
                    test_features.append(feats)
                    test_labels.append(o)

        self.model.fit(train_features, train_labels)

        if return_test:
            return (train_data, train_features, train_labels), (test_data, test_features, test_labels)
        else:
            return train_data, train_features, train_labels


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    objects = ['matous', 'ada', 'katka']
    train_data = []
    train_features = []
    labels = []

    test_data = []
    test_features = []

    # desc = Descriptor('hsvhist')

    # training model
    # classif = Classifier(train_features, labels)
    # classif.train()
    classif = Classifier()
    classif.desc = Descriptor('hsvhist')
    train_set, test_set = classif.train_from_protos(train_size=3, return_test=True)

    test_data, test_features, test_labels = test_set
    # testing
    results = []
    for td, tf in zip(test_data, test_features):
        lab = classif.predict(tf.reshape(1, -1))
        results.append((lab, td.copy()))

    # display results
    for i, (l, r) in enumerate(results):
        im_vis = imutils.resize(r, height=640)
        cv2.putText(im_vis, '{}'.format(l[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow('test #{}'.format(i + 1), im_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
