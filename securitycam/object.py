from __future__ import division
import cv2
import numpy as np
import imutils
from imutils import paths
from collections import defaultdict
import pickle
import gzip
import os

from back_projector import BackProjector
from tracker import Tracker
from descriptor import Descriptor


class Object(object):
    def __init__(self, name):
        self.name = name
        self.protos = []
        self.masks = []
        self.features = []
        self.mean_features = None
        self.median_features = None

        self.back_projector = BackProjector(space='hsv', channels=[0, 1])
        self.tracker = Tracker()
        self.descriptor = Descriptor('hsvhist')