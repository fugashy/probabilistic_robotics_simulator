# -*- coding: utf-8 -*-
u"""ランドマーク"""
from abc import abstractmethod
import numpy as np

import utilities


class Point2DLandmark():
    u"""2Dの点ランドマーク"""

    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = np.nan


class Point2DLandmarkEstimated(Point2DLandmark):
    def __init__(self):
        super().__init__(0., 0.)
        self.cov = None
