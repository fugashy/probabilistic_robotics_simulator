# -*- coding: utf-8 -*-
u"""ランドマーク"""
from abc import abstractmethod
import numpy as np

import utilities


class Point2DLandmark():
    u"""2Dの点ランドマーク"""

    def __init__(self, x, y):
        self._pos = np.array([x, y]).T
        self._id = np.nan

    def pos(self):
        return self._pos

    def get_id(self):
        return self._id


class Point2DLandmarkEstimated(Point2DLandmark):
    def __init__(self):
        super().__init__(0., 0.)
        self._cov = None
