# -*- coding: utf-8 -*-
u"""マップクラスモジュール

ランドマークを保持する
"""


class Map():
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        u"""ランドマークにIDを与えて保持する

        Args:
            landmark(landmarks.Landmark): ランドマーク
        """
        landmark.id = len(self.landmarks) - 1
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)
