# -*- coding: utf-8 -*-
u"""マップクラスモジュール

ランドマークを保持する
"""

class Map():
    def __init__(self):
        self._landmarks = []

    def landmarks(self):
        return self._landmarks

    def append_landmark(self, landmark):
        u"""ランドマークにIDを与えて保持する

        Args:
            landmark(landmarks.Landmark): ランドマーク
        """
        landmark.id = len(self._landmarks)
        self._landmarks.append(landmark)
