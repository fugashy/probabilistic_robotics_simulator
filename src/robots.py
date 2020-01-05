# -*- coding: utf-8 -*-
u"""さまざまな2D移動ロボットを表現するクラス

ノイズ無し・ありなど
"""
from math import sin, cos
import matplotlib.patches as patches

class IdealRobot():
    u"""理想的な2D移動をするロボット"""

    def __init__(self, pose, color):
        u"""初期位置・色の設定

        Args:
            pose(np.array): [x(m), y(m), yaw(rad)]
            color(string): red, green, blue, black, etc...
        """
        self.pose = pose
        self.r = 0.2
        self.color = color

    def draw(self, ax, elems):
        u"""描画

        Args:
            ax(AxesSubplot): 描画オブジェクト
            elems(???)

        Returns:
            なし
        """
        x, y, theta = self.pose

        # ロボットの向きを示す線分
        xn = x + self.r * cos(theta)
        yn = y + self.r * sin(theta)
        # sx.plotがlistを返すので+=とする not append
        elems += ax.plot([x, xn], [y, yn], color=self.color)

        c = patches.Circle(
            xy=(x, y), radius=self.r, fill=False, color=self.color)

        elems.append(ax.add_patch(c))
