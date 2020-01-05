# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class World(object):
    def __init__(self):
        # 様々なオブジェクトのいれもの
        self.objects = []

    def append(self, obj):
        if not hasattr(obj, 'draw'):
            raise AttributeError('{0} does\'t have draw')
        self.objects.append(obj)

    def draw(self):
        u"""持っているオブジェクトを描画する

        8x8インチの描画エリア
        サブプロット準備
        アスペクト比をそろえる
        5x5mのスケール
        ラベルの表示
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)

        for obj in self.objects:
            obj.draw(ax)

        plt.show()
