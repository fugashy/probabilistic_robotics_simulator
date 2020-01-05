# -*- coding: utf-8 -*-
from matplotlib import (
    pyplot as plt,
    animation as anm,
    use as enable_plt_option
)
# notebook上でアニメーションできるようになるらしい
enable_plt_option('nbagg')

class World(object):
    def __init__(self, time_span, time_interval, debuggable=False):
        u"""主にシミュレート時間の設定

        Args:
            time_span(float): 何秒シミュレートするのか
            time_interval(float): 何秒間隔でシミュレートするか
            debuggable(bool): アニメートさせるかどうか
        """
        self.time_span = time_span
        self.time_interval = time_interval
        # 様々なオブジェクトのいれもの
        self.objects = []
        # notebook側でエラーがわかりにくいので、必要に応じてメッセージを出すためのフラグ
        self.debuggable = debuggable
        self.ani = None

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
        fig.suptitle('World')
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)

        elems = []

        if self.debuggable:
            for i in range(1000):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(
                fig, self.one_step, fargs=(elems, ax),
                frames=int(self.time_span / self.time_interval),
                interval=int(self.time_interval * 1000), repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        u"""1コマすすめる"""
        while elems:
            elems.pop().remove()

        elems.append(ax.text(
            -4.4, 4.5,
            't= %.2f[s]' % (self.time_interval * i),
            fontsize=10))

        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, 'one_step'):
                obj.one_step(self.time_interval)
