# -*- coding: utf-8 -*-
u"""ベイスの定理をより理解するための遊び場を作る

遊びやすくするために，データの保持，可視化を効率よく行えるようにする

そんなクラスを作る
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MultiModalDistributionHelper():
    def __init__(self, data_dir_path):
        u"""
                date    time    ir   lidar
            20180202  110001    28     627
                日付  時分秒  強度    強度
        """
        self._data_raw = pd.read_csv(
            data_dir_path, delimiter=' ', header=None,
            names=('date', 'time', 'ir', 'lidar'))

    def print_data_raw(self):
        print(self._data_raw)

    def show_histgram(self, key_name):
        self._data_raw[key_name].hist(
            bins=max(self._data_raw[key_name])-min(self._data_raw[key_name]),
            align='left')
        plt.show()

    def show_plot(self, key_name):
        self._data_raw[key_name].plot()
        plt.show()
