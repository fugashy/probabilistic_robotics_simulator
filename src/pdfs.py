# -*- coding: utf-8 -*-
u"""確率密度関数群

表示したりする
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon, norm


class PdfPlotter():
    def __init__(self, pdf_name):
        self._fig = plt.figure(figsize=(4, 4))
        self._fig.suptitle('PDF of {}'.format(pdf_name))
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel('X', fontsize=2)
        self._ax.set_ylabel('Y', fontsize=2)

    def plot(self, xs):
        raise NotImplementedError(
            'Dear developpers, this is a pure-virtual class')


class ExponPdfPlotter(PdfPlotter):
    def __init__(self, scale):
        super().__init__('expon')
        self._pdf = expon(scale=scale)

    def plot(self, xs):
        ys = [self._pdf.pdf(x) for x in xs]

        self._ax.set_xlim(min(xs), max(xs) + max(xs) * 0.1)
        self._ax.set_ylim(0., max(ys) + max(ys) * 0.1)

        self._ax.scatter(
            xs, ys, s=100, marker='.', color='blue')

        plt.show()
