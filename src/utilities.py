# -*- coding: utf-8 -*-
import numpy as np
from math import atan2, cos, pi, sin, sqrt
from matplotlib.patches import Ellipse

def sigma_ellipse(p, cov, n):
    u"""共分散行列の楕円描画オブジェクトを返す

    Args:
        p(list): 描画基準位置
        cov(np.ndarray): 共分散行列
        n(float): スケール?

    Returns:
        matplotlib.patches.Ellipse: 誤差楕円
    """
    eigen_values, eigen_vector = np.linalg.eig(cov)
    angle = atan2(eigen_vector[:, 0][1], eigen_vector[:, 0][0]) / pi * 180.
    return Ellipse(
        p, width=2*n*sqrt(eigen_values[0]), height=2*n*sqrt(eigen_values[1]),
        angle=angle, fill=False, color='blue', alpha=0.5)

def matM(nu, omega, time, stds):
    u"""nu, omega空間の共分散行列を計算する

    弓状になったりする分布をガウス分布で近似したもの

    Args:
        nu(float): 速度[m/s]
        omega(float): 角速度[rad/s]
        time(float): 前回からの経過時間[sec]
        stds(dict): 並進速度，回転速度2x2=4Dの標準偏差

    Returns:
        共分散行列M_{t}
    """
    return np.diag(
        [
            stds['nn']**2. * abs(nu) / time + stds['no']**2. * abs(omega) / time,
            stds['on']**2. * abs(nu) / time + stds['oo']**2. * abs(omega) / time
        ])

def matA(nu, omega, time, theta):
    u"""テイラー展開したときの1次の項に登場する係数行列を計算する
    Args:
        nu(float): 速度[m/s]
        omega(float): 角速度[rad/s]
        time(float): 前回からの経過時間[sec]
        theta(float): 角度

    Returns:
        係数行列A_{t}
    """
    st = sin(theta)
    ct = cos(theta)

    stw = sin(theta + omega * time)
    ctw = cos(theta + omega * time)

    return np.array(
        [
            [(stw - st) / omega, -nu / (omega**2) * (stw - st) + nu / omega * time * ctw],
            [(-ctw + ct) / omega, -nu / (omega**2) * (-ctw + ct) + nu / omega * time * stw],
            [0., time]
        ])

def matF(nu, omega, time, theta):
    u"""状態方程式fを\mu_{t-1}周りでx_{t-1}で偏微分したときのヤコビ行列を計算する
    Args:
        nu(float): 速度[m/s]
        omega(float): 角速度[rad/s]
        time(float): 前回からの経過時間[sec]
        theta(float): 角度

    Returns:
        ヤコビ行列F_{t}
    """
    F = np.diag([1., 1., 1.])

    F[0, 2] = nu / omega * (cos(theta + omega * time) - cos(theta))
    F[1, 2] = nu / omega * (sin(theta + omega * time) - sin(theta))

    return F

def matH(pose, landmark_pos):
    mx, my = landmark_pos
    mux, muy, mut = pose

    q = (mux - mx)**2 + (muy - my)**2

    return np.array(
        [
            [(mux - mx) / np.sqrt(q), (muy - my) / np.sqrt(q), 0.0],
            [(my - muy) / q, (mux - mx) / q, -1.0]
        ])

def matQ(distance_dev, direction_dev):
    return np.diag(np.array([distance_dev**2, direction_dev**2]))
