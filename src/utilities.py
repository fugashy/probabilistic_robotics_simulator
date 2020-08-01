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

def matM(nu, omega, dt, stds):
    u"""nu, omega空間の共分散行列を計算する

    速度・角速度の誤差モデル
    正規分布N(u, M)のMを求める

    Args:
        nu(float): 速度[m/s]
        omega(float): 角速度[rad/s]
        dt(float): 前回からの経過時間[sec]
        stds(dict): 並進速度，回転速度2x2=4Dの標準偏差

    Returns:
        共分散行列M_{t}
    """
    # 各標準偏差を求める
    sigma_nn = stds['nn'] * sqrt(abs(nu) / dt)
    sigma_no = stds['no'] * sqrt(abs(omega) / dt)
    sigma_on = stds['on'] * sqrt(abs(nu) / dt)
    sigma_oo = stds['oo'] * sqrt(abs(omega) / dt)

    # 2乗すると分散になる
    # 分散の加法性から，速度，角速度それぞれの成分を足し合わせる
    return np.diag(
        [
            sigma_nn**2 + sigma_no**2,
            sigma_on**2 + sigma_oo**2,
        ])

def matA(nu, omega, dt, theta):
    u"""状態遷移関数をテイラー展開したときの1次の項に登場する係数行列を計算する
    Args:
        nu(float): 速度[m/s]
        omega(float): 角速度[rad/s]
        dt(float): 前回からの経過時間[sec]
        theta(float): 角度

    Returns:
        係数行列A_{t}
    """
    st = sin(theta)
    ct = cos(theta)

    stw = sin(theta + omega * dt)
    ctw = cos(theta + omega * dt)

    a00 = (stw - st) / omega
    a01 = -nu / (omega**2) * (stw - st) + nu / omega * dt * ctw
    a10 = (-ctw + ct) / omega
    a11 = -nu / (omega**2) * (-ctw + ct) + nu / omega * dt * stw
    a20 = 0.
    a21 = dt

    return np.array(
        [
            [a00, a01],
            [a10, a11],
            [a20, a21]
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
