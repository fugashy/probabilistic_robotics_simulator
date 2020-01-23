# -*- coding: utf-8 -*-
from math import sqrt

import robots

class Particle():
    def __init__(self, init_pose):
        self.pose = init_pose

    def motion_update(self, nu, omega, time, noise_rate_pdf):
        # nn, no, on, ooの順にドローされる
        v_nn, v_no, v_on, v_oo = noise_rate_pdf.rvs()

        noised_nu = \
            nu + v_nn * sqrt(abs(nu) / time) + v_no * sqrt(abs(omega) / time)
        noised_omega = \
            omega + v_on * sqrt(abs(nu) / time) + v_oo * sqrt(abs(omega) / time)

        self.pose = robots.IdealRobot.state_transition(
            noised_nu, noised_omega, time, self.pose)
