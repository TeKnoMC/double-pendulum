"""
This module contains methods to return numerical solutions for
the Double Pendulum problem.
"""
from typing import Iterable, List
from scipy.integrate import odeint

import numpy as np

class DoublePendulum:
    def __init__(self, l1: float, l2: float, m1: float, m2: float):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

    @staticmethod
    def derivative(init_conditions: List[float], t,
                    l1: float, l2: float, m1: float,
                    m2: float, g: float = 9.81) -> List[float]:
        """
        Returns the vector [theta_1', theta_2', w1', w2']
        using the equations of motion for a double pendulum
        """

        theta_1, theta_2, w1, w2 = init_conditions

        sine = np.sin(theta_1 - theta_2)
        cosine = np.cos(theta_1 - theta_2)

        denominator1 = l1 * (m1 + m2 * sine * sine)
        denominator2 = l2 * (m1 + m2 * sine * sine)

        numerator1 = (m2 * g * np.sin(theta_2) * cosine
                    - m2 * sine * (l1 * w1 * w1 * cosine + l2 * w2 * w2)
                    - (m1 + m2) * g * np.sin(theta_1))

        numerator2 = ((m1 + m2) * (l1 * w1 * w1 * sine
                    - g * np.sin(theta_2) + g * np.sin(theta_1) * cosine)
                    + m2 * l2 * w2 * w2 * sine * cosine)

        w1_dot = numerator1 / denominator1
        w2_dot = numerator2 / denominator2

        return [w1, w2, w1_dot, w2_dot]

    def numerical_solve(self, dt: float, tmax: float,
                        y0: Iterable[float]) -> Iterable[List[float]]:
        """
        Solves the ODE using scipy's odeint(), given the initial
        conditions in y0 and the time variables.
        """
        
        t = np.arange(0, tmax + dt, dt)

        # y has form [[theta_1(0), theta_2(0), w1(0), w2(0)],
        #             [theta_1(dt), theta_2(dt), w1(dt), w2(dt)], ...]
        y = odeint(DoublePendulum.derivative, y0, t, args=(self.l1, self.l2, self.m1, self.m2))
        
        theta_1 = y[:, 0]   # ':, 0' -> from all elements, return idx 0
        theta_2 = y[:, 1]   # ':, 1' -> from all elements, return idx 1

        return t, theta_1, theta_2