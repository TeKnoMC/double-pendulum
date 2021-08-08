"""
This module contains methods to return numerical solutions for
the Double Pendulum problem.
"""
from typing import Iterable, List, Tuple, Callable, Protocol
from scipy.integrate import odeint, solve_ivp

import numpy as np

vector = np.ndarray

class ODE:
    #[[vector, float, Tuple], vector]
    @staticmethod
    def rk4_solve(deriv: Callable,
                    y0: vector, t_arr: vector, args: Tuple) -> vector:
        # while current_t < t_arr[-1]
        # for each eq in deriv and each init value in y0
        #   k1 = deriv(t_n, y_n)
        #   k2 = deriv(t_n + h/2, y_n + h*k1/2)
        #   k3 = deriv(t_n + h/2, y_n + h*k2/2)
        #   k4 = deriv(t_n + h, y_n + h*k3)
        #   y_{n+1} = y_n + (1/6) * h * (k1 + 2*k2 + 2*k3 + k4)
        #   t_{n+1} = t_n + h

        prev_y = y0
        h = t_arr[1] - t_arr[0]

        solution = np.full((len(t_arr), len(y0)), y0)

        for idx in range(1, len(t_arr)):
            k1 = deriv(prev_y, t_arr[idx], *args)
            k2 = deriv(prev_y + (h / 2) * k1, t_arr[idx] + h / 2, *args)
            k3 = deriv(prev_y + (h / 2) * k2, t_arr[idx] + h / 2, *args)
            k4 = deriv(prev_y + h * k3, t_arr[idx] + h, *args)

            y = prev_y + (1.0 / 6.0) * h * (k1 + 2 * k2 + 2 * k3 + k4)
            solution[idx] = y

            prev_y = y

        return solution

class DoublePendulum:
    def __init__(self, l1: float, l2: float, m1: float, m2: float):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

    @staticmethod
    def derivative(init_conditions: vector, t,
                    l1: float, l2: float, m1: float,
                    m2: float, g: float = 9.81) -> vector:
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

        return np.array([w1, w2, w1_dot, w2_dot])

    def numerical_solve(self, dt: float, tmax: float,
                        y0: vector) -> Iterable[vector]:
        """
        Solves the ODE using scipy's odeint(), given the initial
        conditions in y0 and the time variables.
        """
        
        t = np.arange(0, tmax + dt, dt)

        # y has form [[theta_1(0), theta_2(0), w1(0), w2(0)],
        #             [theta_1(dt), theta_2(dt), w1(dt), w2(dt)], ...]
        y = ODE.rk4_solve(DoublePendulum.derivative, y0, t,
                            args=(self.l1, self.l2, self.m1, self.m2))
        
        theta_1 = y[:, 0]   # ':, 0' -> from all elements, return idx 0
        theta_2 = y[:, 1]   # ':, 1' -> from all elements, return idx 1

        return t, theta_1, theta_2
    
    def numerical_solve_scipy(self, dt: float, tmax: float,
                        y0: vector) -> Iterable[vector]:
        """
        Solves the ODE using scipy's odeint(), given the initial
        conditions in y0 and the time variables.
        """
        
        t = np.arange(0, tmax + dt, dt)

        # y has form [[theta_1(0), theta_2(0), w1(0), w2(0)],
        #             [theta_1(dt), theta_2(dt), w1(dt), w2(dt)], ...]
        y = odeint(DoublePendulum.derivative, y0, t,
                            args=(self.l1, self.l2, self.m1, self.m2))
        
        theta_1 = y[:, 0]   # ':, 0' -> from all elements, return idx 0
        theta_2 = y[:, 1]   # ':, 1' -> from all elements, return idx 1

        return t, theta_1, theta_2

    def numerical_solve_scipy_ivp(self, dt: float, tmax: float,
                        y0: vector) -> Iterable[vector]:
        """
        Solves the ODE using scipy's odeint(), given the initial
        conditions in y0 and the time variables.
        """
        
        t = np.arange(0, tmax + dt, dt)

        # y has form [[theta_1(0), theta_2(0), w1(0), w2(0)],
        #             [theta_1(dt), theta_2(dt), w1(dt), w2(dt)], ...]
        y = solve_ivp(DoublePendulum.derivative, (t[0], t[-1]), y0,
                            t_eval=t, method="LSODA",
                            args=(self.l1, self.l2, self.m1, self.m2)).y
        print(y)
        
        theta_1 = y[:, 0]   # ':, 0' -> from all elements, return idx 0
        theta_2 = y[:, 1]   # ':, 1' -> from all elements, return idx 1

        return t, theta_1, theta_2