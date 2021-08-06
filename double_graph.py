from double_pendulum import DoublePendulum
from typing import Generator, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

M1 = 1
M2 = 1
L1 = 0.5
L2 = 1

DT = 0.05
TMAX = 30

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(20, 14)

curve_ax = axs[0]
pendulum_ax = axs[1]

arm1, = pendulum_ax.plot([], [], 'k')
arm2, = pendulum_ax.plot([], [], 'k')
bob1, = pendulum_ax.plot([], [], 'bo', markersize=15)
bob2, = pendulum_ax.plot([], [], 'ro', markersize=15)

t1_data, t2_data = [], []
curve, = curve_ax.plot([], [])

def frame_generator() -> Generator:
    p = DoublePendulum(L1, L2, M1, M2)
    y0 = np.array([np.pi / 8, np.pi / 8, 0, 0])
    t, theta_1, theta_2 = p.numerical_solve(DT, TMAX, y0)

    for i in range(len(t)):
        yield (t[i], theta_1[i], theta_2[i])

def init() -> Tuple:
    curve_ax.set_xlim(-2 * np.pi / 8, 2 * np.pi / 8)
    curve_ax.set_ylim(-2 * np.pi / 8, 2 * np.pi / 8)
    curve_ax.set_title("theta_2 against theta_1")
    curve_ax.set_xlabel("theta_1")
    curve_ax.set_ylabel("theta_2")
    curve_ax.grid()
    curve_ax.set_aspect('equal')
    curve.set_data([], [])

    bounds = (11 / 10) * (L1 + L2)

    pendulum_ax.set_xlim(-bounds, bounds)
    pendulum_ax.set_ylim(-bounds, bounds)
    pendulum_ax.set_title("Pendulum")
    pendulum_ax.set_xlabel("x")
    pendulum_ax.set_ylabel("y")
    pendulum_ax.grid()
    pendulum_ax.set_aspect("equal")
    arm1.set_data([], [])
    arm2.set_data([], [])
    bob1.set_data([], [])
    bob2.set_data([], [])
    return arm1, arm2, bob1, bob2,

def update(frame: Tuple[float, float, float], *kwargs: Tuple) -> Tuple:
    t, theta_1, theta_2 = frame

    t1_data.append(theta_1)
    t2_data.append(theta_2)
    curve.set_data(t1_data, t2_data)

    x1 = L1 * np.sin(theta_1)
    y1 = -L1 * np.cos(theta_1)
    x2 = x1 + L2 * np.sin(theta_2)
    y2 = y1 - L2 * np.cos(theta_2)

    arm1.set_data([0, x1], [0, y1])
    arm2.set_data([x1, x2], [y1, y2])
    bob1.set_data([x1], [y1])
    bob2.set_data([x2], [y2])

    return arm1, arm2, bob1, bob2

anim = animation.FuncAnimation(fig, update, frame_generator,
                                init_func=init, blit=True, save_count=800)
anim.save('output\\double.gif', writer=animation.PillowWriter(fps=int(1 / DT)))
