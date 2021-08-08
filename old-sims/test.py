import time
from typing import Generator, Tuple

import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

G = 9.81
K = 1
M = 1

fig, axs = plt.subplots(1, 2)
sine_ax = axs[0]
pendulum_ax = axs[1]

xdata, ydata = [], []
line, = sine_ax.plot([], [])

def displacement_generator() -> Generator:
    dt = 0.05
    end_time = 10
    t = 0

    prev_acc = -G
    prev_vel = 0
    prev_x = 0

    while t <= end_time:
        vel = prev_vel + dt * prev_acc
        x = prev_x + dt * vel
        acc = (-K / M) * x - G

        prev_acc = acc
        prev_vel = vel
        prev_x = x

        t += dt

        yield (t, x)

"""
def frame_data_generator() -> Generator:
    for i in np.linspace(0, 2, 150):
        yield (i, np.sin(2 * np.pi * i))
"""

def init() -> Tuple[Line2D]:
    sine_ax.set_xlim(0, 10)
    sine_ax.set_ylim(-20, 1)
    sine_ax.set_title("Sine Graph")
    sine_ax.set_xlabel("Time (s)")
    sine_ax.set_ylabel("Displacement (m)")
    line.set_data([], [])
    return line,

def update(frame: Tuple[int, float], *kwargs: Tuple) -> Tuple[Line2D]:
    x, y = frame
    xdata.append(x)
    ydata.append(y)
    line.set_data(xdata, ydata)

    return line,

anim = animation.FuncAnimation(fig, update, displacement_generator,
                                init_func=init, blit=True, save_count=200)
anim.save('output\\sine.gif', writer=animation.PillowWriter(fps=20))
