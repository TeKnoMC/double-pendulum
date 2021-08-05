from typing import Generator, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

G = 9.81
L = 1

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(20, 14)

curve_ax = axs[0]
pendulum_ax = axs[1]

pendulum, = pendulum_ax.plot([], [], 'k')
bob, = pendulum_ax.plot([], [], 'bo', markersize=15)

t_data, theta_data = [], []
curve, = curve_ax.plot([], [])

def displacement_generator() -> Generator:
    dt = 0.05
    end_time = 10
    t = 0

    prev_theta = np.pi / 4
    prev_angular_vel = 0
    prev_angular_acc = (-G / L) * np.sin(prev_theta)
    

    while t <= end_time:
        angular_vel = prev_angular_vel + dt * prev_angular_acc
        theta = prev_theta + dt * angular_vel
        angular_acc = (-G / L) * np.sin(theta)

        prev_angular_acc = angular_acc
        prev_angular_vel = angular_vel
        prev_theta = theta

        t += dt

        yield (t, theta)


def init() -> Tuple:
    curve_ax.set_xlim(0, 10)
    curve_ax.set_ylim(-4, 4)
    curve_ax.set_title("Angular Displacement over Time")
    curve_ax.set_xlabel("Time (s)")
    curve_ax.set_ylabel("Angular Displacement (rad)")
    curve_ax.grid()
    curve_ax.set_aspect('equal')
    curve.set_data([], [])

    screen_bounds = (11/10) * L

    pendulum_ax.set_xlim(-screen_bounds, screen_bounds)
    pendulum_ax.set_ylim(-screen_bounds, screen_bounds)
    pendulum_ax.set_title("Pendulum")
    pendulum_ax.set_xlabel("x")
    pendulum_ax.set_ylabel("y")
    pendulum_ax.grid()
    pendulum_ax.set_aspect("equal")
    pendulum.set_data([], [])
    bob.set_data([], [])
    return curve, pendulum, bob,

def update(frame: Tuple[int, float], *kwargs: Tuple) -> Tuple:
    t, theta = frame
    t_data.append(t)
    theta_data.append(theta)
    curve.set_data(t_data, theta_data)

    x = L * np.sin(theta)
    y = -L * np.cos(theta)

    pendulum.set_data([0, x], [0, y])
    bob.set_data([x], [y])

    return curve, pendulum, bob,

anim = animation.FuncAnimation(fig, update, displacement_generator,
                                init_func=init, blit=True, save_count=200)
anim.save('output\\pendulum.gif', writer=animation.PillowWriter(fps=20))
