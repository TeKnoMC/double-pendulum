"""
Generates a map visualising the input space for the double pendulum,
where the colour represents how long it takes for the system to
flip over (exhibiting chaotic motion).
"""

from typing import Tuple
from double_pendulum import DoublePendulum

import numpy as np
import matplotlib.pyplot as plt

L1 = 1
L2 = 1
M1 = 1
M2 = 1
T1_MIN = -np.pi
T1_MAX = np.pi
T2_MIN = -np.pi
T2_MAX = np.pi

def get_chaos_rating(init_conditions: np.ndarray) -> int:
    """
    Returns the time it takes for the system to loop over (maximum value if never)
    """

    dp = DoublePendulum(L1, L2, M1, M2)
    t, theta_1, theta_2 = dp.numerical_solve(0.01, 10, init_conditions)

    for i in range(len(t)):
        if abs(theta_1[i]) > np.pi or abs(theta_2[i]) > np.pi:
            return i

    return len(t)

def map_to_range(i: int, j: int, imin: float, imax: float,
                    jmin: float, jmax: float, length: int) -> Tuple[float, float]:
    return ((i / (length-1)) * (imax - imin) + imin,
            (j / (length-1)) * (jmax - jmin) + jmin)

def generate_image(dimension: int) -> np.ndarray:
    img_arr = np.full((dimension, dimension), 255)

    for i in range(dimension):
        for j in range(dimension):
            x, y = map_to_range(j, i, T1_MIN, T1_MAX, T2_MIN, T2_MAX, dimension)
            img_arr[i][j] = get_chaos_rating(np.array([x, y, 0, 0]))
        
    
    return img_arr

img_data = generate_image(100)

#np.save("output\\pixel_data.npy", img_data)
#img_data = np.load("output\\pixel_data_350.npy")

plt.imshow(img_data, cmap="BuPu")
plt.axis("off")
plt.show()
