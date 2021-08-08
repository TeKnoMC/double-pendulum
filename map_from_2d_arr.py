from double_pendulum import DoublePendulum

import matplotlib.pyplot as plt
import numpy as np

def row_map(row):
    return list(map(int, row))

with open("cuda\\500-test.txt", "r") as f:
    d = f.readlines()
    data = [row[:-1].split(',') for row in d[0][:-1].split(';')]
    int_data = list(map(row_map, data))

plt.imshow(int_data, cmap="bone")
plt.axis("off")
plt.show()