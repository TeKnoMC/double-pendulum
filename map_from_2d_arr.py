from double_pendulum import DoublePendulum

import matplotlib.pyplot as plt
import numpy as np

def row_map(row):
    return list(map(int, row))

with open("cuda\\cuda-playground\\img.txt", "r") as f:
    d = f.readlines()
    data = [row.split(',') for row in d[0][:-1].split(';')]
    #print(data)
    int_data = list(map(row_map, data))

plt.imshow(int_data, cmap="inferno")
plt.axis("off")
plt.show()