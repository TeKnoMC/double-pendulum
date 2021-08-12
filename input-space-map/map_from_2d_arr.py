import matplotlib.pyplot as plt

def row_map(row):
    return list(map(int, row))

with open("image-output/img-4000-40.txt", "r") as f:
    d = f.readlines()
    data = [row.split(',') for row in d[0][:-1].split(';')]
    #print(data)
    int_data = list(map(row_map, data))

plt.imshow(int_data, cmap="inferno")
plt.axis("off")
plt.savefig("img.pdf")
plt.show()