import numpy as np


def load_data(file_name):
    num_feat = len(open(file_name).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regres(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    x_T_x = x_mat.T * x_mat
    if np.linalg.det(x_T_x) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = x_T_x.I * (x_mat.T * y_mat)
    return ws


x, y = load_data('ex0.txt')
result = stand_regres(x, y)
x = np.mat(x)
y = np.mat(y)
y_hat = x * result

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x[:, 1].flatten().A[0], y.T[:, 0].flatten().A[0])
x_copy = x.copy()
x_copy.sort(0)
y_hat = x_copy * result
ax.plot(x_copy[:, 1], y_hat)
plt.show()
y_hat = x * result
print(np.corrcoef(y_hat.T, y))
