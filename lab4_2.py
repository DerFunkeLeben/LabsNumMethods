import numpy as np
import math
import matplotlib.pyplot as plt


def f(x):
    return np.sin(np.power(x, 2))/x


def r_p(t):
    return np.abs(f(t) - Neuton(t))


def r_l(t):
    return np.abs(f(t) - Lagrange(t))


def Lagrange(value):
    lagrange = 0
    for i in range(n + 1):
        mul = 1
        for j in range(n + 1):
            if i != j:
                mul *= (value - x[j]) / (x[i] - x[j])
        lagrange += y[i] * mul
    return lagrange


def delta_y():
    difs = list()
    difs.append(y)
    for i in range(n):
        row = []
        for j in range(0, len(difs[i])-1):
            row.append(difs[i][j+1] - difs[i][j])
        difs.append(list(row))
    return difs


def Neuton(value):
    index, nearest = min(enumerate([a, b]), key=lambda arr: abs(value - arr[1]))
    q = (value - nearest) / step
    diffs = delta_y()
    P = 0
    for i in range(n):
        mul = 1
        for j in range(i):
            mul *= (q-j) if index == 0 else (q+j)
        if index != 0:
            index = len(diffs[i])-1
        mul *= diffs[i][index] / math.factorial(i)
        P += mul
    return P


a = 1
b = 4
epsilon = 0.001
n = 20
step = (b - a) / n

x = list(np.arange(a, b+0.1*step, step))
y = list(f(x))

fig = plt.subplots()
x_data = np.linspace(a, b, 100)
y_data_l = np.zeros(100)
y_data_p = np.zeros(100)
for i in range(100):
    y_data_l[i] = r_l(x_data[i])
    y_data_p[i] = r_p(x_data[i])

plt.plot(x_data, y_data_l, label=f'$RL(x)$')
plt.plot(x_data, y_data_p, label=f'$RP(x)$')
plt.legend()
plt.show()

fig = plt.subplots()
for i in range(100):
    y_data_l[i] = Lagrange(x_data[i])
    y_data_p[i] = Neuton(x_data[i])

plt.plot(x_data, y_data_l, label=f'$L(x)$')
plt.plot(x_data, y_data_p, label=f'$P(x)$')
plt.plot(x_data, f(x_data), label=f'$f(x)$')
plt.legend()
plt.show()




