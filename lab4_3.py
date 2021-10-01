# %%
import numpy as np
import math
import matplotlib.pyplot as plt


def f(x):
    return x*np.sin(x)


def s(x, n):
    sum = 0
    for i in range(n+1):
        sum += c[i] * np.power(x, i)
    return sum


def factor(n):
    c = [0]
    for i in range(1, n+1):
        c.append(np.round(np.sin((i-1)*np.pi/2) / math.factorial(i-1), 11))
    return c


def T(n, x):
    if n == 0:
        return 1
    if n == 1:
        return x
    return 2 * x * T(n - 1, x) - T(n - 2, x)


def eco(x, power):
    units = {
        '1': T(1, x),
        '2': 1 / 2 * (1 + T(2, x)),
        '3': 1 / 4 * (3 * x + T(3, x)),
        '4': 1 / 8 * (8 * x ** 2 - 1 + T(4, x)),
        '5': 1 / 16 * (20 * x ** 3 - 5 * x + T(5, x)),
        '6': 1 / 32 * (48 * x ** 4 - 18 * x ** 2 + 1 + T(6, x)),
        '7': 1 / 64 * (112 * x ** 5 - 56 * x ** 3 + 7 * x + T(7, x)),
        '8': 1 / 128 * (256 * x ** 6 - 160 * x ** 4 + 32 * x ** 2 - 1 + T(8, x)),
        '9': 1 / 256 * (576 * x ** 7 - 432 * x ** 5 + 120 * x ** 3 - 9 * x + T(9, x)),
        '10': 1 / 512 * (1280 * x ** 8 - 1120 * x ** 6 + 400 * x ** 4 - 50 * x ** 2 + 1 + T(10, x)),
        '11': 1 / 1024 * (2816 * x ** 9 - 2816 * x ** 7 + 1232 * x ** 5 - 220 * x ** 3 + 11 * x + T(11, x)),
        '12': 1 / 2048 * (6144 * x ** 10 - 6912 * x ** 8 + 3584 * x ** 6 - 840 * x ** 4 + 72 * x ** 2 - 1 + T(12, x))
    }
    return units[power.__str__()]


def eco_step(n):
    fig = plt.subplots()
    for i in range(100):
        s_data[i] = s(x[i], n) + s_data[i] - s(x[i], n + 2) + c[n + 2] * eco(x[i], n + 2) - c[n + 2] * T(n + 2, x[i]) / 2 ** (n + 1)

    plt.plot(x, np.abs(f(x) - s_data), label=f'$f(x)-s{n}(x)$')
    plt.legend()
    plt.show()


a = -1
b = 1
n = 12
c = factor(n)
print(c)

fig = plt.subplots()
x = np.linspace(a, b, 100)
s_data = np.zeros(100)
for i in range(100):
    s_data[i] = s(x[i], n)

plt.plot(x, np.abs(f(x) - s_data), label=f'$f(x)-s{n}(x)$')
plt.legend()
plt.show()


eco_step(10)
eco_step(8)
eco_step(6)

# %%
