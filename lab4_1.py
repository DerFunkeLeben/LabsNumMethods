import numpy as np


def create_factor_mnk(n, data):
    A = []
    m = len(x)
    for i in range(n):
        row = []
        for j in range(n):
            s = 0
            for k in range(m):
                s += np.power(data[k], i + j)
            row.append(s)
        A.append(np.array(row))
    return np.array(A)


def create_b_mnk(n, x, y):
    b = []
    m = len(x)
    for j in range(n):
        s = 0
        for k in range(m):
            s += np.power(x[k], j) * y[k]
        b.append(s)
    return np.array(b)


def least_squares(n, x, y, value):
    A = create_factor_mnk(n+1, x)
    b = create_b_mnk(n+1, x, y)
    factor = np.linalg.solve(A, b)
    sig = sigma(x, y, factor)
    return polynomial(factor, value), sig


def interpolation_lagrange(n, x, y, value):
    L = 0
    for i in range(n+1):
        mul = 1
        for j in range(n+1):
            if i != j:
                mul *= (value - x[j]) / (x[i] - x[j])
        L += y[i] * mul
    return L


def sigma(x, y, factor):
    s = 0
    for k in range(len(x)):
        s += (y[k]-polynomial(factor, x[k]))**2
    return np.sqrt(1/len(x) * s)


def polynomial(factor, value):
    s = 0
    for k in range(len(factor)):
        s += np.power(value, k) * factor[k]
    return s


def P(power):
    value, deviation = least_squares(power, x, y, 2.019)
    print('Least squares method, P' + power.__str__() + '(2019)', '=', round(value, 4))
    print('Standard deviation: ', round(deviation, 4))
    print()


x = list(np.arange(1.950, 2.021, 0.01))
y = list([4.5, 5.2, 6.2, 8.1, 11, 14.8, 23.36, 32.52])


angola_2019 = 31.83
interpolation = interpolation_lagrange(2, x[5:], y[5:], 2.019)

P(1)
P(2)
P(5)

print('Lagrange interpolation method: ', round(interpolation, 4))
print()
print('Real data of 2019: ', angola_2019)
