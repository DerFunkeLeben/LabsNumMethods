import numpy
import scipy
from scipy import integrate
from sympy import *


def f(x):
    return sqrt(x) * (sin(3 * x)) ** 2


def Runge(h):
    return (integral_miln(h / 2) - integral_miln(h)) / (2 ** 6 - 1)


def integral_miln(h):
    n = int(numpy.ceil((b - a) / h))
    h = (b - a) / n
    x = []
    for x_i in numpy.arange(a, b + h * 0.01, h):
        x.append(x_i)

    result = 0
    for i in range(1, n + 1):
        result += 7 * f(x[i - 1])\
                  + 32 * f(x[i - 1] + h / 4)\
                  + 12 * f(x[i - 1] + h / 2)\
                  + 32 * f(x[i] - h / 4)\
                  + 7 * f(x[i])
    return result * h / 90


def miln():
    h = (b - a) / 2
    while abs(Runge(h)) > epsilon:
        h /= 2
    h = h / 2

    n = int(numpy.ceil((b - a) / h))
    I_h = integral_miln(h)
    I_h_2 = integral_miln(h / 2)
    I_clarify = I_h_2 + Runge(h)

    print('Число разбиений: n =', n)
    print('Интеграл с шагом h: I =', I_h)
    print('Интеграл шаг h / 2: I =', I_h_2)
    print('Погрешность по Рунге с шагом h: R =', abs(I_h - integral))
    print('Погрешность по Рунге шаг h / 2: R =', abs(I_h_2 - integral))
    print('Уточнение по Рунге: I_уточн =', I_clarify)
    print('Погрешность R_уточн =', abs(I_clarify - integral))


a = 0
b = 10
epsilon = 10 ** (-12)
integral = scipy.integrate.quad(f, a, b, epsabs=epsilon, epsrel=epsilon)[0]

print('\nАналитическое значение интеграла: I =', integral, '\n')
miln()
