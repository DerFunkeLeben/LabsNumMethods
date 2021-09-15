import numpy
import math
import matplotlib.pyplot as plt


class Root:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.x0 = (a + b) / 2


def f(x):
    return 800 * numpy.arctan((4 * x - 11) / (5 * x + 21)) - 168 * x + 16 * x**2 + 341


def df(x):
    return 8 * (164 * x**3 - 373 * x**2 - 314 * x + 2098) / (41 * x**2 + 122 * x + 562)


def newton_step(x, m):
    return x - m * f(x) / df(x)


def newton(x, m, n=50):
    j = 0
    x_curr = x.x0
    x_next = newton_step(x_curr, m)
    while ((abs(x_next - x_curr) > epsilon) and (j < n)):
        x_curr = x_next
        x_next = newton_step(x_curr, m)
        j += 1
    return [x_next, j]


def transfer_axis(plt):
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


epsilon = math.pow(10, -12)
a0 = -6.
b0 = 6.

fig = plt.subplots()
x_data = numpy.linspace(a0, b0, 1000)
plt.plot(x_data, f(x_data))
plt.ylim([-100, 100])
transfer_axis(plt)
plt.show()

x1 = Root(-4., -3.)
x2 = Root(2., 3.)

result = newton(x1, 1)
print('x1 = ' + round(result[0], 12).__str__() + ' by ' + result[1].__str__() + ' iterations. Multiplicity: 1')
result = newton(x2, 1)
print('x2 = ' + round(result[0], 12).__str__() + ' by ' + result[1].__str__() + ' iterations. Multiplicity: 1')