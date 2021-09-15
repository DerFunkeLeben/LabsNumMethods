import numpy
import math
import matplotlib.pyplot as plt


class Root:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        x_data_ = numpy.linspace(a, b, 100)
        self.max = max(df(x_data_))
        self.min = min(df(x_data_))
        self.alpha = 2 / (self.max + self.min)
        self.q = numpy.abs((self.max - self.min) / (self.max + self.min))
        self.epsilon1 = (1 - self.q) / self.q * epsilon


def f(x):
    return x**5 - 5.1 * x**4 + 9.6 * x**3 + 9.8 * x**2 - 8.8 * x - 5


def df(x):
    return 5 * x**4 - 5.1 * 4 * x**3 + 9.6 * 3 * x**2 + 9.8 * 2 * x - 8.8


def find_root(x):
    x_curr = x.a
    x_next = x_curr - x.alpha * f(x_curr)
    i = 0
    while (abs(x_next - x_curr) > x.epsilon1):
        i += 1
        x_curr = x_next
        x_next = x_curr - x.alpha * f(x_curr)
    return [x_next, i]


def transfer_axis(plt):
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


epsilon = math.pow(10, -8)
a0 = -1.
b0 = 1.

fig = plt.subplots()
x_data = numpy.linspace(a0, b0, 1000)
plt.plot(x_data, f(x_data))
plt.ylim([-10, 10])
transfer_axis(plt)
plt.show()

fig = plt.subplots()
plt.plot(x_data, df(x_data))
plt.ylim([-25, 25])
transfer_axis(plt)
plt.show()

x1 = Root(-1., -.75)
x2 = Root(-.5, -.25)
x3 = Root(.75, 1.)

print('x1 = ' + round(find_root(x1)[0], 8).__str__() + ' Iterations: ' + find_root(x1)[1].__str__())
print('x2 = ' + round(find_root(x2)[0], 8).__str__() + ' Iterations: ' + find_root(x2)[1].__str__())
print('x3 = ' + round(find_root(x3)[0], 8).__str__() + ' Iterations: ' + find_root(x3)[1].__str__())