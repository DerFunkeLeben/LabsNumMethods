import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Root:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.x0 = (a + b) / 2
        x_data_ = numpy.linspace(a, b, 100)
        self.max = max(df(x_data_))
        self.min = min(df(x_data_))
        self.alpha = 2 / (self.max + self.min)
        self.q = numpy.abs((self.max - self.min) / (self.max + self.min))
        self.epsilon1 = (1 - self.q) / self.q * epsilon


def f(x):
    return numpy.sqrt(x) * numpy.cos(x**2 / 3) - numpy.sin(x**3 / 3)


def df(x):
    return (-4 * x**2 * numpy.sin(x**2 / 3) + 3 * numpy.cos(x**2 / 3) - 6 * x**2 * numpy.sqrt(x) * numpy.cos(x**3 / 3)) / (6 * numpy.sqrt(x))


def newton_step(x):
    return x - f(x) / df(x)


def newton(x, n):
    j = 0
    r = []
    x_curr = x.x0
    x_next = newton_step(x_curr)
    while ((abs(x_next - x_curr) > epsilon) and (j < n)):
        r.append(abs(f(x_curr)))
        x_curr = x_next
        x_next = newton_step(x_curr)
        j += 1
    return [x_next, j, r]


def simple_iteration(x, n):
    j = 0
    r = []
    x_curr = x.x0
    x_next = x_curr - x.alpha * f(x_curr)
    while ((abs(x_next - x_curr) > x.epsilon1) and (j < n)):
        r.append(abs(f(x_curr)))
        j += 1
        x_curr = x_next
        x_next = x_curr - x.alpha * f(x_curr)
    return [x_next, j, r]


def transfer_axis(plt):
    ax = plt.gca()
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


epsilon = math.pow(10, -12)
a0 = 0.
b0 = 4.

fig = plt.subplots()
x_data = numpy.linspace(a0, b0, 1000)
plt.plot(x_data, f(x_data))
plt.ylim([-3, 3])
transfer_axis(plt)
plt.show()

roots = []
#roots.append(Root(-.1, .1))
roots.append(Root(1.25, 1.75))
roots.append(Root(2., 2.25))
roots.append(Root(2.4, 2.6))
roots.append(Root(3.5, 3.75))
roots.append(Root(3.85, 3.9))
roots.append(Root(3.94, 4.))

print('x = 0.0 Iterations: 0')
print()

for i in roots:
    result_1 = newton(i, 10)
    result_2 = simple_iteration(i, 10)
    a0 = 1
    b0 = result_1[2].__len__()
    fig, ax = plt.subplots()
    n_data = numpy.linspace(a0, b0, b0)
    plt.plot(n_data, result_1[2], label='Newton x=' + round(result_1[0], 12).__str__())

    b0 = result_2[2].__len__()
    n_data = numpy.linspace(a0, b0, b0)
    plt.plot(n_data, result_2[2], label='Simple it. x=' + round(result_2[0], 12).__str__())
    plt.yscale('log')
    plt.grid()
    plt.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

    print('Newton: x = ' + round(result_1[0], 12).__str__() + ' by ' + result_1[1].__str__() + ' iterations')
    print('Simple iteration: x = ' + round(result_2[0], 12).__str__() + ' by ' + result_2[1].__str__() + ' iterations')
    print()