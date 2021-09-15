import numpy
import math
import matplotlib.pyplot as plt


class Root:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.x0 = (a + b) / 2


def f(x):
    return (numpy.cos(x))**2 + 2/35*numpy.cos(x)-numpy.exp(-x)


def f1(x):
    return (numpy.cos(x))**2 + 2/35*numpy.cos(x)+2


def g(x):
    return 2+numpy.exp(-x)


def df(x):
    return numpy.exp(-x)-numpy.sin(x)*(2*numpy.cos(x)+2/35)


def newton_step(x):
    return x - f(x) / df(x)


def newton(x):
    x_curr = x.x0
    x_next = newton_step(x_curr)
    while ((abs(x_next - x_curr) > epsilon)):
        x_curr = x_next
        x_next = newton_step(x_curr)
    return x_next


epsilon = math.pow(10, -5)
a0 = -1.
b0 = 10.

fig = plt.subplots()
x_data = numpy.linspace(a0, b0, 1000)
plt.plot(x_data, f(x_data))
plt.ylim([-.25, .25])
plt.show()

roots = []
roots.append(Root(-.5, .3))
roots.append(Root(.75, 1.2))
roots.append(Root(1.8, 2.2))
roots.append(Root(4.5, 5.5))
roots.append(Root(7.5, 8.5))

print('Roots:')
for i in roots:
    result = newton(i)
    print('x = ' + round(result, 5).__str__())
    print('y = ' + round(g(result), 5).__str__())
    print()




