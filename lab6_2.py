# %%
import numpy
import matplotlib.pyplot as plt

dy = lambda y, t: y / (t - 4) + (t - 4) / t
Euler = lambda y, t, h: y + h * dy(y, t)
Euler_modified = lambda y, t, h: y + h * dy(y + h / 2 * dy(y, t), t + h / 2)
get_n = lambda h: int(numpy.ceil((t_n - t_0) / h))


def get_y_data(h, Method):
    n = get_n(h)
    t_data = numpy.linspace(t_0, t_n, n + 1)
    y_data = [0]
    for i in range(1, len(t_data)):
        y_data.append(Method(y_data[i - 1], t_data[i - 1], h))
    return y_data


def Runge(y_data_h, y_data_2h):
    r_data = [0]
    for i in range(1, len(y_data_2h)):
        r_data.append(numpy.abs((y_data_h[2 * i] - y_data_2h[i])))
    return False if max(r_data) > epsilon else True


def solve(Method, methodName, plt):
    y_data_h = []
    h = 0.2

    while True:
        y_data_h = get_y_data(h, Method)
        y_data_2h = get_y_data(2 * h, Method)
        if Runge(y_data_h, y_data_2h):
            break
        h /= 2

    n = get_n(h)
    print("\n", methodName, ':')
    print("h = ", h)
    print("n = ", n)
    t_data = numpy.linspace(t_0, t_n, n + 1)
    plt.plot(t_data, y_data_h, label=methodName)


t_0 = 1
t_n = 1.8
epsilon = 10 ** (-6)
plt.subplots()

solve(Euler, "Euler", plt)
solve(Euler_modified, "Euler modified", plt)

plt.legend()
plt.show()

# %%
