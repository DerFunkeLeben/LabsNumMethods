# %%
import numpy
import matplotlib.pyplot as plt

# задача Коши
def dy(y, t):
    u, v = y
    du = t * (u ** 2) * numpy.sqrt(v)
    dv = numpy.sqrt(u) + numpy.sqrt(v)
    return [du, dv]

# формула метода Эйлера
def Euler(y, t, h):
    u, v = y
    y0, y1 = dy(y, t)
    euler1, euler2 = u + h * y0, v + h * y1
    return [euler1, euler2]

# формула усовершенствованного метода Эйлера
def Euler_modified(y, t, h):
    u, v = y
    y0, y1 = dy(y, t)
    y_support0, y_support1 = dy([u + h / 2 * y0, v + h / 2 * y1], t + h / 2)
    eulerMod1, eulerMod2 = u + h * y_support0, v + h * y_support1
    return [eulerMod1, eulerMod2]

# расчет числа точек N для шага h
get_n = lambda h: int(numpy.ceil((t_n - t_0) / h))

# получить массив решений выбранным методом с шагом h для двух компонент u и v
def get_y_data(h, Method):
    n = get_n(h)
    t_data = numpy.linspace(t_0, t_n, n + 1)
    y_data = [[1, 1]]
    for i in range(1, len(t_data)):
        y_data.append(Method(y_data[i - 1], t_data[i - 1], h))
    y_data = numpy.array(y_data)
    u, v = y_data[:, 0], y_data[:, 1]
    return [u, v]


# максимальное значение погрешности по правилу Рунге
def Runge(y_data_h, y_data_2h):
    r_data = [0]
    for i in range(1, len(y_data_2h)):
        r_data.append(numpy.abs((y_data_h[2 * i] - y_data_2h[i])))
    return max(r_data)

# функция, осуществляющая: поиск решения выбранным методом;
# оценку погрешности по правилу Рунге
def solve(Method, methodName):
    u_h, v_h = get_y_data(h, Method)
    u_2h, v_2h = get_y_data(2 * h, Method)

    max_r_u = Runge(u_h, u_2h)
    max_r_v = Runge(v_h, v_2h)

    print("\n", methodName, ":")
    print("Max Runge error for u(t): ", max_r_u)
    print("Max Runge error for v(t): ", max_r_v)
    return [u_h, v_h]

# построение графиков для двух методов одной компоненты решения
def show_graph(data_euler, data_eulerMod, title):
    n = get_n(h)
    t_data = numpy.linspace(t_0, t_n, n + 1)
    fig, ax = plt.subplots()
    plt.plot(t_data, data_euler, label="Euler")
    plt.plot(t_data, data_eulerMod, label="Euler modified")
    ax.set_title(title)
    plt.legend()
    plt.show()


t_0 = 0
t_n = 1
h = 0.01

u_euler, v_euler = solve(Euler, "Euler")
u_eulerMod, v_eulerMod = solve(Euler_modified, "Euler modified")

show_graph(u_euler, u_eulerMod, "u(t)")
show_graph(v_euler, v_eulerMod, "v(t)")

# %%
