# %%
import numpy
import matplotlib.pyplot as plt

R_ = lambda R0, k, i: R0 * (1 + k * i ** 2)

# задача Коши
def f(y, t):
    u, v = y
    # замена u = i
    # замена v = di/dt
    R = R_(R0, k, u)
    du = v
    dv = (-R * v - u / C + U * w * numpy.cos(w * t)) / L
    return numpy.array([du, dv])


# метод Рунге-Кутты 3 порядка
def Runge_Kutta_3(y, t, h):
    k1 = f(y, t) * h
    k2 = f(y + k1 / 2, t + h / 2) * h
    k3 = f(y - k1 + 2 * k2, t + h) * h
    y = y + (k1 + 4 * k2 + k3) / 6
    t = t + h
    return y


# метод Адамса-Башфорта 3 порядка
def Adams_Bashforth(y, t, i, h):
    f_1 = f(y[i - 1], t[i - 1])
    f_2 = f(y[i - 2], t[i - 2])
    f_3 = f(y[i - 3], t[i - 3])
    return y[i - 1] + h / 12 * (23 * f_1 - 16 * f_2 + 5 * f_3)


# расчет числа точек N для шага h
get_n = lambda h: int(numpy.ceil((t_n - t_0) / h))

# максимальное значение погрешности по правилу Рунге
def Runge(y_data_h, y_data_2h):
    u_h, v_h = y_data_h
    u_2h, v_2h = y_data_2h
    r_u_data = [0]
    r_v_data = [0]
    for i in range(1, len(u_2h)):
        r_u_data.append(numpy.abs((u_h[2 * i] - u_2h[i])) / (2 ** 3 - 1))

    for i in range(1, len(v_2h)):
        r_v_data.append(numpy.abs((v_h[2 * i] - v_2h[i])) / (2 ** 3 - 1))

    return max(max(r_u_data), max(r_v_data))


# получить массив решений по методу Адамса-Башфорта 3 порядка
# с помощью стартовых решений по методу Рунге-Кутты 3 порядка
def get_y_data(h, y0):
    n = get_n(h)
    t_data = numpy.linspace(t_0, t_n, n)
    y_data = [y0]
    y_data.append(Runge_Kutta_3(y_data[0], t_0, h))
    y_data.append(Runge_Kutta_3(y_data[1], t_0 + h, h))

    for i in range(3, n):
        y_data.append(Adams_Bashforth(y_data, t_data, i, h))

    y_data = numpy.array(y_data)
    u, v = y_data[:, 0], y_data[:, 1]
    return numpy.array([u, v])


# построение графика
def show_graph(h, y_data, label):
    fig, axs = plt.subplots()
    n = get_n(h)
    t_data = numpy.linspace(t_0, t_n, n)
    plt.plot(t_data, y_data, label="Adams Bashforth " + label)
    plt.legend()
    # axs.spines['bottom'].set_position('center')
    plt.show()


# адаптивная процедура поиска оптимального шага
# если точность не достигнута - уменьшаем шаг в 2 раза
# если точность превышает заданную на порядок - увеличиваем шаг в 1.5 раза
# иначе - нашли оптимальный шаг
def get_optimal_h(h, y0):
    y_data_h = get_y_data(h, y0)
    y_data_2h = get_y_data(2 * h, y0)
    err = Runge(y_data_h, y_data_2h)
    if err > epsilon:
        return get_optimal_h(h / 2, y0)
    elif err < epsilon / 10:
        return get_optimal_h(1.5 * h, y0)
    else:
        return h


# исходные данные
L = 50
R0 = 3
C = 0.047
f_ = 2 * 10 ** 5
k = 5 * 10 ** 10
w = 2 * numpy.pi * f_
U = 1

epsilon = 10 ** -9
t_0 = 0
t_n = 1
h = 0.01
y0 = numpy.array([0, 0])


y_data_h = get_y_data(h, y0)
y_data_2h = get_y_data(2 * h, y0)

show_graph(h, y_data_h[0], "i(t)")

# h_opt = get_optimal_h(h, y0)
# print("Оптимальный шаг: ", h_opt)


# %%
