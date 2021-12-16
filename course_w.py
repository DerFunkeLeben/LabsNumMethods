# %%
import numpy
import matplotlib.pyplot as plt

# класс входных данных
class InputData:
    def __init__(self, R0, L, C, f, k):
        self.L = L * 10 ** -6
        self.C = C * 10 ** -6
        self.R0 = R0
        self.f = f
        self.k = k
        self.U = 1
        self.w = 2 * numpy.pi * f
        self.period = 1 / f

    R = lambda self, i: self.R0 * (1 + self.k * i ** 2)


# задача Коши
def f(y, t):
    u, v = y
    # замена u = i
    # замена v = di/dt
    R = data.R(u)
    du = v
    dv = (-R * v - u / data.C + data.U * data.w * numpy.cos(data.w * t)) / data.L
    return numpy.array([du, dv], dtype=numpy.float64)


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
    for i in range(1, len(u_2h)):
        r_u_data.append(numpy.abs((u_h[2 * i] - u_2h[i])) / (2 ** 3 - 1))

    return max(r_u_data)


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

    y_data = numpy.array(y_data, dtype=numpy.float64)
    u, v = y_data[:, 0], y_data[:, 1]
    return numpy.array([u, v], dtype=numpy.float64)


# построение графика
def show_graph(h, y_data, label):
    fig, axs = plt.subplots()
    n = get_n(h)
    t_data = numpy.linspace(t_0, t_n, n)
    plt.plot(t_data, y_data, label="Adams Bashforth " + label)
    plt.legend()
    axs.spines["bottom"].set_position("center")
    plt.show()

# динамика поиска оптимального шага
def show_graph_h(h_data):
    fig, axs = plt.subplots()
    x_data = numpy.arange(len(h_data))
    plt.plot(x_data, numpy.log10(h_data), label="Динамика поиска оптимального шага")
    plt.legend()
    plt.show()


# адаптивная процедура поиска оптимального шага
# если точность не достигнута - уменьшаем шаг в 2 раза
# если точность превышает заданную на порядок - увеличиваем шаг в 1.5 раза
# иначе - нашли оптимальный шаг
def get_optimal_h(h, y0):
    y_data_h = get_y_data(h, y0)
    y_data_2h = get_y_data(2 * h, y0)
    err = Runge(y_data_h, y_data_2h)
    h_dynamics.append(h)
    print(
        "Error:",
        "{:.2e}".format(err),
        "\nStep: ",
        "{:.2e}".format(h),
    )
    if err > EPSILON:
        print("Decreasing...\n")
        return get_optimal_h(h / 2, y0)
    elif err < EPSILON / 10:
        print("Increasing...\n")
        return get_optimal_h(1.5 * h, y0)
    else:
        show_graph(h, y_data_h[0], "i(t)")
        show_graph_h(h_dynamics)
        return h, err


# все варианты входных данных
allData = {
    1: InputData(R0=2, L=1, C=0.001, f=10 ** 6, k=8 * 10 ** 10),  # 60sec
    2: InputData(R0=3, L=5, C=0.01, f=5 * 10 ** 6, k=2 * 10 ** 14),  # 14sec
    3: InputData(R0=5, L=10, C=0.1, f=10 ** 5, k=10 ** 14),  # 120sec
    4: InputData(R0=3, L=50, C=0.047, f=2 * 10 ** 5, k=5 * 10 ** 10),  # 30sec
    5: InputData(R0=7.5, L=2, C=0.0047, f=2 * 10 ** 6, k=4 * 10 ** 15),  # 120sec
    6: InputData(R0=1, L=10, C=0.068, f=6 * 10 ** 4, k=7 * 10 ** 12),  # 60sec
}
data = allData[2]  # выбираем вариант

EPSILON = 10 ** -8
y0 = numpy.array([0, 0], dtype=numpy.float64)
t_0 = 0
t_n = data.period * 2  # охватываем два периода колебаний
h = (t_n - t_0) / 100
h_dynamics = []

h_opt, err = get_optimal_h(h, y0)
print("Оптимальный шаг: ", "{:.2e}".format(h_opt))
print("Погрешность: ", "{:.2e}".format(err))

# %%
