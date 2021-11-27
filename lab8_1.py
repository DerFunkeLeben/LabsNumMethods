# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from sympy import DiracDelta, integrate
from sympy.core.symbol import Symbol

f = lambda x: -8 * x ** 2 + 5 * x + 1
q = lambda x: x / (x + 3)
k = lambda x: 1 + 2 * x if x <= x_jump else 3
k_definite_integral = lambda t: np.log(2 * t + 1) / 2 if t <= x_jump else t / 3

# точное решение для тестового примера
u_test = lambda x: x ** 2 + 8 if x <= x_jump else -1.75 * x ** 2 + 5.5 * x + 5.25

# расчет числа точек N для шага h
get_n = lambda h: int(np.ceil((b - a) / h))

# вычисление коэффициента q_i
def q_i(x, h):
    definite_integral = lambda t: t - 3 * np.log(t + 3)
    q_right = definite_integral(x + h / 2)
    q_left = definite_integral(x - h / 2)

    return (q_right - q_left) / h


# вычисление коэффициента f_i
def f_i(x, h, mode):
    # различные значения определенного интеграла,
    # в зависимости от задачи, которая решается
    # mode = test - тестовые пример
    # mode = main - исходная задача
    # mode = experiment_{a, b, c} - вычислительный эксперимент a, b и с
    def definite_integral(t):
        F = -8 / 3 * t ** 3 + 5 / 2 * t ** 2 + t
        if mode == "test":
            f_1 = -(4 * t ** 2) - (2 * t)
            f_2 = 10.5 * t
            return f_1 if t <= x_jump else f_2
        elif mode == "main":
            return F
        elif mode == "experiment_a":
            return F if t <= x_jump or t >= x_jump_2 else 0
        elif mode == "experiment_b":
            return 0 if t <= x_jump or t >= x_jump_2 else F
        else:
            return 0

    if mode == "experiment_c":
        c = -80  # мощность источника
        x0 = 2.4
        z = Symbol("z")
        f = c * integrate(DiracDelta(z - x0), (z, x - h / 2, x + h / 2))
        return f / h

    f_right = definite_integral(x + h / 2)
    f_left = definite_integral(x - h / 2)

    # если на отрезке [x[i-1/2];x[i+1/2]] есть точка разрыва функции,
    # необходимо обработать отдельно эти случай,
    # так как функция не интегрируема на этом отрезке
    if x + h / 2 >= x_jump > x - h / 2:
        f_middle = definite_integral(x_jump)
        return (f_middle - f_left) / h

    if mode == "experiment_a" or mode == "experiment_b":
        if x + h / 2 >= x_jump_2 > x - h / 2:
            f_middle = definite_integral(x_jump_2)
            return (f_middle - f_left) / h

    return (f_right - f_left) / h


# построение трехдиагональной матрицы для системы
def make_matrix(x_data, h, mode):
    n = get_n(h)
    A = np.eye(n + 1)
    for i in range(1, n):
        k_i_left_value = k(x_data[i] - h / 2)
        k_i_right_value = k(x_data[i] + h / 2)

        A[i, i - 1] = -k_i_left_value
        A[i, i] = k_i_left_value + k_i_right_value
        A[i, i] += q_i(x_data[i], h) * h ** 2 if mode != "test" else 0
        A[i, i + 1] = -k_i_right_value
    return A


# построение вектора правой части для системы
def make_vector(x_data, h, mode):
    n = get_n(h)
    B = np.ones(n + 1) * h ** 2

    B[0] = u_a
    for i in range(1, n):
        B[i] *= f_i(x_data[i], h, mode)
    B[-1] = u_b

    return B


# найти решение задачи
def solve(h, mode):
    n = get_n(h)
    x_data = np.linspace(a, b, n + 1)
    A = make_matrix(x_data, h, mode)
    B = make_vector(x_data, h, mode)
    return LA.solve(A, B)


# построение графиков для тестового примера
def show_graph_test(h, u_data):
    n = get_n(h)
    x_data = np.linspace(a, b, n + 1)
    u_test_data = [u_test(x_data[i]) for i in range(n + 1)]
    r_data = np.abs(u_test_data - u_data)

    plt.subplots()
    plt.plot(x_data, u_data, label="approx solution")
    plt.plot(x_data, u_test_data, label="exact solution")
    plt.legend()
    plt.show()

    plt.subplots()
    plt.plot(x_data, r_data, label="error")
    plt.legend()
    plt.show()


# построение графиков для исходной задачи
def show_graph_main(h, u_data_h, u_data_h2):
    u_data_h4 = solve(h / 4, "main")

    n_h = get_n(h)
    n_h2 = get_n(h / 2)

    x_data_h = np.linspace(a, b, n_h + 1)
    x_data_h2 = np.linspace(a, b, n_h2 + 1)

    r_data_h = Runge(u_data_h2, u_data_h)
    r_data_h2 = Runge(u_data_h4, u_data_h2)

    plt.subplots()
    plt.plot(x_data_h, u_data_h, label="solution h = 3/20")
    plt.plot(x_data_h2, u_data_h2, label="solution h = 3/40")
    plt.legend()
    plt.show()

    plt.subplots()
    plt.plot(x_data_h, r_data_h, label="error h = 3/20")
    plt.plot(x_data_h2, r_data_h2, label="error h = 3/40")
    plt.legend()
    plt.show()


# построение графиков для вычислительных экспериментов
def show_graph_experiment(h, u_data_a, u_data_b, u_data_c):
    n = get_n(h)
    x_data = np.linspace(a, b, n + 1)

    plt.subplots()
    plt.plot(x_data, u_data_a, label="experiment a")
    plt.plot(x_data, u_data_b, label="experiment b")
    plt.plot(x_data, u_data_c, label="experiment c")
    plt.legend()
    plt.show()


# погрешность решения по правилу Рунге
def Runge(y_data_h, y_data_2h):
    r_data = [0]
    for i in range(1, len(y_data_2h)):
        r_data.append(np.abs((y_data_h[2 * i] - y_data_2h[i])) / (2 ** 2 - 1))
    return r_data


a = 0
x_jump = 1
x_jump_2 = 2
b = 3
u_a = 8
u_b = 6
epsilon = 0.01
h = (b - a) / 20

u_data = solve(h, "test")
show_graph_test(h, u_data)

u_data = solve(h, "main")
u_data_2 = solve(h / 2, "main")
show_graph_main(h, u_data, u_data_2)

u_data_a = solve(h, "experiment_a")
u_data_b = solve(h, "experiment_b")
u_data_c = solve(h, "experiment_c")
show_graph_experiment(h, u_data_a, u_data_b, u_data_c)


# %%
