# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

f = lambda x: -8 * x ** 2 + 5 * x + 1
q = lambda x: x / (x + 3)
k = lambda x: 9 if x <= x_jump else 13-2*x
u_test = lambda x: 6*x ** 2 if x <= x_jump else -10 * x ** 2 + 48 * x -32
# расчет числа точек N для шага h
get_n = lambda h: int(np.ceil((b - a) / h))


def q_i(x, h):
    definite_integral = lambda t: t - 3 * np.log(t + 3)
    q_right = definite_integral(x + h / 2)
    q_left = definite_integral(x - h / 2)
    return (q_right - q_left) / h


def f_i(x, h, mode):
    def definite_integral(t):
        if mode == "test":
            return -108*t if t <= x_jump else -40 * t ** 2 + 356 * t
        else:
            return -8 / 3 * t ** 3 + 5 / 2 * t ** 2 + t

    f_right = definite_integral(x + h / 2)
    f_left = definite_integral(x - h / 2)
    return (f_right - f_left) / h


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


def make_vector(x_data, h, mode):
    n = get_n(h)
    B = np.ones(n + 1) * h ** 2

    B[0] = u_a
    for i in range(1, n):
        B[i] *= f_i(x_data[i], h, mode)
    B[-1] = u_b

    return B


def solve_test(h):
    n = get_n(h)
    x_data = np.linspace(a, b, n + 1)
    A = make_matrix(x_data, h, mode="test")
    B = make_vector(x_data, h, mode="test")
    return LA.solve(A, B)


# построение графиков для тестового примера
def show_graph(h, u_data, label):
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


# проверка достижения заданной точности по праивлу Рунге
def Runge(y_data_h, y_data_2h):
    r_data = [0]
    for i in range(1, len(y_data_2h)):
        r_data.append(numpy.abs((y_data_h[2 * i] - y_data_2h[i])))
    return False if max(r_data) > epsilon else True

a = 1
x_jump = 2
b = 4
u_a = 6
u_b = 0
epsilon = 0.01
h = (b - a) / 20

u_data = solve_test(h)
show_graph(h, u_data, "test")


# %%
