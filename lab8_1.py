# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

f = lambda x: -8 * x ** 2 + 5 * x + 1
q = lambda x: x / (x + 3)
k = lambda x: 1 + 2 * x if x <= x_jump else 3
u_test = lambda x: x ** 2 + 8 if x <= x_jump else -1.75 * x ** 2 + 5.5 * x + 5.25
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
            return -(4 * t ** 2) - (2 * t) if t <= x_jump else 5.25 * t ** 2
        else:
            return -8 / 3 * t ** 3 + 5 / 2 * t ** 2 + t

    f_right = definite_integral(x + h / 2)
    f_left = definite_integral(x - h / 2)
    return (f_right - f_left) / h


def make_matrix(x_data, h, mode):
    n = get_n(h)
    A = np.eye(n)
    for i in range(1, n - 1):
        k_i_left_value = k(x_data[i] - h / 2)
        k_i_right_value = k(x_data[i] + h / 2)

        A[i, i - 1] = -k_i_left_value
        A[i, i] = k_i_left_value + k_i_right_value
        A[i, i] += q_i(x_data[i], h) * h ** 2 if mode != "test" else 0
        A[i, i + 1] = -k_i_right_value
    return A


def make_vector(x_data, h, mode):
    n = get_n(h)
    B = np.ones(n) * h ** 2

    B[0] = u_a
    for i in range(1, n - 1):
        B[i] *= f_i(x_data[i], h, mode)
    B[-1] = u_b

    return B


def solve_test(h):
    n = get_n(h)
    x_data = np.linspace(a, b, n + 1)
    A = make_matrix(x_data, h, mode="test")
    B = make_vector(x_data, h, mode="test")
    return LA.solve(A, B)


# построение графика
def show_graph(h, u_data, label):
    fig, axs = plt.subplots()
    n = get_n(h)
    x_data = np.linspace(a, b, n)

    u_test_data = [u_test(x_data[i]) for i in range(n)]
    plt.plot(x_data, u_data, label="approx solution")
    plt.plot(x_data, u_test_data, label="exact solution")
    plt.legend()
    # axs.spines['bottom'].set_position('center')
    plt.show()


a = 0
x_jump = 1
b = 3
u_a = 8
u_b = 6
epsilon = 0.01
h = (b - a) / 20

u_data = solve_test(h)
show_graph(h, u_data, "test")


# %%
