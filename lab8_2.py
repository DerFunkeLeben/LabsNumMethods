# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from sympy import DiracDelta, integrate
from sympy.core.symbol import Symbol

# расчет числа точек N для шага h
get_n = lambda h: int(np.ceil((b - a) / h))

# расчет размерности системы для шага h
get_dimensions = lambda h: (get_n(h) - 1) ** 2

# функция правой части
f = lambda x, y: -np.e ** -y * np.sin(np.pi * x) * (1 - np.pi ** 2)

# функция тестового примера
#u = lambda x, y: np.e ** -y * np.sin(np.pi * x)
u = lambda x, y: 5/17 * np.sin(np.pi * x * 4) * np.sin(np.pi * y) 

# получить значение на границе
def u_boundary(x, y, h):
    if y - h == a:
        return np.sin(np.pi * x)
    if y + h == b:
        return np.e ** -1 * np.sin(np.pi * x)
    return 0


# решение системы методом сопряженных градиентов
def solve_conjugate_gradient(A, b, epsilon=np.power(10.0, -9)):
    x = np.zeros(np.size(b))
    residual_0 = np.abs(np.dot(A, x) - b)
    descent_direction_n = residual_0
    residual_n = residual_0

    while LA.norm(residual_n) / LA.norm(residual_0) > epsilon:
        alpha = np.dot(residual_n, residual_n) / np.dot(
            np.dot(A, descent_direction_n), descent_direction_n
        )
        x = x + alpha * descent_direction_n
        residual_n_next = residual_n - alpha * np.dot(A, descent_direction_n)
        beta = np.dot(residual_n_next, residual_n_next) / np.dot(residual_n, residual_n)
        descent_direction_n = residual_n_next + beta * descent_direction_n
        residual_n = residual_n_next
    return x


# построение пятидиагональной матрицы для прямоугольной области
def make_matrix_test(h):
    n = get_n(h)
    dimension = get_dimensions(h)
    A = -4 * np.eye(dimension)
    for i in range(dimension):

        if i + n - 1 < dimension:
            A[i, i + n - 1] = 1
            A[i + n - 1, i] = 1

        if i % (n - 1) != 0:
            if i != 0:
                A[i, i - 1] = 1
            if i != dimension:
                A[i - 1, i] = 1

    return A


# построение вектора правой части для системы
def make_vector(h):
    n = get_n(h)
    B = []

    for i in range(1, n):
        for j in range(1, n):
            x = j * h
            y = i * h
            res = f(x, y) - u_boundary(x, y, h)
            B.append(res)

    B = [i * h ** 2 for i in B]
    return B


# найти решение задачи
def solve(h):
    dimension = int(np.sqrt(get_dimensions(h)))
    A = make_matrix_test(h)
    B = make_vector(h)
    solution = solve_conjugate_gradient(A, B)
    reshaped = np.reshape(solution, (dimension, dimension))
    u_data = add_boundary(reshaped)
    return u_data


def add_boundary(solution):
    dim = len(solution)
    x_data = np.linspace(a, b, dim + 2)

    left = right = np.zeros(dim).reshape(-1, 1)
    solution = np.concatenate((left, solution, right), axis=1)

    up = [u_boundary(x_data, a, 0)]
    down = [u_boundary(x_data, b, 0)]
    solution = np.concatenate((up, solution, down), axis=0)

    return solution


# построение графика
def show_graph_main(h, u_data):

    n_h = get_n(h)

    x = np.linspace(a, b, n_h + 1)
    y = np.linspace(a, b, n_h + 1)
    X, Y = np.meshgrid(x, y)
    Z = u(X, Y) if u_data == ["test"] else u_data

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(20, 40)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis")
    plt.show()


a = 0
b = 1
h = 1 / 20
n = get_n(h)

u_data_h = solve(h)
show_graph_main(h, u_data_h)

show_graph_main(h, ["test"])

# %%
