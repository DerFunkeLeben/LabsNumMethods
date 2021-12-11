# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from prettytable import PrettyTable

# расчет числа точек N для шага h
get_n = lambda h: int(np.ceil((b - a) / h))

# расчет размерности системы для шага h
get_dimensions = lambda h: (get_n(h) - 1) ** 2

# функция правой части
def f(x, y, mode):
    if mode == "test_1":
        return -5 * np.pi ** 2 * np.sin(np.pi * x * 4) * np.sin(np.pi * y)
    if mode == "test_2":
        return np.e ** -y * np.sin(np.pi * x) * (1 - np.pi ** 2)
    return 0


# решение тестового примера
def u(x, y, mode):
    if mode == "test_1":
        return 5 / 17 * np.sin(np.pi * x * 4) * np.sin(np.pi * y)
    if mode == "test_2":
        return np.e ** -y * np.sin(np.pi * x)
    return 0


# значения на границе
def u_boundary(x, y, mode, h=0):
    if mode == "test_1":
        return 0
    if mode == "test_2":
        if y - h == a:
            return np.sin(np.pi * x)
        if y + h == b:
            return np.e ** -1 * np.sin(np.pi * x)
    return 0


# построение пятидиагональной матрицы для прямоугольной области
def make_matrix_test(h):
    n = get_n(h)
    dimension = get_dimensions(h)
    A = -2 * np.eye(dimension)
    for i in range(dimension):
        if i + n - 1 < dimension:
            A[i + n - 1, i] = 1

        if i % (n - 1) != 0 and i != 0:
            A[i, i - 1] = 1
    return A + A.T


# построение вектора правой части для системы
def make_vector(h, mode):
    n = get_n(h)
    B = []

    for i in range(1, n):
        for j in range(1, n):
            x = j * h
            y = i * h
            res = f(x, y, mode) * h ** 2 - u_boundary(x, y, mode, h)
            B.append(res)

    return B


# метод Зейделя
def Gauss_Seidel(A, B):
    n = len(A)
    x = np.ones(n)
    err = 1
    while err >= epsilon:
        x_new = np.copy(x)
        for i in range(len(A)):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (B[i] - s1 - s2) / A[i][i]
        err = sum(abs(x_new[i] - x[i]) for i in range(n))
        x = x_new
    return x


# найти решение задачи
def solve(h, mode):
    dimension = int(np.sqrt(get_dimensions(h)))

    A = make_matrix_test(h)
    B = make_vector(h, mode)

    solution = LA.solve(A, B)
    # solution = Gauss_Seidel(A, B)

    reshaped = np.reshape(solution, (dimension, dimension))
    u_data = add_boundary(reshaped, mode)
    return u_data


# вырезать лишнюю область из решения
def cut_area(u_data):
    n = len(u_data)
    sq_1 = n / 3 - 1
    sq_2 = n - sq_1 - 1

    for i in range(n):
        for j in range(n):
            if 0 <= i <= sq_1 and sq_2 <= j <= n:
                u_data[i, j] = 0
            if sq_1 < i <= n and 0 <= j <= sq_1:
                u_data[i, j] = 0
            if sq_2 <= i <= n and 0 <= j < sq_2:
                u_data[i, j] = 0

    return u_data


# добавление к матрице решений на границах
def add_boundary(solution, mode):
    dim = len(solution) + 2
    x = np.linspace(a, b, dim)

    # четыре вектора - решения на границе
    left = np.array([u_boundary(a, x[i], mode) for i in range(dim)]).reshape(-1, 1)
    right = np.array([u_boundary(b, x[i], mode) for i in range(dim)]).reshape(-1, 1)
    up = [[u_boundary(x[i], a, mode) for i in range(dim)]]
    down = [[u_boundary(x[i], b, mode) for i in range(dim)]]

    left = left[1:-1, :]
    right = right[1:-1, :]

    solution = np.concatenate((left, solution, right), axis=1)
    solution = np.concatenate((up, solution, down), axis=0)

    return solution


# построение графика
def show_graph(h, u_data, mode, label):

    n_h = get_n(h)

    x = np.linspace(a, b, n_h + 1)
    y = np.linspace(a, b, n_h + 1)
    X, Y = np.meshgrid(x, y)

    Z = []
    if label == "Exact solution":
        Z = u(X, Y, mode)
    elif label == "Approx solution":
        Z = u_data
    elif label == "Error":
        Z = np.abs(u_data - u(X, Y, mode))
    elif label == "Error complex area":
        Z = u_data

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title(label + " " + mode)
    ax.view_init(60, 70)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="inferno")
    # inferno # twilight_shifted
    plt.show()


# вывод таблицы результатов
def print_table(u_test):
    table = PrettyTable()

    field_names = np.array(["x[" + str(i) + "]" for i in range(len(u_test))])
    field_names = np.concatenate((["u(x, y)"], field_names), axis=0)
    table.field_names = field_names

    for row in range(len(u_test)):
        label = "y[" + str(row) + "]"
        data = np.round(u_test[row], 3)
        table.add_row([label, *data])
    print(table)


# погрешность решения по правилу Рунге
def Runge(y_h, y_2h, h):
    n = len(y_2h)
    r_data = np.zeros(n)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            r_data = np.append(
                r_data, np.abs((y_h[2 * i, 2 * j] - y_2h[i, j])) / (2 ** 2 - 1)
            )
        r_data = np.append(r_data, [0, 0])
    r_data = np.append(r_data, np.zeros(n))

    if max(r_data) > epsilon:
        h *= 3 / 4
        u_main = cut_area(solve(h, "test_2"))
        u_main_runge = cut_area(solve(h / 2, "test_2"))
        return Runge(u_main_runge, u_main, h)

    show_graph(h, y_2h, "test_2", "Approx solution")
    return np.reshape(r_data, (n, n)), h


a = 0
b = 1
h = 1 / 4
epsilon = 0.001


# u_test_1 = solve(h, "test_1")
# show_graph(h, 0, "test_1", "Exact solution")
# show_graph(h, u_test_1, "test_1", "Approx solution")
# show_graph(h, u_test_1, "test_1", "Error")
# print_table(u_test_1)

u_test_2 = solve(h, "test_2")
show_graph(h, 0, "test_2", "Exact solution")
show_graph(h, u_test_2, "test_2", "Approx solution")
show_graph(h, u_test_2, "test_2", "Error")
print_table(u_test_2)


u_main = cut_area(u_test_2)
u_main_runge = cut_area(solve(h / 2, "test_2"))
r_main, h = Runge(u_main_runge, u_main, h)
show_graph(h, r_main, "test_2", "Error complex area")
print_table(u_main)


# %%
