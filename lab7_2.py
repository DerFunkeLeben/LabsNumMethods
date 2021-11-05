# %%
import numpy
from numpy import linalg as LA
import matplotlib.pyplot as plt

# рассуждения на тему собственных значений матрицы, устойчивости решений, жесткости системы
def stiff_coef(matrix, matrixName):
    print("\nFor matrix " + matrixName + ":")

    l = LA.eigvals(matrix)
    re_l = numpy.real(l)
    s = max(numpy.abs(re_l)) / min(numpy.abs(re_l))
    h_euler_stiff = min(2 / numpy.abs(l))

    print("Eigenvalues are: ")
    for i in range(len(l)):
        print("l[", i, "] = ", numpy.round(l[i]))

    print("Stiff coefficient is: ", "%.2f" % s)
    if h <= h_euler_stiff:
        print("Explicit Euler method is stable")
        print("Modified Euler method is stable")
    else:
        print("Explicit Euler method is unstable")
        print("Modified Euler method is unstable")

    if s > 10:
        # для жесткой системы оцениваем шаг для неявного метода Эйлера, чтобы система уравнений была устойчивой
        print("Matrix " + matrixName + " is stiff")
        print(
            "\nh estimation for Explicit Euler method to make stiff linear system of equations stable is: ",
            "%.5f" % h_euler_stiff,
        )


# формула явного метода Эйлера
Euler = lambda y, matrix, h: y + h * matrix * y

# формула неявного метода Эйлера
def Euler_implicit(y, matrix, h=0.01):
    E = numpy.eye(3)
    invMatrix = LA.inv(E - h * matrix)
    return invMatrix * y

# задача Коши
dy = lambda y, matrix: numpy.array(matrix * y)

# формула усовершенствованного метода Эйлера
def Euler_modified(y, matrix, h):
    y_predict = y + h / 2 * dy(y, matrix)
    y_correct = y + h * dy(y_predict, matrix)
    return y_correct


# расчет числа точек N для шага h
get_n = lambda h: int(numpy.ceil((t_n - t_0) / h))

# получить массив решений выбранным методом с шагом h для трех компонент u, v, w
def get_y_data(h, Method, matrix, y0):
    n = get_n(h)
    y_data = [y0]
    for i in range(1, n):
        y_data.append(Method(y_data[i - 1], matrix, h))
    y_data = numpy.array(y_data)
    u, v, w = y_data[:, 0], y_data[:, 1], y_data[:, 2]
    return [u, v, w]


# построение графика для одной компоненты решения
def show_graph(data, title, comp_title):
    n = get_n(h)
    data = data[:, 0]
    t_data = numpy.linspace(t_0, t_n, n)
    fig, ax = plt.subplots()
    plt.plot(t_data, data, label=title)
    ax.set_title(comp_title)
    plt.legend()
    plt.show()


# построение графиков для двух методов для одной компоненты решения
def show_graph_2(data1, data2, title1, title2, comp_title, h_try):
    n1 = get_n(h_try)
    n2 = get_n(h)

    t_data1 = numpy.linspace(t_0, t_n, n1)
    t_data2 = numpy.linspace(t_0, t_n, n2)

    fig, ax = plt.subplots()
    plt.plot(t_data1, data1[:, 0], label=title1)
    plt.plot(t_data2, data2[:, 0], label=title2)
    ax.set_title(comp_title)
    plt.legend()
    plt.show()


# функция, осуществляющая:
# поиск решения явным методом Эйлера;
# поиск решения модифицированным методом Эйлера
def solve(matrix, y0, matrixName):
    print("\nFor matrix " + matrixName + ": ")
    u, v, w = get_y_data(h, Euler, matrix, y0)
    show_graph(u, "Euler", "u(t)")
    show_graph(v, "Euler", "v(t)")
    show_graph(w, "Euler", "w(t)")
    u, v, w = get_y_data(h, Euler_modified, matrix, y0)
    show_graph(u, "Euler modified", "u(t)")
    show_graph(v, "Euler modified", "v(t)")
    show_graph(w, "Euler modified", "w(t)")


# функция, осуществляющая:
# сравнение результатов явного и неявного метода Эйлера 
def solve_for_stiff(matrix, y0):
    print("\n\nFor stiff matrix: ")

    h_try = 10 ** (-6)
    print("h to try to match graphics is: ", h_try)
    u, v, w = get_y_data(h_try, Euler, matrix, y0)
    u_imp, v_imp, w_imp = get_y_data(h, Euler_implicit, matrix, y0)
    show_graph_2(u, u_imp, "Euler", "Euler implicit", "u(t)", h_try)
    show_graph_2(v, v_imp, "Euler", "Euler implicit", "v(t)", h_try)
    show_graph_2(w, w_imp, "Euler", "Euler implicit", "w(t)", h_try)


A = numpy.matrix(
    [
        [-116.967, 38.887, -110.397],
        [-52.101, -102.573, 275.23],
        [104.81, -277.406, -100.46],
    ]
)

Y0 = numpy.array([[4.4, 3.2, 5.2]]).transpose()

B = numpy.matrix(
    [[-47.173, 40.843, 27.459], [34.392, -81.93, 26.961], [35.24, 14.016, -86.897]]
)

Z0 = numpy.array([[4, 3.6, 4.8]]).transpose()

t_0 = 0
t_n = 1
h = 0.01

stiff_coef(A, "A")
stiff_coef(B, "B")
solve(A, Y0, "A")
solve(B, Z0, "B")
solve_for_stiff(B, Z0)


# %%
