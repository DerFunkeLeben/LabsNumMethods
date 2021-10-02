# %%
import numpy
import scipy
import math
from prettytable import PrettyTable
from sympy import *
import matplotlib.pyplot as plt


f = lambda x: math.sqrt(x) * (math.sin(3 * x)) ** 2
df_central = lambda x, h: (f(x + h) - f(x - h)) / 2 / h
df_right = lambda x, h: (f(x + h) - f(x)) / h
d2f_left = (
    lambda x, h: (2 * f(x) - 5 * f(x - h) + 4 * f(x - 2 * h) - f(x - 3 * h)) / h / h
)
df_error = lambda df_value, c, n: abs(df_value - df(c, n))


def df(t, n):
    x = Symbol("x")
    y = sqrt(x) * (sin(3 * x)) ** 2
    dx = diff(y, x, n)
    ddx = lambdify(x, dx)
    return ddx(t)


def count_arrays(c):
    h = []

    df_right_values = []
    df_central_values = []
    d2f_left_values = []

    df_right_errors = []
    df_central_errors = []
    d2f_left_errors = []

    for k in range(1, 16):
        h.append(10 ** (-k))

        df_right_values.append(df_right(c, h[k - 1]))
        df_central_values.append(df_central(c, h[k - 1]))
        d2f_left_values.append(d2f_left(c, h[k - 1]))

        df_right_errors.append(df_error(df_right_values[k - 1], c, 1))
        df_central_errors.append(df_error(df_central_values[k - 1], c, 1))
        d2f_left_errors.append(df_error(d2f_left_values[k - 1], c, 2))

    table = PrettyTable()
    table.add_column("h", h)
    table.add_column("Right derivative", df_right_values)
    table.add_column("Right der. error", df_right_errors)
    table.add_column("Central derivative", df_central_values)
    table.add_column("Central der. error", df_central_errors)
    table.add_column("Left second derivative", d2f_left_values)
    table.add_column("Left second der. error", d2f_left_errors)

    index_r = numpy.argmin(df_right_errors)
    index_c = numpy.argmin(df_central_errors)
    index_2l = numpy.argmin(d2f_left_errors)

    print(
        "\nRight derivative(optimal result): ",
        df_right_values[index_r],
        "\n\twith error ",
        df_right_errors[index_r],
        "\n\tgot with step h =",
        h[index_r],
    )

    print(
        "\nCentral derivative(optimal result): ",
        df_central_values[index_c],
        "\n\twith error ",
        df_central_errors[index_c],
        "\n\tgot with step h =",
        h[index_c],
    )

    print(
        "\nLeft second derivative(optimal result): ",
        d2f_left_values[index_2l],
        "\n\twith error ",
        d2f_left_errors[index_2l],
        "\n\tgot with step h =",
        h[index_2l],
    )

    fig, ax = plt.subplots()
    plt.plot(
        numpy.log10(h),
        numpy.log10(df_right_errors),
        "o-b",
        label="Right derivative",
        lw=2,
        mec="b",
        mew=2,
        ms=5,
    )
    plt.plot(
        numpy.log10(h),
        numpy.log10(d2f_left_errors),
        "o-g",
        label="Left second derivative",
        lw=2,
        mec="g",
        mew=2,
        ms=5,
    )
    plt.plot(
        numpy.log10(h),
        numpy.log10(df_central_errors),
        "o-r",
        label="Central derivative",
        lw=2,
        mec="r",
        mew=2,
        ms=5,
    )
    plt.legend()
    ax.set_xlabel('h')
    ax.set_ylabel('error')
    plt.show()

    return table


a = 0
b = 10
c = 7

print("\nАналитическое значение f'(%s) =" % (c), df(c, 1))
print("\nАналитическое значение f''(%s) =" % (c), df(c, 2))
print(count_arrays(c))


# %%
