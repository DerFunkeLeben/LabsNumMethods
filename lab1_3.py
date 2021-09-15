import numpy
import math


class Number:
    def __init__(self, value, abs_error):
        self.value = value
        self.abs_error = abs_error
        self.rel_error = abs_error / numpy.abs(value)


def f(a, b, c):
    return (a.value**2 + b.value**2) / (a.value-c.value)


def f_abs_error_manually(a, b, c):
    return f_rel_error_manually(a, b, c) * f(a, b, c)


def f_rel_error_manually(a, b, c):
    a_pow_2_error = a.rel_error * 2
    b_pow_2_error = b.rel_error * 2
    numerator_error = ((a.value**2 * a_pow_2_error) + (b.value**2 * b_pow_2_error)) / (a.value**2 + b.value**2)
    denominator_error = ((a.value * a.rel_error) + (c.value * c.rel_error)) / (a.value - c.value)
    return numerator_error + denominator_error


def df_da(a, b, c):
    return (a.value**2 - 2 * a.value * c.value - b.value**2) / (a.value - c.value)**2


def df_db(a, b, c):
    return 2 * b.value / (a.value - c.value)


def df_dc(a, b, c):
    return (a.value**2 + b.value**2) / (a.value-c.value)**2


def f_abs_error_auto(a, b, c):
    a_ = numpy.abs(df_da(a, b, c)) * a.abs_error
    b_ = numpy.abs(df_db(a, b, c)) * b.abs_error
    c_ = numpy.abs(df_dc(a, b, c)) * c.abs_error
    return a_ + b_ + c_


def f_rel_error_auto(a, b, c):
    return f_abs_error_auto(a, b, c) / numpy.abs(f(a, b, c))


def first_digit(x):
   for i in str(x):
       if i != '0' and i != '.':
           return int(i)


def true_digits(x, rel_er):
    return int(round(-math.log10(first_digit(x) * rel_er) + 1))


A = Number(25.18, .01)
B = Number(24.98, .01)
C = Number(23.18, .01)

print('Counted manually: ' +
      str(f(A, B, C)) +
      ' ± ' +
      str(round(f_abs_error_manually(A, B, C), 4)) +
      ' ;  \u03B4(f) = ' +
      str(round(f_rel_error_manually(A, B, C) * 100, 4)) +
      '%')
print('Number of true digits: ' + true_digits(f(A, B, C), f_rel_error_manually(A, B, C)).__str__())
print()

print('Counted with formula: ' +
      str(f(A, B, C)) +
      ' ± ' +
      str(round(f_abs_error_auto(A, B, C), 4)) +
      ' ;  \u03B4(f) = ' +
      str(round(f_rel_error_auto(A, B, C) * 100, 4)) +
      '%')
print('Number of true digits: ' + true_digits(f(A, B, C), f_rel_error_auto(A, B, C)).__str__())