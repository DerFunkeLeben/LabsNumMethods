import numpy
import scipy
from scipy import optimize


def f(x):
    result = 0
    for i in range(len(coefs)):
        result += coefs[i] * x**i
    return result


# производная функции в точке x

def df_dx(x):
    result = 0
    for i in range(1, len(coefs)):
        result += coefs[i] * i * x ** (i - 1)
    return result


# нахождение интеграла по формуле левых прямоугольников с заданным шагом h и числом разбиений n

def integral_left_rect(h, n):
    f_i = []
    for x in numpy.arange(a, b, h):
        f_i.append(f(x))
    result = 0
    for i in range(n - 1):
        result += f_i[i]
    return result * h


# нахождение интеграла по формуле Милна с заданным шагом h и числом разбиений n

def integral_miln(h, n):
    # берем только граничные точки
    x = [a, b]
    result = 0
    for i in range(1, n + 1):
        result += 7 * f(x[i - 1])\
                  + 32 * f(x[i - 1] + h / 4)\
                  + 12 * f(x[i - 1] + h / 2)\
                  + 32 * f(x[i] - h / 4)\
                  + 7 * f(x[i])
    return result * h / 90


# функция, выводящая результаты, относящиеся к формуле левых прямоугольников

def left_rect():
    # максимум производной на [a, b]
    M1 = -scipy.optimize.minimize_scalar(lambda x: -numpy.abs(df_dx(x)), bounds=[a, b], method='bounded').fun
    # оценка шага h
    h_estimate = 2 * epsilon / (M1 * (b - a))
    # число разбиений n - округляем в большую сторону
    n = int(numpy.ceil((b - a) / h_estimate))
    h = (b - a) / n
    I = integral_left_rect(h, n)
    print('------МЕТОД ЛЕВЫХ ПРЯМОУГОЛЬНИКОВ------')
    print('n =', n)
    print('h =', "{:.3f}".format(h * 10 ** 6), '* 10^-6')
    print('I =', I)
    print('R =', numpy.abs(integral - I))
    print()


# функция, выводящая результаты, относящиеся к формуле Милна

def miln():
    # максимум шестой производной равен нуля для данного многочлена
    # поэтому отрезок не разбиваем
    n = 1
    h = b - a
    I = integral_miln(h, n)
    print('------МЕТОД МИЛНА------')
    print('n =', n)
    print('h =', h)
    print('I =', I)
    print('R =', 0)
    print()


# функция,вычисляющая интеграл и выводящая результаты, относящиеся к формуле Гаусса

def gauss():
    n = 3
    A = [5/9, 8/9, 5/9]
    t = [-numpy.sqrt(3/5), 0, numpy.sqrt(3/5)]
    result = 0
    for i in range(n):
        result += A[i] * f((a + b) / 2 + (b - a) / 2 * t[i])
    result *= (b - a) / 2
    print('------МЕТОД ГАУССА------')
    print('I =', result)
    print('n =', n)
    print()


coefs = [4.8, 1.5, 6.3, -2.7, 3.7, 4.4]
integral = 729.146666666668
a = 1
b = 3
epsilon = 0.05

print('\nАналитическое значение интеграла: I =', integral, '\n')

left_rect()
miln()
gauss()

