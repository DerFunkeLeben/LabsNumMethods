import numpy
import matplotlib.pyplot as plt
import math


def f(x):
    return numpy.exp(-x) + numpy.cos(x)


def taylor_term(x, k):
    return (-x)**k / math.factorial(k) + ((-1)**k * x**(2.*k)) / math.factorial(2.*k)


def S(x, N):
    result = 0.
    for i in range(N):
        result = result + taylor_term(x, i)
    return result


def S_rounded_(x, N):
    result = 0.
    for i in range(N):
        result = ROUND(result + ROUND(taylor_term(x, i)))
    return result


def S_rounded(x, N):
    result = x.copy()
    for i in range(len(x)):
        result[i] = S_rounded_(x[i], N)
    return result


def absError(s, f):
    return numpy.abs(s-f)


def relError(s, f):
    return absError(s, f) / numpy.abs(s)


# machine epsilon
def eps():
    epsilon = 1.
    while 1. + epsilon * .5 != 1.:
        epsilon *= .5
    return epsilon * .5


def ROUND(x):
    return float(numpy.format_float_scientific(x, precision=4))


a = 1.
b = 7.
c = (a + b) / 2
x_data = numpy.linspace(a, b, 1000)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(x_data, f(x_data), label='$f(x)$', color="black")

line_styles = ((0, (3, 1, 1, 1, 1, 1)), 'dotted', 'dashed', 'dashdot', (0, (1, 10)))
for i in range(1,6):
    axs[0].plot(x_data, S(x_data, i), label=f'$S(x,{i})$', ls=line_styles[i-1])

for i in range(1,6):
    axs[1].plot(x_data, absError(S(x_data, i), f(x_data)), label=f'$\Delta S(x,{i})$', ls=line_styles[i-1])

axs[0].legend()
axs[1].legend()
axs[0].set(ylim=(-5., 5.))
axs[1].set(ylim=(0, 5.))
plt.show()

p_sum = taylor_term(c, 1)
n_t = taylor_term(c, 2)
N_machine_error = 2

while numpy.abs(n_t/p_sum) > eps():
    p_sum += n_t
    N_machine_error += 1
    n_t = taylor_term(c, N_machine_error)
print(N_machine_error)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))

axs[0].plot(x_data, absError(S(x_data, N_machine_error), f(x_data)), label=f'$\Delta S(x,{N_machine_error})$', color="black")
axs[1].plot(x_data, relError(S(x_data, N_machine_error), f(x_data)), label=f'$\delta S(x,{N_machine_error})$', color="black")
axs[0].legend()
axs[1].legend()
axs[0].set(ylim=(0.0, 10E-12))
axs[1].set(ylim=(0.0, 10E-12))
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].plot(x_data, absError(S_rounded(x_data, N_machine_error), f(x_data)), label=f'$\Delta S(x,{N_machine_error})$')
axs[1].plot(x_data, relError(S_rounded(x_data, N_machine_error), f(x_data)), label=f'$\delta S(x,{N_machine_error})$')
axs[0].legend()
axs[1].legend()
plt.show()