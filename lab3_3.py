import numpy as np
import numpy.linalg as LA


def create_A(m, beta):
    A = []
    for i in range(m):
        row = []
        for j in range(m):
            row.append(np.cos(i+j) / .1 / beta + .1 * beta * np.exp(-(i - j)**2))
        A.append(np.array(row))
    return np.array(A)


def create_b(A, m, x_i):
    x = []
    for i in range(m):
        x.append(x_i)
    return np.dot(A, np.array(x))


def solve_conjugate_gradient(A, b, epsilon=np.power(10., -9)):
    x = np.zeros(np.size(b))
    residual_0 = np.abs(np.dot(A, x) - b)
    descent_direction_n = residual_0
    residual_n = residual_0

    while LA.norm(residual_n) / LA.norm(residual_0) > epsilon:
        alpha = np.dot(residual_n, residual_n) / np.dot(np.dot(A, descent_direction_n), descent_direction_n)
        x = x + alpha * descent_direction_n
        residual_n_next = residual_n - alpha * np.dot(A, descent_direction_n)
        beta = np.dot(residual_n_next, residual_n_next) / np.dot(residual_n, residual_n)
        descent_direction_n = residual_n_next + beta * descent_direction_n
        residual_n = residual_n_next
    return x


m = 25
n = 52
beta = (np.abs(66-n) + 5) * m
A = create_A(m, beta)
b = create_b(A, m, n)

x = solve_conjugate_gradient(A, b)
np.set_printoptions(precision=9, suppress=True)

print(x)
print(LA.solve(A, b))
