import numpy as np
from scipy.linalg import lu_factor, lu_solve


def solve_2_x_2(b_left=36+22, b_right=36+22):
    return lu_solve((lu, piv), np.array([b_left, b_right]))


def solve_a(testing):
    x = np.zeros(n)
    for left in range(center_index_left, -1, -1):
        right = n - left - 1
        if left == 7 or right == 23:
            continue
        x[left], x[right] = solve_2_x_2() if testing else solve_2_x_2(n + left, n + right)

    s = sum(x)
    b4 = np.array([10*(n-2) + 36 + 22 - s * 10, 36 + 22, 14*(n-2) + 36 + 22 - s * 14, 36 + 22]) if testing \
        else np.array([n + 7 - s * 10, n + 8, n + 23 - s * 14, n + 24])
    x[7], x[8], x[23], x[24] = np.linalg.solve(A4, b4)
    return x


A2 = np.array([[36, 22], [22, 36]])

A4 = np.array([[36, 10, 10, 22], \
              [0, 36, 22, 0], \
              [14, 22, 36, 14], \
              [22, 0, 0, 36]])

lu, piv = lu_factor(A2)
n = 32
center_index_left = round(n / 2) - 1

np.set_printoptions(precision=4, suppress=True)
print('Solution of Ax=b:', solve_a(testing=False))
print('Solution of test:', solve_a(testing=True))
