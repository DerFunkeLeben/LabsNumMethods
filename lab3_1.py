import numpy as np
import numpy.linalg as LA
from scipy.linalg import lu_solve
import matplotlib.pyplot as plt


def create_A(n):
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(np.arctan(.1*(10*i+j+1)))
        A.append(np.array(row))
    return np.array(A)


def create_A_perturbed(A):
    A_perturbed = A.copy()
    A_perturbed[0, 0] += .001
    return A_perturbed


def create_b(A, n, x_i):
    x = []
    for i in range(n):
        x.append(x_i)
    return np.dot(A, np.array(x))


def get_lu_and_piv(A):
    LU = A.copy()
    n = np.size(A, 0)
    piv = np.zeros(n)

    for i in range(n):
        LU_t = LU.transpose()
        max_index = np.argmax(LU_t[i])
        piv[i] = max_index
        LU[i], LU[max_index] = LU[max_index], LU[i].copy()
        for j in range(i+1, n):
            LU[j][i] /= LU[i][i]
            for k in range(i + 1, n):
                LU[j][k] -= LU[j][i] * LU[i][k]

    return LU, piv


def lu_with_pivots(A, b):
    return lu_solve(get_lu_and_piv(A), b)


def lu_no_pivots(A, b):
    n = np.size(A, 0)
    LU = A.copy()
    for i in range(n):
        for j in range(i+1, n):
            LU[j][i] /= LU[i][i]
            for k in range(i + 1, n):
                LU[j][k] -= LU[j][i] * LU[i][k]

    piv = []
    for k in range(n):
        piv.append(k)
    piv = np.array(piv)

    return lu_solve((LU, piv), b)


A = create_A(5)
b = create_b(A, 5, 52)
np.set_printoptions(precision=4, suppress=True)

x_no_pivots = lu_no_pivots(A, b)
x_with_pivots = lu_with_pivots(A, b)
print('Initial System:')
print('Solution LU:', x_no_pivots)
print('Solution LU(with Partial Pivoting):', x_with_pivots)
print()

A_perturbed = create_A_perturbed(A)
x_no_pivots_perturbed = lu_no_pivots(A_perturbed, b)
x_with_pivots_perturbed = lu_with_pivots(A_perturbed, b)
print('Perturbed System:')
print('Solution LU:', x_no_pivots_perturbed)
print('Solution LU(with Partial Pivoting):', x_with_pivots_perturbed)
print()

A_inv = LA.inv(A)
A_inv_norm = LA.norm(A_inv, ord=np.inf)
prior_estimate = .001 * A_inv_norm

posterior_estimate = LA.norm(x_no_pivots - x_no_pivots_perturbed, ord=np.inf) / LA.norm(x_no_pivots, ord=np.inf)

print('Prior error estimation: ', prior_estimate)
print('Posterior error estimation: ', posterior_estimate)
print('_____________________________________')

error_no_pivots_list = []
error_with_pivots_list = []

for k in range(5, 16):
    A = create_A(k)
    b = create_b(A, k, 52)
    x_no_pivots = lu_no_pivots(A, b)
    x_with_pivots = lu_with_pivots(A, b)

    A_perturbed = create_A_perturbed(A)
    x_no_pivots_perturbed = lu_no_pivots(A_perturbed, b)
    x_with_pivots_perturbed = lu_with_pivots(A_perturbed, b)

    error_no_pivots = LA.norm(x_no_pivots - x_no_pivots_perturbed, ord=np.inf) \
                                   / LA.norm(x_no_pivots, ord=np.inf)
    error_with_pivots = LA.norm(x_with_pivots - x_with_pivots_perturbed, ord=np.inf) \
                                     / LA.norm(x_with_pivots, ord=np.inf)

    error_with_pivots_list.append(error_with_pivots)
    error_no_pivots_list.append(error_no_pivots)

    print('N = ' + k.__str__() + ':')

    print('Error:', error_no_pivots)
    print('Error(with Partial Pivoting):', error_with_pivots)
    print()

fig = plt.subplots()
x_data = np.linspace(5, 15, 11)
plt.plot(x_data, error_no_pivots_list, label=f'$\delta x$')
plt.plot(x_data, error_with_pivots_list, label=f'$\delta x(with Partial Pivoting)$')
plt.legend()
plt.show()