import numpy as np

def f(x):
    return -3+3*x-2*x**2+3*x**3+2*x**4

h = 0.0625
x = np.zeros(9)
x[0] = -2
for i in range(1, 9):
    x[i] = x[i-1]+h

s = 0
for i in range(1, 8):
    s += f(x[i])
print((s+(f(x[0])+f(x[8]))/2)*h)

