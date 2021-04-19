# Solve the ordinary differential equation
#       -u''(t) + a(t) * u(t) = phi(t), t \in [0, 1]
#               u(0) = alpha_0
#               u(1) = alpha_1
# using the finite difference method.
#
# Approximate the solution at points t_i = i * h, i = 0, ... , n+1
# whith stepsize h = 1/(n + 1), n \in IN 
#
# The difference quotient for u''(t) is
#       1/h^2 * (u(t-h) - 2 * u(t) + u(t+h))
#
# So the finite difference equation is for the give ODE is
#   1/h^2 * (- u_{i-1} + (2 + a_i) * u_i - u_{i+1}) = phi_i
#
# Thus the matrix form is
#           [[2 + a_1   -1              0                   ... 0   ]       [u_1            [phi_1 - alpha_0/h^2
#            [-1        2 + a_2 - 1     0                   ... 0   ]         .                 .
#    1/h^2   [0         -1              2 + a_3     1   0   ... 0   ]         .     =           .
#            [                      ...                             ]         .                 .
#            [0         ...             0           -1      2 + a_n]]       u_n]            phi_n - alpha_1/h^2]

import numpy
import matplotlib.pyplot as plt

# boundary values
alpha_0 = 0
alpha_1 = 1

# number of grid points
n = 50

# step size
h = (alpha_1 - alpha_0) / n

# parameters
a = numpy.random.rand(50)
phi = [0.7] * n

# Construct LHS
A = numpy.zeros((n, n))
A[0, 0] = 2 + a[0]
A[0, 1] = -1

A[n - 1, n - 1] = 2 + a[n - 1]
A[n - 1, n - 2] = -1

for i in range(1, n - 1):
    A[i, i - 1] = -1
    A[i, i] = 2 + a[i-1]
    A[i, i + 1] = -1

# Construct RHS
b = numpy.zeros(n)
b[0] = phi[0] * h**2 - alpha_0
b[n - 1] = phi[n - 1] * h**2 - alpha_1

for i in range(1, n - 1):
    b[i] = phi[i]

# Solve the system
y = numpy.linalg.solve(A, b)

t = numpy.linspace(0, 1, 50)

plt.figure(figsize=(10,8))
plt.plot(t, y)
plt.show()