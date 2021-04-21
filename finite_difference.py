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


class FiniteDifference():
    def __init__(self, grid_points, lower_boundary_value, upper_boundary_value, a, phi):
        self.grid_points = grid_points

        # Stepsize
        n = len(grid_points) - 1
        h = (grid_points[0] - grid_points[n - 1]) / n

        # Construct LHS
        self.A = numpy.zeros((n + 1, n + 1))
        self.A[0, 0] = 1
        self.A[n, n] = 1

        for i in range(1, n):
            t_i = lower_boundary_value + i * h
            self.A[i, i - 1] = -1
            self.A[i, i] = 2 + a(t_i)
            self.A[i, i + 1] = -1

        # Construct RHS
        self.b = numpy.zeros(n + 1)
        self.b[0] = lower_boundary_value
        self.b[n] = upper_boundary_value

        for i in range(1, n):
            t_i = lower_boundary_value + i * h
            self.b[i] = phi(t_i)

    def solve(self):
        y = numpy.linalg.solve(self.A, self.b)
        return y

    def plot_solution(self):
        y = self.solve()
        plt.plot(self.grid_points, y)
        plt.show()
