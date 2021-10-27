import json
from typing import List

import numpy

with open("config/config.json") as config_file:
    config = json.load(config_file)

discretized_domain = numpy.linspace(
    config["parameters"]["lower_bound"],
    config["parameters"]["upper_bound"],
    config["parameters"]["number_of_grid_points"],
)


def compute_finite_difference_solution(left_hand_side_values: List[int], right_hand_side_values: List[int]) -> List[int]:
    """ Solve the ordinary differential equation -u''(t) + a(t)u(t) = phi(t) using the finite difference solution

    Parameters
    ----------
    left_hand_side_values : List[int]
        Function values for phi

    right_hand_side_values : List[int]
        Function values for a

    Returns
    -------
    
    """
    
    # Stepsize
    n = len(discretized_domain) - 1
    h = (discretized_domain[0] - discretized_domain[n - 1]) / n

    # Construct LHS
    A = numpy.zeros((n + 1, n + 1))
    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i - 1] = -1
        A[i, i] = 2 + right_hand_side_values[i]
        A[i, i + 1] = -1

    #A = 1/(h**2) * A

    # Construct RHS
    b = numpy.zeros(n + 1)
    b[0] = config["parameters"]["lower_boundary_value"]
    b[n] = config["parameters"]["upper_boundary_value"]

    for i in range(1, n):
        b[i] = left_hand_side_values[i] * h**2

    # Solve linear system and return solution
    y = numpy.linalg.solve(A, b)
    return y
