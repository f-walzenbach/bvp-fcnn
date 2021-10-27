from random import random
import json

import numpy
from numpy.polynomial import Polynomial as P

with open("config/config.json") as config_file:
    config = json.load(config_file)

discretized_domain = numpy.linspace(
    config["parameters"]["lower_bound"],
    config["parameters"]["upper_bound"],
    config["parameters"]["number_of_grid_points"],
)


def generate_data_sets(size):
    # Generate training set
    # Use set() to make sure that entries are unique
    setOfNumbers = set()
    while len(setOfNumbers) < size:
        setOfNumbers.add((
            5 * random(),
            5 * random(),
            5 * random(),
            5 * random(),
            5 * random()
        ))

    with open('data_set.txt', 'w') as f:
        for val in setOfNumbers:
            f.write("{} {} {} {} {}\n".format(
                val[0], val[1], val[2], val[3], val[4]))


def read_data_set(filename):
    """ Data set for the ordinary differential equation -u''(t) + a(t)u(t) = phi(t) 

    
    """
    
    data = []
    with open(filename) as f:
        line = f.readline()
        while line:
            coefficients = [float(s) for s in line.strip().split(" ")]
            polynomial = P(coefficients)
            
            data.append((
                polynomial(discretized_domain),
                numpy.sin(discretized_domain)
            ))
            
            line = f.readline()
    return data
