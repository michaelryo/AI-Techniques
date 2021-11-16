from notebook import psource, heatmap, gaussian_kernel, show_graph, plot_NQueens
from collections import Counter
from utils import *
import sys
import random
# Needed to hide warnings in the matplotlib sections
import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
import math


# ______________________________________________________________________________
# hill_climbing algorithms

class Problem:

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self, state):
        raise NotImplementedError

class NQueensProblem(Problem):

    def __init__(self, N):
        super().__init__(list([1] * N))
        self.N = N

    def rand_neighbor(self, state):
        neighbour_state = state[:]
        vertical_rand = random.randint(0, len(state) -1)
        horizontal_rand = random.randint(1, len(state))
        neighbour_state[vertical_rand] = horizontal_rand
        return neighbour_state

    def actions(self, state):
        return ([self.rand_neighbor])

    def result(self, state, action):
        return action(state)

    def path_cost(self, c, state1, action, state):
        cost = 0
        martix = np.zeros((self.N, self.N))
        size = len(state)
        for i in range(size):
            martix[i][state[i] - 1] = 1

        for i in range(size):
            horizonal = martix[i].sum()
            if horizonal >= 2:
                cost += horizonal
            vertical = martix[:,i].sum()
            if vertical >= 2:
                cost += vertical

        list_i = []
        list_j = []
        l1 = []
        for i in range(2 * size -1):
            record_temp = np.zeros((0,))
            for j in range(i + 1):
                k = i - j
                if k < size and k >= 0 and j < size:
                    record_temp = np.concatenate((record_temp, [martix[j][k]]))
                    list_i.append(j)
                    list_j.append(k)
            if record_temp.sum() >= 2:
                cost += record_temp.sum()

        for i in range(len(list_j)):
            a = list_j.pop()
            l1.append(a)
        rec = -1
        for i in range(2 * size -1):
            record_temp = np.zeros((0,))
            for j in range(i + 1):
                k = i - j
                if k < size and k >= 0 and j < size:
                    rec += 1
                    record_temp = np.concatenate((record_temp, [martix[list_i[rec]][l1[rec]]]))
            if record_temp.sum() >= 2:
                cost += record_temp.sum()
        return cost

    def value(self, state):
        return -1 * self.path_cost(None, None, None, state)


def define_map(city_location):
    global distances
    city_locations = city_location
    all_cities = []
    distances = {}

    for city in city_locations.keys():
        distances[city] = {}
        all_cities.append(city)
    all_cities.sort()

    for name_1, coordinates_1 in city_locations.items():
            for name_2, coordinates_2 in city_locations.items():
                distances[name_1][name_2] = np.linalg.norm(
                    [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])
                distances[name_2][name_1] = np.linalg.norm(
                    [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])

    return (all_cities, distances)

class FindMax_Problem(Problem):

    def __init__(self, initial, func, step = 0.05):
        super().__init__(initial)
        self.function = func
        self.step = step

    def step_neighbor(self, state):
        neighbour_state = state[:]
        rand_bool = random.randint(0, 1)
        if rand_bool:
            neighbour_state[0] += self.step
        else:
            neighbour_state[0] -= self.step
        return neighbour_state    

    def actions(self, state):
        return ([self.step_neighbor])

    def result(self, state, action):
        return action(state)

    def path_cost(self, c, state1, action, state):
        cost = self.function(state[0])
        return cost

    def value(self, state):
        return -1 * self.path_cost(None, None, None, state)

    def draw_function(self):
        x = []
        y = []
        i = -20
        while i < 20:
            x.append(i)
            y.append(self.function(i))
            i += 0.2
        plt.plot(x, y, ls="-", lw=2, label="plot figure")
        plt.show()

def draw_function(function):
    x = []
    y = []
    i = -20
    while i < 20:
        x.append(i)
        y.append(function(i))
        i += 0.2
    plt.plot(x, y, ls="-", lw=2, label="plot figure")
    plt.show()
# ______________________________________________________________________________
# simulated_annealing algorithms

def update_quadratic_function(func):
    global quadratic_function_for_genetic
    quadratic_function_for_genetic = func

def fitness_fn_for_queens(sample):
    cost = 0
    size = len(sample)
    martix = np.zeros((size, size))
    for i in range(size):
        martix[i][sample[i] - 1] = 1

    for i in range(size):
        horizonal = martix[i].sum()
        if horizonal >= 2:
            cost += horizonal
        vertical = martix[:,i].sum()
        if vertical >= 2:
            cost += vertical

    list_i = []
    list_j = []
    l1 = []
    for i in range(2 * size -1):
        record_temp = np.zeros((0,))
        for j in range(i + 1):
            k = i - j
            if k < size and k >= 0 and j < size:
                record_temp = np.concatenate((record_temp, [martix[j][k]]))
                list_i.append(j)
                list_j.append(k)
        if record_temp.sum() >= 2:
            cost += record_temp.sum()

    for i in range(len(list_j)):
        a = list_j.pop()
        l1.append(a)
    rec = -1
    for i in range(2 * size -1):
        record_temp = np.zeros((0,))
        for j in range(i + 1):
            k = i - j
            if k < size and k >= 0 and j < size:
                rec += 1
                record_temp = np.concatenate((record_temp, [martix[list_i[rec]][l1[rec]]]))
        if record_temp.sum() >= 2:
            cost += record_temp.sum()
    cost = 100. - cost
    return cost

def fitness_fn_for_TSP(sample):
    """ total distance for the Traveling Salesman to be covered if in state2  """
    cost = 0
    for i in range(len(sample) - 1):
        cost += distances[sample[i]][sample[i + 1]]
    cost += distances[sample[0]][sample[-1]]
    cost = 10000 - cost
    return cost

def convert_dec(sample):
    dec = 0
    dec_1 = sample[1] * 10 + sample[2]
    dec_2 = (sample[4] * 10 + sample[5]) / 100.
    if sample[0] == '-':
        dec = -(dec_1 + dec_2)
    else:
        dec = dec_1 + dec_2
    return dec

def fitness_fn_for_quadratic(bit):
    cost = 0
    convert_able = True
    if (bit[0] == '+' or bit[0] == '-'):
        cost += 0
    else:
        cost += 20000
        convert_able = False
    if(bit[3] == '.'):
        cost += 0
    else:
        cost += 20000
        convert_able = False
    if isinstance(bit[1] ,int) and isinstance(bit[2] ,int) and isinstance(bit[4] ,int) and isinstance(bit[5] ,int):
        cost += 0
    else:
        cost += 20000
        convert_able = False
    if convert_able:
        dec = convert_dec(bit)
        quad = quadratic_function_for_genetic(dec)
        cost += quad
    cost = 100000 - cost
    return cost

def init_population_for_Non_Duplication(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""
    g = len(gene_pool)
    initial_population = gene_pool
    population = []
    for i in range(pop_number):
        random.shuffle(initial_population)
        new_individual = initial_population[:]
        population.append(new_individual)
    return population

def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        population.append(new_individual)
    return population
