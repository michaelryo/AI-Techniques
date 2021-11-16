from heading import *

def hill_climbing(city_list, neighbour_function, cost_fun, times=500, iteration = 1000, start_point = None):
    current = city_list
    iterations = iteration
    count = 0
    while iterations:
        neighbours = [neighbour_function(current)]
        if not neighbours:
            break
        neighbour = argmax_random_tie(neighbours, key=lambda node: cost_fun(node))
        if count >= times:
            break
        if cost_fun(neighbour) < cost_fun(current):
            current = neighbour
            count = 0
        else:
            count += 1
        iterations -= 1
    if start_point != None:
        current.append(start_point)
    return (current,cost_fun(current))