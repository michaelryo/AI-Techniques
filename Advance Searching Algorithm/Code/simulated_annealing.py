from heading import *

def simulated_annealing(init, energy, neighbor, schedule, start_point = None):
    current = init
    for t in range(sys.maxsize):
        T = schedule()(t)
        if T == 0:
            break
        neighbors = [neighbor(current)]
        if not neighbors:
            break
        next_choice = random.choice(neighbors)
        delta_e =  energy(next_choice) - energy(current)
        if delta_e < 0 or probability(np.exp(- delta_e / T)):
            current = next_choice
    if start_point != None:
        current.append(start_point)
    return (current,energy(current))