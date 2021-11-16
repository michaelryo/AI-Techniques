from heading import *

def test_duplicated(x):
    duplicated = False
    duplicate = dict(Counter(x))
    for key,value in duplicate.items():
        if value > 1:
            duplicated = True
    return duplicated

def generate_for_Non_Duplication(x, y):
    new_gen1 = x[:]
    new_gen2 = y[:]
    new_gen1.remove(start_point)
    new_gen2.remove(start_point)
    n = len(new_gen1)
    new_gen = []
    while True:
        c = random.randint(0, n)
        new_gen = new_gen1[:c] + new_gen2[c:]
        if test_duplicated(new_gen):
            pass
        else:
            new_gen.insert(0, start_point)
            break
    return new_gen

def recombine_for_Non_Duplication(x, y):
    new_gen1 = x[:]
    new_gen2 = y[:]
    new_gen1.remove(start_point)
    new_gen2.remove(start_point)
    n = len(new_gen1)
    duplicated = False
    c = random.randrange(0, n)
    new_gen = new_gen1[:c] + new_gen2[c:]
    new_gen.insert(0, start_point)
    duplicated = test_duplicated(new_gen)
    if duplicated:
        new_gen = generate_for_Non_Duplication(x, y)
    return new_gen

def mutate_for_Non_Duplication(x, gene_pool, pmut):
    if random.uniform(0, 1) >= pmut:
        return x
    mutate_population = x[:]
    mutate_population.remove(start_point)
    random.shuffle(mutate_population)
    mutate_population.insert(0, start_point)
    return mutate_population


def fitness_threshold(fitness_fn, f_thres, population):
    if not f_thres:
        return None

    fittest_individual = min(population, key=fitness_fn)
    if fitness_fn(fittest_individual) <= f_thres:
        return fittest_individual

    return None


def select(population, fitness_fn):
    fitnesses = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(2)]


def recombine(x, y):
    n = len(x)
    c = random.randrange(0, n)
    return x[:c] + y[c:]


def recombine_uniform(x, y):
    n = len(x)
    result = [0] * n
    indexes = random.sample(range(n), n)
    for i in range(n):
        ix = indexes[i]
        result[ix] = x[ix] if i < n / 2 else y[ix]

    return ''.join(str(r) for r in result)


def mutate(x, gene_pool, pmut):
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    return x[:c] + [new_gene] + x[c + 1:]


def genetic_algorithm(population, fitness_fn, gene_pool, f_thres=None, ngen=1200, pmut=0.1, start = None, non_duplication = True):
    global start_point
    start_point = start
    if non_duplication:
        for generation in range(ngen):
            population = [mutate_for_Non_Duplication(recombine_for_Non_Duplication(*select(population, fitness_fn)), gene_pool, pmut) for i in range(len(population))]
            # stores the individual genome with the highest fitness in the current population
            current_best = min(population, key = fitness_fn)
            current_best.append(start_point)
            print(f'Current best: {current_best}\t\tGeneration: {str(generation)}\t\tFitness: {fitness_fn(current_best)}\r', end='')
            del current_best[-1]
            # compare the fitness of the current best individual to f_thres
            fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
            
            # if fitness is greater than or equal to f_thres, we terminate the algorithm
            if fittest_individual:
                return fittest_individual, generation
    else:       
        for generation in range(ngen):
            population = [mutate(recombine(*select(population, fitness_fn)), gene_pool, pmut) for i in range(len(population))]
            # stores the individual genome with the highest fitness in the current population
            current_best = min(population, key = fitness_fn)
            print(f'Current best: {current_best}\t\tGeneration: {str(generation)}\t\tFitness: {fitness_fn(current_best)}\r', end='')
            
            # compare the fitness of the current best individual to f_thres
            fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
            
            # if fitness is greater than or equal to f_thres, we terminate the algorithm
            if fittest_individual:
                return fittest_individual, generation
    return max(population, key=fitness_fn) , generation        