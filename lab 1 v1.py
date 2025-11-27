import matplotlib.pyplot as plt
import math
import random
import numpy as np

N = 50 #population 
T = 20 #quantity of generations

left_edge = 0
right_edge = 1

eps = 0.0000001
epsN = int(-math.log10(eps))

selection_n = N
mutation_n = 5

def func(x):
    term1 = 0.5 * abs(x) + 0.5
    term2 = -0.5 * math.cos((70.0/6) * (math.tan(x) - 1.5 * math.sin(x)))
    term3 = math.sin((120.0/7) * x) * abs(x - 0.5)
    term4 = -math.cos(x) * abs(math.sin(x))
    return (term1 + term2 + term3 + term4)
    # return 0.4 + abs(math.sin(2*math.pi*x))*abs(math.cos(1.5*x))-0.5*abs(x-0.7)

def starting_individuals(l, r, n):
    individuals = []
    step = abs(r-l)/n
    while (l <= r):
        individuals.append(l)
        l += step
    return individuals

def generate_population(individuals):
    population = []
    for i in individuals:
        population.append([i, func(i)])
    return population

def encoding(individual, n):
    individual_genes = []
    if individual == 1.0:
        individual = int(individual*pow(10,n))-1
    else:
        individual = int(individual*pow(10,n))
    for j in range(n):
        individual_genes.append(individual % 10)
        individual //= 10
    return individual_genes

def decoding(genes, n):
    individual = 0
    for i in range(n):
        individual += genes[n-1-i] * pow(10, -i-1)
    return round(individual, epsN)

def selection(population, num): #num - кількість особистостей, що ми хочемо обрати
    selected_individuals = []
    population.sort(key=lambda x:x[1], reverse = True)
    s_max = population[0][1]
    n = 0
    while n < num:
        random_individual = random.gauss(s_max, 0.5)
        if random_individual > s_max:
            random_individual = abs(random_individual-s_max)
        individual_index = 0
        for individual, survavability in population:
            if n >= num:
                break
            if survavability >= random_individual:
                selected_individuals.append(individual)
                n += 1
            else:
                break
    return selected_individuals

def selection(population, num): #num - кількість особистостей, що ми хочемо обрати
    selected_individuals = []
    population.sort(key=lambda x:x[1], reverse = True)
    s_max = population[0][1]
    n = 0
    while n < num:
        random_individual = random.gauss(s_max, 0.5)
        if random_individual > s_max:
            random_individual = abs(random_individual-s_max)
        individual_index = 0
        for individual, survavability in population:
            if n >= num:
                break
            if survavability >= random_individual:
                selected_individuals.append(individual)
                n += 1
            else:
                break
    return selected_individuals

def crossover(individuals):

    def get_fitness(ind):
        return func(decoding(ind, epsN))

    new_individuals = []
    for i in range(0, len(individuals), 2):
        alpha = random.uniform(0,1)
        parent1 = individuals[i]
        parent2 = individuals[i+1]
        child1 = []
        child2 = []
        for g1,g2 in zip(parent1, parent2):
            child1.append(math.floor(alpha*g1+(1-alpha)*g2))
            child2.append(math.floor((1-alpha)*g1+alpha*g2))
        family = [parent1, parent2, child1, child2]
        family.sort(key=get_fitness, reverse=True)
        new_individuals.extend(family[:2])

    return new_individuals

def mutation(individuals, n):
    for i in range(n):
        random_i = random.randint(0,len(individuals)-1)
        random_num = random.randint(0,9)
        random_pow = random.randint(1,epsN)
        random_sign = random.choice([1, -1])
        individuals[random_i] = individuals[random_i] + random_sign*random_num*pow(10, -random_pow)
        if individuals[random_i] > right_edge:
            individuals[random_i] = right_edge
        if individuals[random_i] < left_edge:
            individuals[random_i] = left_edge
    return individuals

def shotgun_mutation(individuals, n):
    used_i = []
    for i in range(n):
        random_ind = random.randint(0,len(individuals)-1)
        if random_ind not in used_i:
            random_num = random.uniform(-0.05,0.05)
            individuals[random_ind] += random_num
        else:
            i -= 1
    return individuals

def draw_plot(result):
    x = [r[0] for r in result]
    y = [r[1] for r in result]

    plt.scatter(x, y, color="red")
    p = generate_population(starting_individuals(left_edge, right_edge, 1000))
    x_true = [r[0] for r in p]
    y_true = [r[1] for r in p]
    plt.scatter(x_true, y_true, color="blue", s=0.2)
    plt.show()

def new_gen(population, T):
    t = 0
    while (t < T):
        print(f"Gen {t}")
        temp = [p[0] for p in population]
        print(temp)
        if t == 0:
            draw_plot(population)
        for i in range(len(population)):
            population[i][0] = encoding(population[i][0], epsN)
        selected = selection(population, selection_n)
        print("Selected")
        temp = []
        pop = []
        for s in selected:
            temp.append(decoding(s, epsN))
        print(temp)
        for i in range(len(temp)):
            pop.append([temp[i],func(temp[i])])
        # draw_plot(pop)
        children = crossover(selected)
        print("Children")
        temp = []
        pop = []
        for c in children:
            temp.append(decoding(c, epsN))
        print(temp)
        for i in range(len(temp)):
            pop.append([temp[i],func(temp[i])])
        for i in range(len(children)):
            children[i] = decoding(children[i], epsN)
        # draw_plot(pop)
        if t != T-1:
            mut_children = shotgun_mutation(children, mutation_n)
            for i in range(len(mut_children)):
                temp.append(mut_children[i])
            print("Mutated")
            population = generate_population(mut_children)
            print(temp)
        else:
            for i in range(len(children)):
                temp.append(children[i])
            print("Final population")
            population = generate_population(children)
            print(temp)
        # draw_plot(population)
        
        t += 1
    return population

ind = starting_individuals(left_edge,right_edge,N)
P = generate_population(ind)
s = []
xr = []
for i in range(N):
    xr.append(random.uniform(left_edge,right_edge))
for i in xr:
    s.append([i, func(i)])
result = new_gen(s, T)

draw_plot(result)

# p = generate_population(starting_individuals(left_edge, right_edge, 1000))
# x_true = [r[0] for r in p]
# y_true = [r[1] for r in p]
# plt.scatter(x_true, y_true, color="blue", s=0.2)
# plt.show()
