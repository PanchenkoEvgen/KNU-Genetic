import matplotlib.pyplot as plt
import math
import random

N = 50 #population 
T = 350 #quantity of generations

left_edge = 0
right_edge = 1

eps = 0.0001
epsN = int(-math.log10(eps))

selection_n = 50
mutation_n = 3

def func(x):
    return 0.4 + abs(math.sin(2*math.pi*x))*abs(math.cos(1.5*x))-0.5*abs(x-0.7)

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

def encoding(individual, n): #Не потрібний
    individual_genes = []
    if individual == 1.0:
        individual = int(individual*pow(10,n))-1
    else:
        individual = int(individual*pow(10,n))
    for j in range(n):
        individual_genes.append(individual % 10)
        individual //= 10
    return individual_genes

def selection(population, num): #num - кількість особистостей, що ми хочемо обрати
    sum = 0
    for individual, survavability in population:
        sum += survavability
    probability_split = [[population[0][0],population[0][1]/sum*100]]
    for i in range(1,len(population)):
        probability_split.append([population[i][0], population[i][1]/sum*100+probability_split[i-1][1]])
    selected_individuals = []
    for n in range(num):
        random_individual = random.uniform(0,100)
        individual_index = 0
        for i in range(len(probability_split)):
            if random_individual < probability_split[i][1]:
                individual_index = i
                break
        selected_individuals.append(probability_split[individual_index][0])
    return selected_individuals

def crossover(individuals):
    new_individuals = []
    if len(individuals) < 2:
        return "Not enough individuals"
    for i in range(0, len(individuals), 2):
        parent1 = individuals[i]
        parent2 = individuals[i+1]
        child1 = math.sqrt(parent1*parent2)
        child2 = (parent1+parent2)/2
        new_individuals.append(child1)
        new_individuals.append(child2)
    return new_individuals

def mutation(individuals, n):
    for i in range(n):
        random_i = random.randint(0,len(individuals)-1)
        random_num = random.randint(0,9)
        random_pow = random.randint(1,epsN)
        random_sign = random.choice([1, -1])
        individuals[random_i] = individuals[random_i] + random_sign*random_num*pow(10, -random_pow)
        if individuals[random_i] > 1:
            individuals[random_i] = 1
        if individuals[random_i] < 0:
            individuals[random_i] = 0
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
        print("Start gen")
        temp = []
        for i in range(len(population)):
            temp.append(population[i][0])
        print(temp)
        # draw_plot(population)
        selected = selection(population, selection_n)
        print("Selected")
        pop = []
        print(selected)
        for s in selected:
            pop.append([s, func(s)])
        # draw_plot(pop)
        children = crossover(selected)
        print("Children")
        pop = []
        print(children)
        for c in children:
            pop.append([c, func(c)])
        # draw_plot(pop)
        mut_children = mutation(children, mutation_n)
        population = generate_population(mut_children)
        print("Mutated")
        temp = []
        for i in range(len(population)):
            temp.append(population[i][0])
        print(temp)
        # draw_plot(population)
        t += 1
    return population

ind = starting_individuals(left_edge,right_edge,N)
P = generate_population(ind)
result = new_gen(P, T)

draw_plot(result)
