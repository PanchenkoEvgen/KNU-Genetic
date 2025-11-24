import matplotlib.pyplot as plt
import math
import random

N = 20 #population 
T = 100 #quantity of generations

left_edge = 0
right_edge = 1

eps = 0.0001
epsN = int(-math.log10(eps))

selection_n = 20
mutation_n = 3

def func(x):
    term1 = 0.5 * abs(x) + 0.5
    term2 = -0.5 * math.cos((70.0/6) * (math.tan(x) - 1.5 * math.sin(x)))
    term3 = math.sin((120.0/7) * x) * abs(x - 0.5)
    term4 = -math.cos(x) * abs(math.sin(x))
    return (term1 + term2 + term3 + term4)

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
        random_individual = random.uniform(0,s_max)
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

def crossover(individuals, num):
    new_individuals = []
    if len(individuals) < 2:
        return "Not enough individuals"
    for i in range(0, len(individuals), 2):
        parent1 = individuals[i]
        parent2 = individuals[i+1]
        child1 = parent1
        child2 = parent2
        for i in range(int(num/2)):
            temp = parent1[i]
            child1[i] = parent2[i]
            child2[i] = temp
        new_individuals.append(child1)
        new_individuals.append(child2)
    return new_individuals

def mutation(individuals, n):
    for i in range(n):
        random_ind = random.randint(0,len(individuals)-1)
        random_gen = random.randint(0,len(individuals[random_ind])-1)
        random_num = random.randint(0,9)
        individuals[random_ind][random_gen] = random_num
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
        temp = [p[0] for p in population]
        print(temp)
        # draw_plot(population)
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
        children = crossover(selected, epsN)
        print("Children")
        temp = []
        pop = []
        for c in children:
            temp.append(decoding(c, epsN))
        print(temp)
        for i in range(len(temp)):
            pop.append([temp[i],func(temp[i])])
        # draw_plot(pop)
        mut_children = mutation(children, mutation_n)
        temp = []
        for i in range(len(mut_children)):
            mut_children[i] = decoding(mut_children[i], epsN)
            temp.append(mut_children[i])
        print("Mutated")
        population = generate_population(mut_children)
        print(temp)
        # draw_plot(population)
        
        t += 1
    return population

ind = starting_individuals(left_edge,right_edge,N)
P = generate_population(ind)
result = new_gen(P, T)
draw_plot(result)
