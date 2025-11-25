import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import math
import random
import numpy as np

N = 6 #population 
T = 10 #quantity of generations

left_edge = -4
right_edge = 4
bottom = -4
top = 4

eps = 0.0001
epsN = int(-math.log10(eps))

selection_n = 36
mutation_n = 3

def func(x,y): 
    return 1.5 - 0.25 * (abs(x) + abs(y)) + 0.6 * math.cos(3*x + y) + 0.4 * math.sin(x*y) - 0.3 * abs(math.cos(2*x - y))

def func_np(x,y):
    return 1.5 - 0.25 * (np.abs(x) + np.abs(y)) + 0.6 * np.cos(3*x + y) + 0.4 * np.sin(x*y) - 0.3 * np.abs(np.cos(2*x - y))

def starting_individuals(l, r, b, t, n):
    individuals = []
    x = []
    y = []
    step = abs(r-l)/n
    while (l <= r):
        x.append(l)
        l += step
    step = abs(t-b)/n
    while (b <= t):
        y.append(b)
        b += step
    for i in range(len(x)):
        for j in range(len(y)):
            individuals.append([x[i],y[j]])
    return individuals

def generate_population(individuals):
    population = []
    for x,y in individuals:
        population.append([x,y, func(x,y)])
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
    population.sort(key=lambda x:x[2], reverse = True)
    s_max = population[0][2]
    n = 0
    while n < num:
        random_individual = random.uniform(0,s_max)
        individual_index = 0
        for x, y, survavability in population:
            if n >= num:
                break
            if survavability >= random_individual:
                selected_individuals.append([x,y])
                n += 1
            else:
                break
    return selected_individuals

def crossover(individuals, num):
    new_individuals = []
    for i in range(0, len(individuals), 2):
        parent1 = individuals[i]
        parent2 = individuals[i+1]
        child1 = [[],[]]
        child2 = [[],[]]
        for j in range(int(num/2)):
            child1[0].append(parent2[0][j])
            child1[1].append(parent2[1][j])
            child2[0].append(parent1[0][j])
            child2[1].append(parent1[1][j])
        for j in range(int(num/2), num):
            child1[0].append(parent1[0][j])
            child1[1].append(parent1[1][j])
            child2[0].append(parent2[0][j])
            child2[1].append(parent2[1][j])
        new_individuals.append(child1)
        new_individuals.append(child2)
    return new_individuals

def mutation(individuals, n):
    for i in range(n):
        random_ind = random.randint(0,len(individuals)-1)
        random_xy = random.randint(0,1)
        random_gen = random.randint(0,len(individuals[random_ind])-1)
        random_num = random.randint(0,9)
        individuals[random_ind][random_xy][random_gen] = random_num
    return individuals

def draw_plot(result):
    x = [r[0] for r in result]
    y = [r[1] for r in result]
    t = [r[2] for r in result]
    p = generate_population(starting_individuals(left_edge, right_edge,bottom, top, 100))
    x_true = [r[0] for r in p]
    y_true = [r[1] for r in p]
    t_true = [r[2] for r in p]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    norm = Normalize(vmin=0, vmax=max(max(t),max(t_true)))
    p1 = axs[0].scatter(left_edge, bottom, c="white")
    p1 = axs[0].scatter(right_edge, top, c="white")
    p1 = axs[0].scatter(x, y, c=t, cmap="coolwarm", norm=norm)
    p2 = axs[1].scatter(x_true, y_true, c=t_true, cmap="coolwarm", norm=norm)

    # cbar = fig.colorbar(p2, ax=axs, location='right')
    plt.show()

def d3_plot(result):

    xr = [r[0] for r in result]
    yr = [r[1] for r in result]
    zr = [r[2] for r in result]

    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = func_np(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.scatter(xr, yr, zr, color='red', s=20)
    

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def result_plot(result):
    #3d
    xr = [r[0] for r in result]
    yr = [r[1] for r in result]
    zr = [r[2] for r in result]
    xD = np.linspace(-5, 5, 50)
    yD = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(xD, yD)
    Z = func_np(X, Y)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(xr, yr, zr, color='red', s=20)
    surf = ax1.plot_surface(X, Y, Z)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    #heat map
    x = [r[0] for r in result]
    y = [r[1] for r in result]
    t = [r[2] for r in result]
    p = generate_population(starting_individuals(left_edge, right_edge,bottom, top, 100))
    x_true = [r[0] for r in p]
    y_true = [r[1] for r in p]
    t_true = [r[2] for r in p]
    norm = Normalize(vmin=0, vmax=max(max(t),max(t_true)))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(x_true, y_true, c=t_true, cmap="coolwarm", norm=norm)
    ax2.scatter(x, y, c="green")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.show()


    

def new_gen(population, T):
    t = 0
    while (t < T):
        print(f"Gen {t}")
        temp = [[r[0], r[1]] for r in population]
        print(temp)
        # draw_plot(population)
        result_plot(population)
        for i in range(len(population)):
            population[i][0] = encoding(population[i][0], epsN)
            population[i][1] = encoding(population[i][1], epsN)
        selected = selection(population, selection_n)
        print("Selected")
        temp = []
        pop = []
        for s in selected:
            temp.append([decoding(s[0], epsN),decoding(s[1], epsN)])
        pop = generate_population(temp)
        print(temp)
        # draw_plot(pop)
        result_plot(pop)
        children = crossover(selected, epsN)
        print("Children")
        temp = []
        pop = []
        for s in children:
            temp.append([decoding(s[0], epsN),decoding(s[1], epsN)])
        pop = generate_population(temp)
        print(temp)
        # draw_plot(pop)
        result_plot(pop)
        mut_children = mutation(children, mutation_n)
        for i in range(len(mut_children)):
            mut_children[i][0] = decoding(mut_children[i][0], epsN)
            mut_children[i][1] = decoding(mut_children[i][1], epsN)
        population = generate_population(mut_children)
        print("Mutated")
        temp = [[r[0], r[1]] for r in population]
        print(temp)
        # draw_plot(population)
        result_plot(population)
        t += 1
    return population

ind = starting_individuals(left_edge,right_edge,bottom, top, N)
P = generate_population(ind)
result = new_gen(P, T)
# d3_plot(result)
# draw_plot(result)
result_plot(result)
