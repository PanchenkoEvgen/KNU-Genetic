import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import math
import random
import numpy as np

N = 50 #population 
T = 50 #quantity of generations

left_edge = -4
right_edge = 4
bottom = -4
top = 4

epsN = 11

selection_n = N
mutation_n = 5

def func(x,y): 
    # return math.sin(3*math.pi*x) * math.sin(3*math.pi*y) + 0.5*math.sin(5*math.pi*x) * math.sin(5*math.pi*y)
    return 1.5 - 0.25 * (abs(x) + abs(y)) + 0.6 * math.cos(3*x + y) + 0.4 * math.sin(x*y) - 0.3 * abs(math.cos(2*x - y))

def func_np(x,y):
    # return np.sin(3*np.pi*x) * np.sin(3*np.pi*y) + 0.5*np.sin(5*np.pi*x) * np.sin(5*np.pi*y)
    return 1.5 - 0.25 * (np.abs(x) + np.abs(y)) + 0.6 * np.cos(3*x + y) + 0.4 * np.sin(x*y) - 0.3 * np.abs(np.cos(2*x - y))

def starting_individuals_old(l, r, b, t, n):
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


def starting_individuals(l, r, b, t, n):
    individuals = []
    for _ in range(n):
        rand_x = random.uniform(l, r)
        rand_y = random.uniform(b, t)
        
        individuals.append([rand_x, rand_y])
        
    return individuals

def generate_population(individuals):
    population = []
    for x,y in individuals:
        population.append([x,y, func(x,y)])
    return population

def encoding(individual, n):
    range_size = right_edge - left_edge 
    normalized_val = (individual - left_edge) / range_size

    normalized_val = max(0.0, min(1.0, normalized_val))

    scaled_val = normalized_val * pow(10, n)
    val_int = int(scaled_val)

    max_val = pow(10, n) - 1
    if val_int > max_val:
        val_int = max_val

    individual_genes = []

    for j in range(n):
        individual_genes.append(val_int % 10)
        val_int //= 10
        
    return individual_genes

def decoding(genes, n):
    val_int = 0

    for i in range(len(genes)):
        val_int += genes[i] * pow(10, i)
        
    normalized_val = val_int / pow(10, n)
    range_size = right_edge - left_edge
    real_val = normalized_val * range_size + left_edge
    
    return round(real_val, epsN)

def selection(population, num, bias=2.0):
    selected_individuals = []

    sorted_pop = sorted(population, key=lambda x: x[2])
    pop_len = len(sorted_pop)

    weights = [i + 1 for i in range(pop_len)]

    k = 3
    
    for _ in range(num):
        candidates = random.choices(sorted_pop, weights=weights, k=k)

        winner = max(candidates, key=lambda x: x[2])

        selected_individuals.append([winner[0][:], winner[1][:]])
        
    return selected_individuals

def crossover(individuals, num):
    new_individuals = []

    def get_fitness(ind):
        return func(decoding(ind[0], num), decoding(ind[1], num))

    for i in range(0, len(individuals), 2):
        if i + 1 >= len(individuals):
            new_individuals.append(individuals[i])
            break
            
        p1, p2 = individuals[i], individuals[i+1]

        c1, c2 = [[], []], [[], []]
        alpha = random.random()
        
        for idx in range(2):
            for j in range(num):
                val1 = alpha * p1[idx][j] + (1 - alpha) * p2[idx][j]
                val2 = (1 - alpha) * p1[idx][j] + alpha * p2[idx][j]
                
                c1[idx].append(max(0, min(9, int(round(val1)))))
                c2[idx].append(max(0, min(9, int(round(val2)))))

        family = [p1, p2, c1, c2]

        family.sort(key=get_fitness, reverse=True)

        new_individuals.extend(family[:2])
        
    return new_individuals

# def mutation(individuals, n):
#     for i in range(n):
#         random_ind = random.randint(0, len(individuals)-1)
#         random_xy = random.randint(0, 1)

#         genes = individuals[random_ind][random_xy]
#         genes_len = len(genes)

#         random_gen = int(random.triangular(0, genes_len - 1, 0))

#         current_digit = genes[random_gen]
#         if random.random() < 0.8:
#             step = random.choice([-1, 1])
#             new_digit = current_digit + step
#             new_digit = max(0, min(9, new_digit))
            
#         else:
#             new_digit = random.randint(0, 9)
            
#         individuals[random_ind][random_xy][random_gen] = new_digit
        
#     return individuals

def shotgun_mutation(individuals, n):
    used_i = []
    for i in range(n):
        random_ind = random.randint(0,len(individuals)-1)
        if random_ind not in used_i:
            random_num = random.uniform(-0.05,0.05)
            individuals[random_ind][0] += random_num
            random_num = random.uniform(-0.05,0.05)
            individuals[random_ind][1] += random_num

        else:
            i -= 1
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
    p = generate_population(starting_individuals_old(left_edge, right_edge,bottom, top, 200))
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
        #print(f"Gen {t}")
        temp = [[r[0], r[1]] for r in population]
        if t == 0:
            print(temp)
            result_plot(population)
        for i in range(len(population)):
            population[i][0] = encoding(population[i][0], epsN)
            population[i][1] = encoding(population[i][1], epsN)
        selected = selection(population, selection_n)
        if t == 0:  
            print("Selected")
            temp = []
            pop = []
            for s in selected:
                temp.append([decoding(s[0], epsN),decoding(s[1], epsN)])
            pop = generate_population(temp)
            print(temp)
            result_plot(pop)
        children = crossover(selected, epsN)
        if t == 0:  
            print("Children")
            temp = []
            pop = []
            for s in children:
                temp.append([decoding(s[0], epsN),decoding(s[1], epsN)])
            pop = generate_population(temp)
            print(temp)
            result_plot(pop)
        for i in range(len(children)):
            children[i][0] = decoding(children[i][0], epsN)
            children[i][1] = decoding(children[i][1], epsN)
        if t != T-1:
            mut_children = shotgun_mutation(children, mutation_n)
            population = generate_population(mut_children)
        else:
            population = generate_population(children)
        # for i in range(len(mut_children)):
        #     mut_children[i][0] = decoding(mut_children[i][0], epsN)
        #     mut_children[i][1] = decoding(mut_children[i][1], epsN)
        if t == 0:  
            print("Mutated")
            temp = [[r[0], r[1]] for r in population]
            print(temp)
            result_plot(population)  
        t += 1
        
    return population

def print_results(population):
    
    population.sort(key=lambda x: x[2], reverse=True)

    max_value = population[0][2]

    winners = []
    for ind in population:
        if abs(ind[2] - max_value) < 0.001:
            winners.append(ind)
        else:
            break

    print("\n" + "="*35)
    print(f"СТАТИСТИКА ЗБІЖНОСТІ")
    print("="*35)
    print(f"Global Maximum: {max_value:.5f}")
    print(f"Кількість точок, які дають максимальне значення: {len(winners)} з {len(population)}")
    
    '''
    print("-" * 35)
    print(f"{'No.':<4} | {'X':<10} | {'Y':<10} | {'f(x_max,y_max)':<10}")
    print("-" * 35)
    
    for i, p in enumerate(winners):
        x_fmt = round(p[0], epsN)
        y_fmt = round(p[1], epsN)
        print(f"{i+1:<4} | {x_fmt:<10} | {y_fmt:<10} | {p[2]:.5f}")
            
    print("="*35 + "\n")
    '''

ind = starting_individuals(left_edge,right_edge,bottom, top, N)
P = generate_population(ind)
result = new_gen(P, T)
print_results(result)
result_plot(result)
