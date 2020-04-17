import tsp
import math
import random
import matplotlib.pyplot as plt

#All GAs return [final pop, score per gen]

#modified provided code
def GA_crossover_search(fname, max_iter, prev_pop):

    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    n = len(city_locs)

    curr_gen = prev_pop


    scores = []
    print(f'crossover_search("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):
        # copy the top 50% of the population to the next generation, and for the rest randomly 
        # cross-breed pairs
        scores.append(curr_gen[0][0])
        top_half = [p[1] for p in curr_gen[:int(n/2)]]
        next_gen = top_half[:]
        while len(next_gen) < pop_size:
            s = random.choice(top_half)
            t = random.choice(top_half)
            first, second = tsp.pmx(s, t)
            next_gen.append(first)
            next_gen.append(second)

        next_gen = next_gen[:pop_size]

        # create the next generation of (score, permutations) pairs
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    # print(f'... crossover_search("{fname}", max_iter={max_iter}, pop_size={pop_size})')
    # print()
    # print(f'After {max_iter} generations of {pop_size} permutations, the best is:')
    # print(f'score = {curr_gen[0][0]}')
    # print(curr_gen[0][1])
    assert tsp.is_good_perm(curr_gen[0][1])

    return curr_gen,scores

def GA_mutate_search(fname, max_iter, prev_pop):

    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    # n = len(city_locs)

    curr_gen = prev_pop
    
    scores = []
    print(f'mutate_search("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):
        # put best permutation from curr_gen into next generation unchanged
        scores.append(curr_gen[0][0])
        best_curr_gen = curr_gen[0][1]
        next_gen = [best_curr_gen]

        # the rest of the next generation is filled with random swap-mutations
        # of best_curr_gen
        for j in range(pop_size-1):
            perm = best_curr_gen[:]  # make a copy of best_curr_gen
            tsp.do_rand_swap(perm)       # randomly swap two cities
            next_gen.append(perm)    # add it to next_gen

        # create the next generation of (score, permutations) pairs
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    # print(f'... mutate_search("{fname}", max_iter={max_iter}, pop_size={pop_size})')
    # print()
    # print(f'After {max_iter} generations of {pop_size} permutations, the best is:')
    # print(f'score = {curr_gen[0][0]}')
    # print(curr_gen[0][1])
    assert tsp.is_good_perm(curr_gen[0][1])

    return curr_gen,scores

def GA_rw_pmx(fname, max_iter, prev_pop):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    
    scores = []
    print(f'rw_pmx("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = selection_rw(curr_gen)

        #gen childern -> next_gen
        next_gen = []
        for pair in parents:
            #print(pair)
            first, second = tsp.pmx(pair[0], pair[1])
            next_gen.append(first)
            next_gen.append(second)

        #mutate childern
        next_gen = single_mutation(next_gen)

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores

def GA_rw_pmx(fname, max_iter, prev_pop):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    
    scores = []
    print(f'rw_pmx("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = selection_rw(curr_gen)

        #gen childern -> next_gen
        next_gen = []
        for pair in parents:
            #print(pair)
            first, second = tsp.pmx(pair[0], pair[1])
            next_gen.append(first)
            next_gen.append(second)

        #mutate childern
        next_gen = single_mutation(next_gen)

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores
    
# uses roulette wheel selection, returns n/2 pairs of parents
def selection_rw(curr_gen):

    n = len(curr_gen)

    #only consider top quater
    curr_gen = curr_gen[0:math.ceil(n/4)]
    n_cut = len(curr_gen)
    tot_fit = 0
    for i in range(n_cut):
        tot_fit +=curr_gen[i][0]


    #compute probability
    p = []
    sum_p = 0
    for i in range(n_cut):
        p_i = (1/(n_cut-1))*(1-((curr_gen[i][0])/tot_fit))
        sum_p += p_i
        p.append(sum_p)    
    p[-1] = 1.0

    #select parents
    sel = []
    for i in range(n):
        x = random.random()
        j = 0
        while p[j] < x:
            j += 1
        sel.append(j)

    #pair parents
    parents = []
    for i in range(0,n,2):
        j_a = sel[i]
        j_b = sel[i+1]
        parents.append((curr_gen[j_a][1],curr_gen[j_b][1]))

    return parents

def selection_sq(curr_gen):
    n = len(curr_gen)
    sq = math.ceil(n/2)

    parents = []
    for i in range(sq):
        for j in range(sq):
            parents.append((curr_gen[i][1],curr_gen[j][1]))
    
    return parents[:sq]

def selection_top(curr_gen):
    n = len(curr_gen)

    parents = []
    for i in range(math.ceil(n/2)):
        parents.append((curr_gen[0][1],curr_gen[i][1]))

    return parents[:math.ceil(n/2)]

#returns mutated pop
def single_mutation(next_gen):
    for lst in next_gen:
        n = len(lst)
        i, j = random.randrange(n), random.randrange(n)
        lst[i], lst[j] = lst[j], lst[i]  # swap lst[i] and lst[j]
    return next_gen

def load_pop(fname):

    #load previous pop 
    #expected name n_x.txt where n is the number of cities and x is save number
    #expected space sperated values 
    # premFname = "20_0.txt"
    # prevPerm = []
    # f = open(premFname, encoding = 'utf-8')
    # for line in f:
    #     prevPerm = line.split(' ')
    # prevPerm = [ x for x in prevPerm if x.isdigit() ]
    # prevPerm = [int(i) for i in prevPerm] 
    return []

def save_pop(fname):

    #save best perm
    #increments t0 n_(x+1).txt
    # x_new = int(premFname[premFname.find('_') + 1]) + 1
    # s = premFname.split('_')
    # s[1] = "_"+ str(x_new) + ".txt"
    # premFname = "".join(s)
    # print("saving "+ premFname)
    # f_w = open(premFname, 'w+')
    # for p in bestPerm:
    #     f_w.write(str(p) + " ")
    # f_w.close()
    return []

def test_all(fname, max_iter, prev_pop):
    fin_pop_pmx, scores_pmx = GA_standard_pmx(fname, max_iter, prev_pop)
    # fin_pop_cs, scores_cs = GA_crossover_search(fname, max_iter, prev_pop)
    # fin_pop_ms, scores_ms = GA_mutate_search(fname, max_iter, prev_pop)

    # plt.plot(scores_cs, label = "crossover search")
    # plt.plot(scores_ms, label = "mutation search")
    plt.plot(scores_pmx, label = "standard with PMX")

    plt.ylabel('Score')
    plt.xlabel("Generation")
    plt.legend()
    plt.show()

def gen_rand_pop(fname, n_pop):

    city_locs = tsp.load_city_locs(fname)
    n = len(city_locs)
    curr_gen = [tsp.rand_perm(n) for i in range(n_pop)]
    curr_gen = [(tsp.total_dist(p, city_locs), p) for p in curr_gen]
    curr_gen.sort()
    assert len(curr_gen) == n_pop

    return curr_gen


if __name__ == '__main__':

    
    n_pop = 100
    max_iter = 1000
    citesFname = "cities20.txt"

    prev_pop = gen_rand_pop(citesFname, n_pop)
    test_all(citesFname, max_iter, prev_pop)
    