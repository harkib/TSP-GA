import tsp
import csv
import math
import numpy
import random
import matplotlib.pyplot as plt
import concurrent.futures

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
    n = len(city_locs)

    curr_gen = prev_pop
    curr_gen.sort()

    scores = []
    print(f'mutate_search("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):
        # put best permutation from curr_gen into next generation unchanged
        scores.append(curr_gen[0][0])
        best_curr_gen = curr_gen[0][1]
        next_gen = [best_curr_gen]

        # the rest of the next generation is filled with random swap-mutations
        # of best_curr_gen
        # for j in range(pop_size-1):
        #     perm = best_curr_gen[:]  # make a copy of best_curr_gen
        #     tsp.do_rand_swap(perm)       # randomly swap two cities
        #     next_gen.append(perm)    # add it to next_gen

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            futures = {executor.submit(mutate_thread_full, best_curr_gen): j for j in range(pop_size-1)}
            for future in concurrent.futures.as_completed(futures):
                next_gen.append(future.result())  

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

def GA_standard(fname, max_iter, prev_pop,selction_func, crossover_func, mutation_func):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    
    scores = []
    print(f'{selction_func},{crossover_func},{mutation_func}("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = selction_func(curr_gen)

        #gen childern -> next_gen
        next_gen = []
        for pair in parents:
            #print(pair)
            first, second = crossover_func(pair)
            next_gen.append(first)
            next_gen.append(second)

        #mutate childern
        next_gen = mutation_func(next_gen)
        #next_gen[-1] = curr_gen[0][1]

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores

def GA_rw_pmx(fname, max_iter, prev_pop):
    
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    keep = math.floor(pop_size/5)
    keep =0
    assert keep <= pop_size

    scores = []
    print(f'rw_pmx("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = []
        for i in range(0,keep,2):
            parents.append((curr_gen[i][1],curr_gen[i+1][1]))
          
        parents.extend(selection_rw(curr_gen,1))
           
        #gen childern -> next_gen
        next_gen = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            futures = {executor.submit(tsp.pmx, pair[0], pair[1]): pair for pair in parents[:pop_size]}
            for future in concurrent.futures.as_completed(futures):
                first, second = future.result()
                next_gen.append(first)
                next_gen.append(second)

        next_gen = next_gen[:pop_size]
        # next_gen = []
        # for pair in parents:
        #     #print(pair)
        #     first, second = tsp.pmx(pair[0], pair[1])
        #     next_gen.append(first)
        #     next_gen.append(second)

        #mutate childern
        next_gen = single_mutation(next_gen)
        #next_gen[-1] = curr_gen[0][1]

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores

def GA_sq_pmx(fname, max_iter, prev_pop):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    
    scores = []
    print(f'rq_pmx("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = selection_sq(curr_gen)

        #gen childern -> next_gen
        next_gen = []
        for pair in parents:
            #print(pair)
            first, second = tsp.pmx(pair[0], pair[1])
            next_gen.append(first)
            next_gen.append(second)

        #mutate childern
        next_gen = single_mutation(next_gen)
        #next_gen[-1] = curr_gen[0][1]

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores

def GA_top_pmx(fname, max_iter, prev_pop):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    
    scores = []
    print(f'top_pmx("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = selection_top(curr_gen)

        #gen childern -> next_gen
        next_gen = []
        for pair in parents:
            #print(pair)
            first, second = tsp.pmx(pair[0], pair[1])
            next_gen.append(first)
            next_gen.append(second)

        #mutate childern
        next_gen = single_mutation(next_gen)
        #next_gen[-1] = curr_gen[0][1]

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores

def GA_top_NWOX(fname, max_iter, prev_pop):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    
    scores = []
    print(f'top_NWOX("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = selection_top(curr_gen)

        #gen childern -> next_gen
        next_gen = []
        for pair in parents:
            #print(pair)
            first, second = crossover_NWOX(pair)
            next_gen.append(first)
            next_gen.append(second)

        #mutate childern
        next_gen = single_mutation(next_gen)
        #next_gen[-1] = curr_gen[0][1]

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores

def GA_sq_NWOX(fname, max_iter, prev_pop):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    
    scores = []
    print(f'sq_NWOX("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = selection_sq(curr_gen)

        #gen childern -> next_gen
        next_gen = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            futures = {executor.submit(crossover_NWOX, pair): pair for pair in parents}
            for future in concurrent.futures.as_completed(futures):
                first, second = future.result()
                next_gen.append(first)
                next_gen.append(second)

        # for pair in parents:
        #     first, second = crossover_NWOX(pair)
        #     next_gen.append(first)
        #     next_gen.append(second)

        #mutate childern
        next_gen = single_mutation(next_gen)
        #next_gen[-1] = curr_gen[0][1]

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores

def GA_rw_NWOX(fname, max_iter, prev_pop):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    curr_gen = prev_pop
    keep = math.floor(pop_size/5)
    keep = 0
    assert keep <= pop_size
    
    scores = []
    print(f'rw_nwox("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    for i in range(max_iter):

        #record historic scores
        scores.append(curr_gen[0][0])

        #select parents
        parents = selection_rw(curr_gen,1)

        #gen childern -> next_gen
        next_gen = []
        for i in range(keep):
            next_gen.append(curr_gen[i][1])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            futures = {executor.submit(crossover_NWOX, pair): pair for pair in parents}
            for future in concurrent.futures.as_completed(futures):
                first, second = future.result()
                next_gen.append(first)
                next_gen.append(second)

        next_gen = next_gen[:pop_size]

        #mutate childern
        next_gen = single_mutation(next_gen)
        #next_gen[-1] = curr_gen[0][1]

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen.sort()

    return curr_gen, scores

def GA_tabo(fname, max_iter, prev_pop):
    pop_size = len(prev_pop)
    city_locs = tsp.load_city_locs(fname)
    n = len(city_locs)
    curr_gen = prev_pop
    keep = math.floor(pop_size/5)
    keep = 0
    stalled = 0
    assert keep <= pop_size
    
    scores = []
    scores.append(curr_gen[0][0])
    print(f'tabo("{fname}", max_iter={max_iter}, pop_size={pop_size}) ...')
    
    #generate tabo data
    print("Generating Tabo crossover data...", end='')
    d_map=[]
    dc=[]
    closest = []
    B =2

    for i in range(0,n):
        d_i_to_js = []
        for j in range(0,n):
            d_i_to_js.append(tsp.city_dist(i+1,j+1,city_locs))
        d_map.append(d_i_to_js)
    
    for i in range(0,n):
        dc.append(sum(d_map[i])/(B*(n-1)))

    for i in range(0,n):
        d_js = d_map[i][:]
        order = [x+1 for x in numpy.argsort(d_js)]
        closest.append(order)

    print("...Done")
    delta = 100
    delta_lim = .1

    for i in range(max_iter):
        best = curr_gen[0]

        #select parents
        parents = selection_rw(curr_gen, 2, True)

        #gen childern -> next_gen
        next_gen = []
        for i in range(keep):
            next_gen.append(curr_gen[i][1])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            futures = {executor.submit(crossover_tabo,pair,d_map,dc,closest): pair for pair in parents}
            for future in concurrent.futures.as_completed(futures):
                first = future.result()
                next_gen.append(first)

        next_gen = next_gen[:pop_size]



        #mutate childern
        next_gen = chuck_mutation(next_gen)
        next_gen = single_mutation(next_gen)

        #score childern
        assert len(next_gen) == pop_size 
        curr_gen = [(tsp.total_dist(p, city_locs), p) for p in next_gen]
        curr_gen[-1] = best
        curr_gen.sort()

        #record historic scores
        scores.append(curr_gen[0][0])

        if len(scores) > 2:
            delta = scores[-2] -scores[-1]
        if delta < delta_lim:
            stalled += 1
        else:
            stalled = 0
        if stalled > 10:
            print("delta limit at itteration:", i)
            break


    return curr_gen, scores

# uses roulette wheel selection, returns n/2 pairs of parents
def selection_rw(curr_gen, divid = 2, n_out = False ):
    
    n = len(curr_gen)

    if n_out:
        n = 2*n
    

    #only consider top quater
    curr_gen = curr_gen[0:math.ceil(n/divid)]
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

def selection_top_n(curr_gen):
    n = len(curr_gen)

    parents = []
    for i in range(n):
        parents.append((curr_gen[0][1],curr_gen[i][1]))

    return parents[:n]

def crossover_NWOX(pair):
    #print(pair)
    p1 = pair[0]
    p2 = pair[1]
    c1 = []
    c2 = []

    n = len(p1)
    a = random.randint(0,n-1)
    b = random.randint(a,n-1)
    #print("ab; ", a,b)

    c1_mid = [elem for elem in p1[a:b] if elem not in p2[a:b]]

    c1_left = [elem for elem in p1[:a] if elem not in p2[a:b]]
    c1_lslide = c1_mid[:a-len(c1_left)]

    # print(c1_left,len(c1_left))
    # print(c1_lslide,len(c1_lslide))

    c1_right =[elem for elem in p1[b:] if elem not in p2[a:b]]
    c1_rslide = [elem for elem in c1_mid if elem not in c1_lslide]

    # print(c1_right,len(c1_right))  
    # print(c1_rslide,len(c1_rslide))

    
    # print(p2[a:b],len(p2[a:b]))
    c1.extend(c1_left+c1_lslide+p2[a:b]+c1_rslide+c1_right)
    # print(c1,len(c1))

    # print("-----c2------")

    c2_mid = [elem for elem in p2[a:b] if elem not in p1[a:b]]

    c2_left = [elem for elem in p2[:a] if elem not in p1[a:b]]
    c2_lslide = c2_mid[:a-len(c2_left)]

    # print(c2_left,len(c2_left))
    # print(c2_lslide,len(c2_lslide))

    c2_right =[elem for elem in p2[b:] if elem not in p1[a:b]]
    c2_rslide = [elem for elem in c2_mid if elem not in c2_lslide]

    # print(c2_right,len(c1_right))
    # print(c2_rslide,len(c2_rslide))

    # print(p1[a:b],len(p1[a:b]))
    c2.extend(c2_left+c2_lslide+p1[a:b]+c2_rslide+c2_right)
    # print(c2,len(c2))

    return c1,c2

#returns one child per pair
def crossover_tabo(pair,d_map,dc,closest):
    n = len(pair[0])
    p1 = pair[0]
    p2 = pair[1]
    cp = random.randint(1,n) #city previuos
    child = [cp]
    for i in range(1,n):

        try:
            j1,j2 = p1[i] -1, p2[i] -1
            cp = child[i-1] -1
            if (d_map[cp][j1] <= d_map[cp][j2]) and (j1+1 not in child) and (d_map[cp][j1] < dc[cp]):
                child.append(j1+1)
            elif (d_map[cp][j1] > d_map[cp][j2]) and (j2+1 not in child) and (d_map[cp][j2] < dc[cp]):
                child.append(j2+1)
            else:
                for city in closest[cp]:
                    if city not in child:
                        child.append(city)
                        break
        except:
            print(i,n,j1,j2,cp)
            print(len(child))
            print(len(dc))
            print(len(d_map))
            print(len(closest))
            print(len(p1))
            print(len(p2))
            break

    return child

#returns mutated pop
def single_mutation(next_gen,tolerance =.1):
    mu_gen = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
        futures = {executor.submit(mutate_thread, perm[:], tolerance): perm for perm in next_gen}
        for future in concurrent.futures.as_completed(futures):
            mu_gen.append(future.result())  

    return mu_gen

def chuck_mutation(next_gen):
    mu_gen = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
        futures = {executor.submit(chunk_mutation_thread, perm[:]): perm for perm in next_gen}
        for future in concurrent.futures.as_completed(futures):
            mu_gen.append(future.result())  

    return mu_gen

def chunk_mutation_thread(perm):

    if random.uniform(0,1) < .3:
        n = len(perm)
        a = random.randint(0,n)
        b = random.randint(a,n)
        left = perm[0:a]
        mid = perm[a:b]
        right = perm[b:n]
        return right + mid + left
    return perm

def mutate_thread(perm, tolerance=.1):
    copy = perm[:]
    if random.uniform(0,1) < tolerance:
        n = len(perm)   
        i, j = random.randrange(n), random.randrange(n)
        copy[i], copy[j] = copy[j], copy[i] 
    return copy

def mutate_thread_full(perm):
    copy = perm[:]
    n = len(copy)
    i, j = random.randrange(n), random.randrange(n)
    copy[i], copy[j] = copy[j], copy[i] 
    return copy

def load_pop(fname):

    #load previous pop 
    #expected name n_x.txt where n is the number of cities and x is save number
    #expected space sperated values 
    gen = []
    f = open(fname, encoding = 'utf-8')
    for line in f:
        ls = line.split(',"')
        perm = ls[1].replace('[','').replace(']','').replace('\n','').replace("\"",'').split(', ')
        #print(perm)
        perm = [int(i) for i in perm] 
        gen.append((float(ls[0]),perm))

    return gen

def save_pop(prevfname, pop):

    #save best perm
    #increments t0 n_(x+1).txt
    x_new = int(prevfname[prevfname.find('_') + 1]) + 1
    s = prevfname.split('_')
    s[1] = "_"+ str(x_new) + ".txt"
    newfname = "".join(s)
    print("saving "+ newfname)
    with open(newfname,'w',newline='', encoding = "utf-8") as file_y:
        csv_y = csv.writer(file_y)
        csv_y.writerows(pop)
    return []

def test_all(fname, max_iter, prev_pop): 

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futureGA_rw_NWOX = executor.submit(GA_rw_NWOX,fname, max_iter, prev_pop)
        #futureGA_sq_NWOX = executor.submit(GA_sq_NWOX,fname, max_iter, prev_pop)
        futureGA_top_NWOX = executor.submit(GA_top_NWOX,fname, max_iter, prev_pop)
        futureGA_rw_pmx = executor.submit(GA_rw_pmx,fname, max_iter, prev_pop)
        #futureGA_sq_pmx = executor.submit(GA_sq_pmx,fname, max_iter, prev_pop)
        futureGA_top_pmx = executor.submit(GA_top_pmx,fname, max_iter, prev_pop)
        #futureGA_mutate_search = executor.submit(GA_mutate_search,fname, max_iter, prev_pop)
        pop_rw_nwox, scores_rw_nwox = futureGA_rw_NWOX.result()
        #pop_sq_nwox, scores_sq_nwox = futureGA_sq_NWOX.result()
        pop_top_nwox, scores_top_nwox = futureGA_top_NWOX.result()
        pop_rw_pmx, scores_rw_pmx = futureGA_rw_pmx.result()
        #pop_sq_pmx, scores_sq_pmx = futureGA_sq_pmx.result()
        pop_top_pmx, scores_top_pmx = futureGA_top_pmx.result()
        #pop_ms, scores_ms = futureGA_mutate_search.result()

        # pops = [pop_rw_nwox,pop_rw_nwox,pop_top_nwox,pop_rw_pmx,pop_sq_pmx,pop_top_pmx,pop_ms]
        # scores_all = [scores_rw_nwox,scores_sq_nwox,scores_top_nwox,scores_rw_pmx,scores_sq_pmx,scores_top_pmx,scores_ms]
        pops = [pop_rw_nwox,pop_top_nwox,pop_rw_pmx,pop_top_pmx]
        scores_all = [scores_rw_nwox,scores_top_nwox,scores_rw_pmx,scores_top_pmx]
        
    # # fin_pop_sq_nwox2, scores_sq_nwox2 = GA_standard(fname, max_iter, prev_pop,selection_sq, crossover_NWOX, single_mutation)
    # # plt.plot(scores_sq_nwox2, label = "SQ NWOX2")

    # # fin_pop_rw_nwox, scores_rw_nwox = GA_rw_NWOX(fname, max_iter, prev_pop)
    # # plt.plot(scores_rw_nwox, label = "RW NWOX")

    # # fin_pop_sq_nwox, scores_sq_nwox = GA_sq_NWOX(fname, max_iter, prev_pop)
    # plt.plot(scores_sq_nwox, label = "SQ NWOX")

    # # fin_pop_top_nwox, scores_top_nwox = GA_top_NWOX(fname, max_iter, prev_pop)
    # plt.plot(scores_top_nwox, label = "top NWOX")

    # # fin_pop_rw_pmx, scores_rw_pmx = GA_rw_pmx(fname, max_iter, prev_pop)
    # plt.plot(scores_rw_pmx, label = "RW PMX")

    # # fin_pop_sq_pmx, scores_sq_pmx = GA_sq_pmx(fname, max_iter, prev_pop)
    # plt.plot(scores_sq_pmx, label = "SQ PMX")

    # # fin_pop_top_pmx, scores_top_pmx = GA_top_pmx(fname, max_iter, prev_pop)
    # plt.plot(scores_top_pmx, label = "Top PMX")

    # # fin_pop_cs, scores_cs = GA_crossover_search(fname, max_iter, prev_pop)
    # # plt.plot(scores_cs, label = "crossover search")

    # # fin_pop_ms, scores_ms = GA_mutate_search(fname, max_iter, prev_pop)
    # plt.plot(scores_ms, label = "mutation search")
    

    # plt.ylabel('Score' + fname)
    # plt.xlabel("Generation")
    # plt.legend()
    # plt.show()

    #return best pop 
    best_pop = pops[0]
    best_scores = scores_all[0]
    for i,pop in enumerate(pops):
        if best_pop[0][0] > pop[0][0]:
            best_pop = pop
            best_scores = scores_all[i]

    return best_pop, best_scores
    

def gen_rand_pop(fname, n_pop):

    city_locs = tsp.load_city_locs(fname)
    n = len(city_locs)
    curr_gen = [tsp.rand_perm(n) for i in range(n_pop)]
    curr_gen = [(tsp.total_dist(p, city_locs), p) for p in curr_gen]
    curr_gen.sort()
    assert len(curr_gen) == n_pop

    return curr_gen


if __name__ == '__main__':

    
    n_pop = 50
    n_concur = 3
    n_cycles = 10
    max_iter = 50
    citesFname = "cities1000.txt"
    prevPermFname = "1000_0.txt"

    pops = [gen_rand_pop(citesFname, n_pop) for i in range(n_concur)]
    #pops[0] = load_pop(prevPermFname)
    total_scores = []
    for i in range(n_cycles):
        print("Cycle:", i)

        #run 5 random tabo GAs
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_concur) as executor:
            futures = {executor.submit(GA_tabo,citesFname, max_iter, pop): pop for pop in pops}
            new_pops = []
            new_scores = []
            for future in concurrent.futures.as_completed(futures):
                tabo_pop, scores_tabo  = future.result()
                new_pops.append(tabo_pop)
                new_scores.append(scores_tabo)

        #rerun with best prev and 4 new random 
        best_pop = new_pops[0]
        best_scores = new_scores[0]
        for i,pop in enumerate(new_pops):
            print(pop[0][0])
            if best_pop[0][0] > pop[0][0]:
                best_pop = pop
                best_scores = new_scores[i]

        pops = [gen_rand_pop(citesFname, n_pop) for i in range(n_concur)]
        pops[0] = best_pop
        total_scores.extend(best_scores)

        #save best
        save_pop(prevPermFname,best_pop)


    plt.plot(total_scores, label = "Tabo + with restarts")
    plt.ylabel('Score')
    plt.xlabel("Generation")
    plt.legend()
    plt.show()
    #test_all(citesFname, max_iter, inti)
    