import tsp

def GA_custom(cites, prevPerm, generations):
    bestPerm = prevPerm
    return bestPerm



if __name__ == '__main__':

    #load cites
    citesFname = "cities20.txt"
    city_locs = tsp.load_city_locs(citesFname)

    #load best previous perm 
    #expected name n_x.txt where n is the number of cities and x is teh perm number
    #expected space sperated values 
    premFname = "20_0.txt"
    prevPerm = []
    f = open(premFname)
    for line in f:
        prevPerm = line.split(' ')
    prevPerm = [int(i) for i in prevPerm] 
    
    #run GA
    bestPerm = GA_custom(citesFname, prevPerm, 1)

    #save best perm
    #increments t0 n_(x+1).txt
    x_new = int(premFname[premFname.find('_') + 1]) + 1
    s = premFname.split('_')
    s[1] = "_"+ str(x_new) + ".txt"
    premFname = "".join(s)
    print("saving "+ premFname)
    f_w = open(premFname, 'w')
    for p in bestPerm:
        f_w.write(str(p) + " ")
    f_w.close()

    #print score
    
    print(tsp.total_dist(bestPerm,city_locs))