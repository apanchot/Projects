from random import uniform, randint, choices
from copy import deepcopy

from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import EncodingDataType
from cifo.problem.population import Population
from cifo.algorithm.n_queens import NQueens
import numpy as np
import collections
from iteround import saferound
from cifo.problem.solution import LinearSolution
###################################################################################################
# INITIALIZATION APPROACHES
###################################################################################################

# (!) REMARK:
# Initialization signature: <method_name>( problem, population_size ):

# -------------------------------------------------------------------------------------------------
# Random Initialization 
# -------------------------------------------------------------------------------------------------
def initialize_randomly( problem, population_size ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 
    """
    solution_list = []

    i = 0
    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        s = problem.build_solution()
        
        # check if the solution is admissible
        while not problem.is_admissible( s ):
            s = problem.build_solution()
        
        s.id = [0, i]
        i += 1
        problem.evaluate_solution ( s )
        
        solution_list.append( s )

    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list )
    
    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Hill Climbing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Hill Climbing
# Remark: remember, you will need a neighborhood functin for each problem
def initialize_using_hc( problem, population_size ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 
    """
    solution_list = []

    i = 0
    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        
        s = problem.build_solution()
        
        # check if the solution is admissible
        
        # generate a local maximum using Hill Climbing 
        s = problem.hill_climbing(s)

        s.id = [0, i]
        i += 1
        
        
        solution_list.append( s )
    
    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list )
    
    return population


# -------------------------------------------------------------------------------------------------
# Initialization using Simulated Annealing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Hill Climbing
# Remark: remember, you will need a neighborhood functin for each problem
def initialize_using_sa( problem, population_size ):
    pass

def initialize_nqueens(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    print("start")
    solution_list = []
    # generate a population of admissible solutions (individuals)
    a=NQueens(data=problem.decision_variabless['data'], n_pop=population_size)
    sols=a.search()

    for i in range(0, population_size):
        permutation=sols[i]
        s = LinearSolution(
            representation=permutation,
            encoding_rule=problem._encoding_rule
        )
        s.id = [0, i]
        problem.evaluate_solution(s)
        solution_list.append(s)

    population = Population(
        problem=problem,
        maximum_size=population_size,
        solution_list=solution_list)
    print("done")
    return population

###################################################################################################
# SELECTION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# class RouletteWheelSelection
# -------------------------------------------------------------------------------------------------
# TODO: implement Roulette Wheel for Minimization
class RouletteWheelSelection:
    """
    Main idea: better individuals get higher chance
    The chances are proportional to the fitness
    Implementation: roulette wheel technique
    Assign to each individual a part of the roulette wheel
    Spin the wheel n times to select n individuals

    REMARK: This implementation does not consider minimization problem
    """
    def select(self, population, objective, params):
        """
        select two different parents using roulette wheel
        """
        index1 = self._select_index(population = population, objective = objective)
        index2 = index1
        
        while index2 == index1:
            index2 = self._select_index( population = population, objective = objective )

        return population.get( index1 ), population.get( index2 )


    def _select_index(self, population, objective ):

        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for solution in population.solutions:
            if (objective == "Maximization"):
                total_fitness += solution.fitness
            else:
                total_fitness += (1/solution.fitness)


        # spin the wheel
        wheel_position = uniform( 0, 1 )

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions :
            if (objective == "Maximization"):
                stop_position += (solution.fitness / total_fitness)
            else:
                stop_position += ((1/solution.fitness) / total_fitness)
            if stop_position > wheel_position :
                break
            index += 1    

        return index    

        
# -------------------------------------------------------------------------------------------------
# class RankSelection
# -------------------------------------------------------------------------------------------------
class RankSelection:
    """
    Rank Selection sorts the population first according to fitness value and ranks them. Then every chromosome is allocated selection probability with respect to its rank. Individuals are selected as per their selection probability. Rank selection is an exploration technique of selection.
    """
    def select(self, population, objective, params):
        # Step 1: Sort / Rank
        population = self._sort( population, objective )

        # Step 2: Create a rank list [0, 1, 1, 2, 2, 2, ...]
        rank_list = []

        for index in range(0, len(population)):
            for _ in range(0, index + 1):
                rank_list.append( index )

       #  print(f" >> rank_list: {rank_list}")

        # Step 3: Select solution index
        index1 = randint(0, len( rank_list )-1)
        index2 = index1
        
        while index2 == index1:
            index2 = randint(0, len( rank_list )-1)

        return population.get( rank_list [index1] ), population.get( rank_list[index2] )

    
    def _sort( self, population, objective ):

        if objective == ProblemObjective.Maximization:
            for i in range (0, len( population )):
                for j in range (i, len (population )):
                    if population.solutions[ i ].fitness > population.solutions[ j ].fitness:
                        swap = population.solutions[ j ]
                        population.solutions[ j ] = population.solutions[ i ]
                        population.solutions[ i ] = swap
                        
        else:    
            for i in range (0, len( population )):
                for j in range (i, len (population )):
                    if population.solutions[ i ].fitness < population.solutions[ j ].fitness:
                        swap = population.solutions[ j ]
                        population.solutions[ j ] = population.solutions[ i ]
                        population.solutions[ i ] = swap

        return population

# -------------------------------------------------------------------------------------------------
# class TournamentSelection
# -------------------------------------------------------------------------------------------------
class TournamentSelection:  
    """
    """
    def select(self, population, objective, params):
        tournament_size = 2
        if "Tournament-Size" in params:
            tournament_size = params[ "Tournament-Size" ]

        index1 = self._select_index( objective, population, tournament_size )    
        index2 = index1
        
        while index2 == index1:
            index2 = self._select_index( objective, population, tournament_size )

        return population.solutions[ index1 ], population.solutions[ index2 ]


    def _select_index(self, objective, population, tournament_size ): 
        
        index_temp      = -1
        index_selected  = randint(0, population.size - 1)

        if objective == ProblemObjective.Maximization: 
            for _ in range( 0, tournament_size ):
                index_temp = randint(0, population.size - 1 )

                if population.solutions[ index_temp ].fitness > population.solutions[ index_selected ].fitness:
                    index_selected = index_temp
        elif objective == ProblemObjective.Minimization:
            for _ in range( 0, tournament_size ):
                index_temp = randint(0, population.size - 1 )

                if population.solutions[ index_temp ].fitness < population.solutions[ index_selected ].fitness:
                    index_selected = index_temp            

        return index_selected         

###################################################################################################
# CROSSOVER APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint crossover
# -------------------------------------------------------------------------------------------------
def singlepoint_crossover( problem, solution1, solution2):
    singlepoint = randint(0, len(solution1.representation)-1)
    #print(f" >> singlepoint: {singlepoint}")

    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    return offspring1, offspring2    

# -------------------------------------------------------------------------------------------------
# Singlepoint crossover for PIP
# -------------------------------------------------------------------------------------------------
def singlepoint_crossover_pip ( problem, solution1, solution2):
    singlepoint = randint(0, len(solution1.representation)-1)
    #print(f" >> singlepoint: {singlepoint}")

    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]
    
    o1 = np.asarray(offspring1.representation)
    o2 = np.asarray(offspring2.representation)

    o1_sum = np.sum(o1)
    o2_sum = np.sum(o2)

    if o1_sum != 0:
        o1 = np.divide(o1, o1_sum)
    if o2_sum != 0:
        o2 = np.divide(o2, o2_sum)
    
    o1 = np.asarray(saferound(o1, 0)) #https://pypi.org/project/iteround/
    o2 = np.asarray(saferound(o2, 0)) 
    
    o1 = o1.astype(int)
    o2 = o2.astype(int)
    
    offspring1.representation = o1.tolist()
    offspring2.representation = o2.tolist()

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Two point crossover for PIP
# -------------------------------------------------------------------------------------------------
def two_point_crossover_pip ( problem, solution1, solution2):
    point1 = randint(0, len(solution1.representation)-2)
    point2 = randint(point1, len(solution1.representation)-1)

    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(point1, point2):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]
    
    o1 = np.asarray(offspring1.representation)
    o2 = np.asarray(offspring2.representation)

    o1_sum = np.sum(o1)
    o2_sum = np.sum(o2)

    if o1_sum != 0:
        o1 = np.divide(o1, o1_sum)
    if o2_sum != 0:
        o2 = np.divide(o2, o2_sum)

    o1 = np.asarray(saferound(o1, 0)) #https://pypi.org/project/iteround/
    o2 = np.asarray(saferound(o2, 0)) 
    
    o1 = o1.astype(int)
    o2 = o2.astype(int)
    
    offspring1.representation = o1.tolist()
    offspring2.representation = o2.tolist()

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Partially Mapped Crossover
# -------------------------------------------------------------------------------------------------

def pmx_crossover_aux(a,b):
    points = list(np.random.choice(range(1,len(a)), 2, replace=False))
    points.sort()
    point1, point2 = points[0], points[1]
    
    conflict_b = a[point1 : point2]
    conflict_a = b[point1 : point2]
    
    child_a = [-1]*len(a)
    child_b = [-1]*len(b)
    
    child_a[point1 : point2] = b[point1 : point2]
    child_b[point1 : point2] = a[point1 : point2]
    
    mapping_a = {child_a[i] : child_b[i] for i in range(point1,point2)}
    mapping_b = {v: k for k, v in mapping_a.items()}
    
    for i in range(0,point1):
        if a[i] not in conflict_a:
            child_a[i] = a[i]
        if b[i] not in conflict_b:
            child_b[i] = b[i]
    for i in range(point2,len(a)):
        if a[i] not in conflict_a:
            child_a[i] = a[i]
        if b[i] not in conflict_b:
            child_b[i] = b[i]
    
    for i in range(0,len(a)):
        if child_a[i] == -1:
            temp = mapping_a[a[i]]
            while temp in conflict_a:
                temp = mapping_a[temp]
            child_a[i] = temp
    for i in range(0,len(b)):
        if child_b[i] == -1:
            temp = mapping_b[b[i]]
            while temp in conflict_b:
                temp = mapping_b[temp]
            child_b[i] = temp

    return child_a, child_b

def pmx_crossover( problem, solution1, solution2):
    offspring1 = deepcopy(solution1) 
    offspring2 = deepcopy(solution2)

    offspring1.representation, offspring2.representation = pmx_crossover_aux(offspring1.representation,offspring2.representation)

    return offspring1, offspring2 
    
# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------

def cycle_xover_aux(a,b):       
    # The inputs can be strings or lists and should have the same size.
    # Converting the inputs to lists.
    new_a = list(a)
    new_b = list(b)    
    
    # Creating a dictionary that contains as many cycles as the number of elements 
    # of the lists.
    dct = {}
    for i in range(len(new_a)):
        dct['cycle_%s' % (i+1)] = []
    
    # Assigning each element to the corresponding cycle.
    # Some cycles can be empty.
    # The cycles are the keys and the genes are the values.
    for key in dct:
        if (len(new_a) > 0) and (len(new_b) > 0):
            dct[key] = [new_a[0], new_b[0]]
            i = new_a.index(dct[key][-1])
            while (new_b[i] not in dct[key]):
                dct[key].append(new_b[i])
                i = new_a.index(dct[key][-1])
            dct[key] = list(dict.fromkeys(dct[key]))
            new_a = [x for x in new_a if x not in set(dct[key])]
            new_b = [x for x in new_b if x not in set(dct[key])]

    # Inverting the logic to have the genes as the keys and the cycles as the values
    inv = dict() 
    for key in dct: 
        for item in dct[key]:
            if item not in inv: 
                inv[item] = key
            else: 
                inv[item].append(key) 
                
    # Creating a list that contains for each position of the original chromossome
    # the cycle related to it.
    cycles = [inv[k] for k in list(a)]
    
    all_cycles = list(dict.fromkeys(cycles))

    # Creating the offspring
    parent_a = list(a)
    parent_b = list(b) 
    child_a = [0]*len(parent_a)
    parent = parent_b
    for i in all_cycles:
        if (parent == parent_b):
            parent = parent_a
        else:
            parent = parent_b
        for j in dct[i]:
            child_a[parent.index(j)] = parent[parent.index(j)]
            
    child_b = [0]*len(parent_b)
    parent = parent_a
    for i in all_cycles:
        if (parent == parent_b):
            parent = parent_a
        else:
            parent = parent_b
        for j in dct[i]:
            child_b[parent.index(j)] = parent[parent.index(j)]

    return child_a, child_b
    
def cycle_crossover( problem, solution1, solution2):
    offspring1 = deepcopy(solution1) 
    offspring2 = deepcopy(solution2)

    offspring1.representation, offspring2.representation = cycle_xover_aux(offspring1.representation,offspring2.representation)

    return offspring1, offspring2 

# -------------------------------------------------------------------------------------------------
# Order 1 Crossover
# -------------------------------------------------------------------------------------------------

def order_crossover_aux(a,b):
    points = list(np.random.choice(range(1,len(a)), 2, replace=False))
    points.sort()
    point1, point2 = points[0], points[1]
    
    child_a = [-1]*len(a)
    child_a[point1 : point2] = a[point1 : point2]
    conflict_a = a[point1 : point2]
    mapping_a = [i for i in b[point2:] + b[:point2] if i not in conflict_a] 
    diff=len(a)-point2
    child_a[point2:] = mapping_a[:diff]
    child_a[:point1] = mapping_a[diff:]
    
    child_b = [-1]*len(b)
    child_b[point1 : point2] = b[point1 : point2]
    conflict_b = b[point1 : point2]
    mapping_b = [i for i in a[point2:] + a[:point2] if i not in conflict_b] 
    diff=len(b)-point2
    child_b[point2:] = mapping_b[:diff]
    child_b[:point1] = mapping_b[diff:]
    
    return child_a, child_b

def order_crossover( problem, solution1, solution2):
    offspring1 = deepcopy(solution1) 
    offspring2 = deepcopy(solution2)

    offspring1.representation, offspring2.representation = order_crossover_aux(offspring1.representation,offspring2.representation)

    return offspring1, offspring2 

# -------------------------------------------------------------------------------------------------
# Edge Crossover
# -------------------------------------------------------------------------------------------------

def edge_crossover_aux(a,b):
    a_aux = [a[-1]] + a + [a[0]]
    adj_a = {a_aux[i]:[a_aux[i-1], a_aux[i+1]] for i in range(1, len(a_aux)-1)}
    
    b_aux = [b[-1]] + b + [b[0]]
    adj_b = {b_aux[i]:[b_aux[i-1], b_aux[i+1]] for i in range(1, len(b_aux)-1)}
    
    adj_aux = {i:adj_a[i] + adj_b[i] for i in list(adj_a.keys())}
    adj = {i:list(np.unique(adj_aux[i])) for i in list(adj_aux.keys())}
    
    common_aux = {i:dict(collections.Counter(adj_aux[i])) for i in adj_aux.keys()}  
    common = {k:[] for k, v in common_aux.items() for j in v.keys() if v[j] > 1}
    for k in common.keys():
        for v in common_aux[k].keys():
            if common_aux[k][v] > 1:
                common[k].append(v)
    
    child = []
    non_used = a.copy()
    
    
    len_global = lambda x: {i:len(x[i]) for i in x}
    len_local = lambda x: {i:len_global(adj)[i] for i in adj[x]}
    min_neighbors = lambda x: [i for i in adj[x] if len_local(x)[i] == minimum]
    
    
    current = np.random.choice(a)
    child.append(current)
    non_used.remove(current)
    
    while (len(child) < len(a)):
        for i in a:
            if current in adj[i]:
                adj[i].remove(current)
        for i in common.keys():        
            if current in common[i]:
                common[i].remove(current)
        
        if (current in common and len(common[current]) != 0):
            current = np.random.choice(common[current])
            child.append(current)
            non_used.remove(current)
        else:
            if len(adj[current]) == 0:
                current = np.random.choice(non_used) 
                child.append(current)
                non_used.remove(current)
            else:
                len_global(adj)
                len_local(current)
                minimum = min(len_local(current).values())
                min_neighbors(current)
                current = np.random.choice(min_neighbors(current))
                child.append(current)
                non_used.remove(current)
       
    return child

def edge_crossover( problem, solution1, solution2):
    parent1 = deepcopy(solution1) 
    parent2 = deepcopy(solution2)
    offspring1 = deepcopy(solution1) 
    offspring2 = deepcopy(solution2)

    offspring1.representation = edge_crossover_aux(parent1.representation,parent2.representation)
    offspring2.representation = edge_crossover_aux(parent1.representation,parent2.representation)

    return offspring1, offspring2 

# -------------------------------------------------------------------------------------------------
# Non-wrapping Order Crossover
# -------------------------------------------------------------------------------------------------

def non_wrapping_order_crossover_aux(parent_a, parent_b):
    
    a = parent_a.copy()
    b = parent_b.copy()
    
    points = list(np.random.choice(range(1,len(a)), 2, replace=False))
    points.sort()
    point1, point2 = points[0], points[1]
   
    holes_a = b[point1:point2].copy()
    holes_b = a[point1:point2].copy()
    
    for i in range(len(a)):
        if a[i] in holes_a:
           a[i]='*' 
    for i in range(len(b)):
        if b[i] in holes_b:
           b[i]='*'  
    
    a = [i for i in a[:] if i != '*'] + [i for i in a[:] if i == '*']
    b = [i for i in b[:] if i != '*'] + [i for i in b[:] if i == '*']
    
    a = [a[i] for i in range(point1)] + holes_a + [a[i] for i in range(point1, point1 + len(a) - point2)] 
    b = [b[i] for i in range(point1)] + holes_b + [b[i] for i in range(point1, point1 + len(a) - point2)] 
    
    return a,b

def non_wrapping_order_crossover( problem, solution1, solution2):
    offspring1 = deepcopy(solution1) 
    offspring2 = deepcopy(solution2)

    offspring1.representation, offspring2.representation = non_wrapping_order_crossover_aux(offspring1.representation,offspring2.representation)

    return offspring1, offspring2 

# -------------------------------------------------------------------------------------------------
# Arithmetic Crossover
# -------------------------------------------------------------------------------------------------
# Used on the Portfolio Investment Problem (PIP)
# the offsprings are arithmetical combinations of the parents

def pipArithmeticCrossover(problem, solution1, solution2):
    alpha = np.random.uniform(0,1)
    
    p1 = np.asarray(solution1.representation)
    p2 = np.asarray(solution2.representation)

    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #solution2.clone()
    
    o1 = alpha*p1 + (1-alpha)*p2
    o2 = alpha*p2 + (1-alpha)*p1
    
    o1 = np.asarray(saferound(o1, 0)) #https://pypi.org/project/iteround/
    o2 = np.asarray(saferound(o2, 0)) 
    
    o1 = o1.astype(int)
    o2 = o2.astype(int)
    
    offspring1.representation = o1.tolist()
    offspring2.representation = o2.tolist()
    
    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Heuristic Crossover
# -------------------------------------------------------------------------------------------------
# Used on the Portfolio Investment Problem (PIP)
#the second offspring will be the arithmetical combination with the best fitness
#from (with adaptation):
#https://www.researchgate.net/publication/286952225_A_heuristic_crossover_for_portfolio_selection

def pipHeuristicCrossover(problem, solution1, solution2):
    alpha = np.random.uniform(0,1)

    p1 = np.asarray(solution1.representation)
    p2 = np.asarray(solution2.representation)

    if solution1.fitness >= solution2.fitness:
        offspring1 = deepcopy(solution1)
    else:
        offspring1 = deepcopy(solution2)

    x = alpha*p1 + (1-alpha)*p2
    x = saferound(x, 0)
    y = alpha*p2 + (1-alpha)*p1
    y = saferound(y, 0)

    offspring2 = deepcopy(solution2)
    offspring3 = deepcopy(solution2)

    for i in range(len(x)):
        x[i]=int(x[i])
        y[i] = int(y[i])

    offspring2.representation = x
    offspring3.representation = y

    problem.evaluate_solution(offspring2)
    problem.evaluate_solution(offspring3)
    
    if offspring2.fitness >= offspring3.fitness:
        return offspring1, offspring2
    else:
        return offspring1, offspring3

###################################################################################################
# MUTATION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint mutation
# -----------------------------------------------------------------------------------------------
def single_point_mutation( problem, solution):
    singlepoint = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")

    encoding    = problem.encoding

    if encoding.encoding_type == EncodingDataType.choices :
        try:
            temp = deepcopy( encoding.encoding_data )

            temp.pop( solution.representation[ singlepoint ] )

            gene = temp[0]
            if len(temp) > 1 : gene = choices( temp )  

            solution.representation[ singlepoint ] = gene

            return solution
        except:
            print('(!) Error: singlepoint mutation encoding.data issues)' )     

    # return solution           

# -------------------------------------------------------------------------------------------------
# Swap mutation
# -----------------------------------------------------------------------------------------------
# -------------- INSERT MUTATION --------------
def insert_mutation( problem, solution):

    pt1 = np.random.randint(len(solution.representation))
    pt2 = pt1
    while pt1 == pt2:
        pt2 = np.random.randint(len(solution.representation))

    if pt1 > pt2:
        pt1, pt2 = pt2, pt1

    saved_val = solution.representation[pt2]
    for j in range(pt2- pt1 - 1):
        solution.representation[pt2 - j] = solution.representation[pt2 - j - 1]
    solution.representation[pt1 + 1] = saved_val
    return solution
# -------------- SWAP MUTATION --------------
def swap_mutation( problem, solution):
    pt1 = np.random.randint(len(solution.representation))
    pt2 = pt1
    while pt1 == pt2:
        pt2 = np.random.randint(len(solution.representation))

    solution.representation[pt1], solution.representation[pt2] = solution.representation[pt2], solution.representation[pt1]
    return solution
# -------------- SWAP MUTATION FOR PIP --------------
# This swap guarantees that at least one mutation point has a gene different from zero
def swap_mutation_pip( problem, solution):
    pt1 = np.random.randint(len(solution.representation))
    while solution.representation[pt1] == 0:
        pt1 = np.random.randint(len(solution.representation))
    pt2 = pt1
    while pt1 == pt2:
        pt2 = np.random.randint(len(solution.representation))

    solution.representation[pt1], solution.representation[pt2] = solution.representation[pt2], solution.representation[pt1]
    return solution

# -------------- INVERSION MUTATION --------------
def inversion_mutation( problem, solution):
    pt1 = np.random.randint(len(solution.representation))
    pt2 = pt1
    while pt1 == pt2:
        pt2 = np.random.randint(len(solution.representation))

    if pt1 > pt2:
        pt1, pt2 = pt2, pt1

    leng = int(pt2- pt1)
    # print(pt,leng)
    fliparray = np.zeros(leng + 1)
    for j in range(leng + 1):
        fliparray[leng - j] = solution.representation[pt1 + j]
    for j in range(leng + 1):
        solution.representation[pt1 + j] = int(fliparray[j])
    return solution
# -------------- SCRAMBLE MUTATION --------------

def scramble_mutation( problem, solution):
    pt1 = np.random.randint(len(solution.representation))
    pt2 = pt1
    while pt1 == pt2:
        pt2 = np.random.randint(len(solution.representation))

    if pt1 > pt2:
        pt1, pt2 = pt2, pt1

    leng = pt2 - pt1
    # print(pt,leng)
    fliparray = np.zeros(leng + 1,dtype=int)
    for j in range(leng + 1):
        fliparray[j] = solution.representation[pt1 + j]
    np.random.shuffle(fliparray)
    for j in range(leng + 1):
        solution.representation[pt1 + j] = fliparray[j]
    return solution

###################################################################################################
# REPLACEMENT APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Standard replacement
# -----------------------------------------------------------------------------------------------
def standard_replacement(problem, current_population, new_population ):
    return deepcopy(new_population)

# -------------------------------------------------------------------------------------------------
# Elitism replacement
# -----------------------------------------------------------------------------------------------
def elitism_replacement(problem, current_population, new_population ):


    if problem.objective == ProblemObjective.Minimization :
        if current_population.fittest.fitness < new_population.fittest.fitness :
           new_population.solutions[0] = current_population.solutions[-1]

    elif problem.objective == ProblemObjective.Maximization :
        if current_population.fittest.fitness > new_population.fittest.fitness :
           new_population.solutions[0] = current_population.solutions[-1]

    return deepcopy(new_population)


def elitism_replacement_2(problem, current_population, new_population):
    if problem.objective == ProblemObjective.Minimization:
        fit=new_population.fittest.fitness
        j=0
        for i in range(len(current_population.solutions)):
            if current_population.solutions[-i-1].fitness < fit:
                new_population.solutions[j] = current_population.solutions[-i-1]
                j+=1
            else:
                break

    elif problem.objective == ProblemObjective.Maximization:
        if current_population.fittest.fitness > new_population.fittest.fitness:
            new_population.solutions[i] = current_population.solutions[len(new_population.solutions)-i]

    return deepcopy(new_population)

def elitism_replacement_3(problem, current_population, new_population):
    if problem.objective == ProblemObjective.Minimization:
        for i in range(int(len(new_population.solutions)/2)):
            new_population.solutions[i]=current_population.solutions[-i-1]

    elif problem.objective == ProblemObjective.Maximization:
        if current_population.fittest.fitness > new_population.fittest.fitness:
            new_population.solutions[i] = current_population.solutions[len(new_population.solutions)-i]

    return deepcopy(new_population)

def elitism_replacement_4(problem, current_population, new_population):
    if problem.objective == ProblemObjective.Minimization:

        j=0
        for i in range(len(current_population.solutions)):
            if current_population.solutions[-i-1].fitness < new_population.solutions[-i-1].fitness:
                new_population.solutions[j] = current_population.solutions[-i-1]
                j+=1
            else:
                break

    elif problem.objective == ProblemObjective.Maximization:
        if current_population.fittest.fitness > new_population.fittest.fitness:
            new_population.solutions[i] = current_population.solutions[len(new_population.solutions)-i]

    return deepcopy(new_population)
