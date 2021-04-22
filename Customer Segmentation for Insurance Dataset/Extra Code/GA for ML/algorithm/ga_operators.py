from random import uniform, randint, choices

from problem.objective import ProblemObjective
from problem.solution import EncodingDataType
from problem.population import Population
from copy import deepcopy


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

###################################################################################################
# SELECTION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# class RouletteWheelSelection
# -------------------------------------------------------------------------------------------------
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
        index1 = self._select_index(population = population)
        index2 = index1

        while index2 == index1:
            index2 = self._select_index( population = population )

        return population.get( index1 ), population.get( index2 )


    def _select_index(self, population ):
        '''This is the roullete wheel itself'''
        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for solution in population.solutions:
            total_fitness += solution.fitness

        # spin the wheel
        wheel_position = uniform( 0, 1 )

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions :
            stop_position += (solution.fitness / total_fitness)
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

        print(f" >> rank_list: {rank_list}")

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
            self._tournament_size = tournament_size[ "Tournament-Size" ]

        index1 = self._select_index( population, tournament_size )
        index2 = index1

        while index2 == index1:
            index2 = self._select_index( population, tournament_size )

        return population[ index1 ], population[ index2 ]


    def _select_index(self, population, tournament_size ):
        '''the tournament selection it self'''
        index_temp      = -1
        index_selected  = randint(0, len( population )-1 )

        for _ in range( 0, tournament_size ):
            index_temp = randint(0, len( population )-1 )

            if population.get[ index_temp ].fitness > population.get[ index_selected ].fitness:
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

#try:
    temp = deepcopy( encoding['Data'][singlepoint] )

    temp.pop( solution.representation[ singlepoint ] )

    gene = temp[0]
    if len(temp) > 1 : gene = choices( temp )

    solution.representation[ singlepoint ] = gene

    return solution
#except:
    print('(!) Error: singlepoint mutation encoding.data issues)' )

    # return solution

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