from algorithm.genetic_algorithm import GeneticAlgorithm
from problem.problem_template import ProblemObjective
from problem.knapsack_problem import KnapsackProblem, knapsack_decision_variables_example, knapsack_constraints_example
from algorithm.ga_operators import RouletteWheelSelection, RankSelection, TournamentSelection, singlepoint_crossover
from util.terminal import Terminal, FontColor
from random import randint


def print_population( population ):
    _ = 0
    for solution in population:
        print(f" i: {_} - {solution.representation} - fitness: {solution.fitness}" )
        _ += 1

# Knapsack Problem
# -------------------------------------------------------------------------------------------------
knapsack_problem_instance = KnapsackProblem( 
    decision_variables = knapsack_decision_variables_example,
    constraints = knapsack_constraints_example
)


ga1 = GeneticAlgorithm( 
    problem_instance = knapsack_problem_instance,
    params = {
        "Population-Size" : 15
    } )
ga1._initialize_randomly()

Terminal.clear()
Terminal.print_box( ["Population Initialization"], font_color = FontColor.Yellow)

print_population( population = ga1._population)

# Parent Selection - Roulette Wheel
# -------------------------------------------------------------------------------------------------
Terminal.print_box( ["Parent Selection - Roulette wheel"], font_color = FontColor.Yellow)
rws = RouletteWheelSelection()
parent1, parent2 = rws.select( 
    population = ga1._population , 
    objective  = knapsack_problem_instance.objective, 
    params = {} 
)

print( str(parent1) )
print( str(parent2) )

print( "Parent Selection")

# Parent Selection - Rank Selection
# -------------------------------------------------------------------------------------------------
Terminal.print_box( ["Parent Selection - Rank Selection"], font_color = FontColor.Yellow)

rbs = RankSelection()

population = rbs._sort(     
    population = ga1._population , 
    objective = knapsack_problem_instance.objective, #ProblemObjective.Minimization
)

print( "> SORT:")
print_population( population )

parent1, parent2 = rbs.select( 
    population = ga1._population , 
    objective = knapsack_problem_instance.objective, 
    params = {} 
 )
print( "> SELECTED:")
print( str(parent1) )
print( str(parent2) )

# Parent Selection - Rank Selection
# -------------------------------------------------------------------------------------------------
Terminal.print_box( ["Parent Selection - Tournament Selection"], font_color = FontColor.Yellow)

ts = TournamentSelection()

parent1, parent2 = ts.select( 
    population = ga1._population , 
    objective = knapsack_problem_instance.objective, 
    params = {} 
 )
print( "> SELECTED:")
print( str(parent1) )
print( str(parent2) )

# Parent Selection - Rank Selection
# -------------------------------------------------------------------------------------------------
Terminal.print_box( ["Single-point Crossover"], font_color = FontColor.Yellow)
offspring1, offspring2 = singlepoint_crossover( parent1, parent2 )
print( "> Parents:")
print( str(parent1) )
print( str(parent2) )

o1_ok = knapsack_problem_instance.is_admissible(offspring1)
o2_ok = knapsack_problem_instance.is_admissible(offspring2) 
print( f" Offspring1 is admissible: {o1_ok} ")
print( f" Offspring2 is admissible: {o2_ok} ")

knapsack_problem_instance.evaluate_solution( offspring1 )
knapsack_problem_instance.evaluate_solution( offspring2 )

print( "> Offsprings:")
print( str(offspring1), f"is admissible: {o1_ok} " )
print( str(offspring2), f"is admissible: {o2_ok} " )

# Crossover
# -------------------------------------------------------------------------------------------------





# Mutation
# -------------------------------------------------------------------------------------------------




# Replacement
# -------------------------------------------------------------------------------------------------

#def print_solution( solution ):

def singlepoint_crossover( solution1, solution2, problem):
    """
    Inputs:
    @solution1 : parent 1 
    @solution2 : parent 2
    return : one or two offspring
    """
    
    size = len(solution1.representation)

    singlepoint = randint(0,size)

    offspring1 = solution1.clone()
    offspring2 = solution2.clone()

    for i in range(singlepoint, size):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]
    
    print(f'singlepoint = {singlepoint}')

    print( f'Parent    1: {solution1}' )
    print( f'Parent    2: {solution2}' )
    print( f'Offspring 1: {offspring1}' )
    print( f'Offspring 2: {offspring2}' )

    return offspring1, offspring2




def singlepoint_mutation( solution ):
    pass