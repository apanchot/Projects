# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Genetic  Meta-Heuristic
----------------------------

Content:

▶ class Genetic Algorithm (Incomplete version)

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0
"""
# -------------------------------------------------------------------------------------------------

# import
from algorithm.ga_operators import (
    initialize_randomly,
    RankSelection, RouletteWheelSelection, TournamentSelection,
    singlepoint_crossover,
    single_point_mutation,
    standard_replacement, elitism_replacement
)

from problem.population import Population
from random import random

from util.terminal import Terminal, FontColor

from copy import deepcopy


# default params
default_params = {
    "Population-Size" : 5,
    "Population-Initialization" : 'Initialize Random',
    "Tournament-Size" : 5,
    "Crossover-Probability" : 0.8,
    "Mutation-Probability" : 0.5

}

# -------------------------------------------------------------------------------------------------
# Genetic Algorithm
# REMARK: Incomplete Version
# -------------------------------------------------------------------------------------------------
class GeneticAlgorithm:
    """
    A genetic algorithm is a search heuristic that is inspired by Charles Darwin’s theory of
    natural evolution. This algorithm reflects the process of natural selection where the fittest
    individuals are selected for reproduction in order to produce offspring of the next generation.

    The process of natural selection starts with the selection of fittest individuals from a population.
    They produce offspring which inherit the characteristics of the parents and will be added to the
    next generation. If parents have better fitness, their offspring will be better than parents and
    have a better chance at surviving. This process keeps on iterating and at the end, a generation with
    the fittest individuals will be found.

    This notion can be applied for a search problem. We consider a set of solutions for a problem and
    select the set of best ones out of them.

    Six phases are considered in a genetic algorithm.

    1. Initial population
    2. Fitness function
    3. Selection
    4. Crossover
    5. Mutation
    6. Replacement
    """
    # Constructor
    # ---------------------------------------------------------------------------------------------
    def __init__(self, problem_instance, params = default_params):
        self._parse_params( params )

        self._problem_instance  = problem_instance

        self._population = None
        self._fittest    = None

    # search
    # ---------------------------------------------------------------------------------------------
    def search( self ):
        """
            Genetic Algorithm Search Algorithm
            1. Initial population

            2. Repeat n generations )(#1 loop )

                2.1. Repeat until generate the next generation (#2 loop )
                    1. Selection
                    2. Try Apply Crossover (depends on the crossover probability)
                    3. Try Apply Mutation (depends on the mutation probability)

                2.2. Replacement

            3. Return the best solution
        """
        problem         = self._problem_instance
        select          = self._selection_approach.select
        cross           = self._crossover_approach
        mutate          = self._mutation_approach
        replace         = self._replacement_approach
        is_admissible   = self._problem_instance.is_admissible

        #Terminal.print_box(['Genetic Algorithms - Search Started'], font_color = FontColor.Green)
        #print(f' - Crossover Probability : {self._crossover_probability}')
        #print(f' - Mutation  Probability : {self._mutation_probability}')

        self._generation = 0

        # 1. Initial population
        self._population = self._initialize( problem, self._population_size )

        #Terminal.print_line(message = f'\nGeneration # {0}')
        #print( f' > Fittest # { self._population.fittest.representation } - fitness: {self._population.fittest.fitness}' )
        self._population.sort()
        #for solution in self._population.solutions:
            #print(f' > {solution.id} - {solution.representation} - Fitness : {solution.fitness}')


        #2. Repeat n generations )(#1 loop )
        for generation in range( 1, self._number_of_generations ):

            new_population = Population( problem = problem, maximum_size = self._population_size, solution_list=[] )
            i = 0

            #Terminal.print_line(message = f'\nGeneration # {generation}')
            # 2.1. Repeat until generate the next generation (#2 loop )
            while new_population.has_space:
                #print('.', end = '')
                # 2.1.1. Selection
                parent1, parent2 = select( self._population, problem.objective, self._params )
                offspring1 = deepcopy(parent1) # parent1.clone()
                offspring2 = deepcopy(parent2) # parent2.clone()
                # 2.1.2. Try Apply Crossover (depends on the crossover probability)
                if self.apply_crossover:
                    offspring1, Offspring2 = cross(problem, parent1, parent2)
                    #print(offspring1,offspring2)
                    offspring1.id = [generation, i]
                    i += 1
                    offspring2.id = [generation, i]
                    i += 2
                    #print(f'XO')
                    #print(f'P1:        {parent1.id}-{parent1.representation}')
                    #print(f'P2:        {parent2.id}-{parent2.representation}')
                    #print(f'O1:        {offspring1.id}-{offspring1.representation}')
                    #print(f'O2:        {offspring2.id}-{offspring2.representation}')

                # 2.1.3. Try Apply Mutation (depends on the mutation probability)
                if self.apply_mutation:
                    offspring1 = mutate( problem, offspring1 )
                    #print(offspring1)
                    offspring1.id = [generation, i]
                    i += 1
                    #print(f'MU : O1    {offspring1.id}-{offspring1.representation}')
                if self.apply_mutation:
                    offspring2 = mutate( problem, offspring2 )
                    #print(offspring2)
                    offspring2.id = [generation, i]
                    i += 1
                    #print(f'MU : O2    {offspring2.id}-{offspring2.representation}')

                # add the offsprings in the new population (New Generation)
                if new_population.has_space and is_admissible( offspring1 ):
                    problem.evaluate_solution ( offspring1 )
                    new_population.solutions.append( offspring1 )
                    #print(f'Added O1 - {offspring1.id}-{offspring1.representation}')

                if new_population.has_space and is_admissible( offspring2 ):
                    problem.evaluate_solution( offspring2 )
                    new_population.solutions.append( offspring2 )
                    #print(f'Added O2 - {offspring2.id}-{offspring2.representation}')

            # 2.2. Replacement

            #print('\n # NEW GENERATION *** BEFORE *** REPLACEMENT')
            #fittest = new_population.fittest
            #print( f' > Fittest # id: {fittest.id} - { fittest.representation } - fitness: {fittest.fitness} - weight: {self.calc_weights( fittest )}' )
            #for solution in new_population.solutions:
            #    print(f' > {solution.id} - {solution.representation} - Fitness : {solution.fitness} - weight: {self.calc_weights( solution )}')

            #print(' # CURRENT GENERATION' )
            #fittest = self._population.fittest
            #print( f' > Fittest # id: {fittest.id} - { fittest.representation } - fitness: {fittest.fitness} - weight: {self.calc_weights( fittest )}' )
            #for solution in self._population.solutions:
            #    print(f' > {solution.id} - {solution.representation} - Fitness : {solution.fitness} - weight: {self.calc_weights( solution )}')

            self._population = replace(problem, self._population, new_population )

            #print(' # NEW GENERATION *** AFTER *** REPLACEMENT' )
            fittest = self._population.fittest
            #print( f' > Fittest # id: {fittest.id} - { fittest.representation } - fitness: {fittest.fitness}' )
            #for solution in self._population.solutions:
            #    print(f' > {solution.id} - {solution.representation} - Fitness : {solution.fitness} - weight: {self.calc_weights( solution )}')
        
        return fittest.id, fittest.representation, fittest.fitness
            #new_population = None


    @property
    def apply_crossover( self ):
        chance = random()
        return chance < self._crossover_probability

    @property
    def apply_mutation( self ):
        chance = random()
        return chance < self._mutation_probability

    # initialize
    # ---------------------------------------------------------------------------------------------
    def _parse_params( self, params ):
        self._params = params
                # set Genetic Algorithm Behavior
        # Initialization signature: <method_name>( problem, population_size ):
        self._initialize            = initialize_randomly
        self._selection_approach    = RouletteWheelSelection()
        self._crossover_approach    = singlepoint_crossover
        self._mutation_approach     = single_point_mutation
        self._replacement_approach  = elitism_replacement

        self._population_size = 10
        if "Population-Size" in params:
            self._population_size = params[ "Population-Size" ]

        self._tournament_size = 5
        if "Tournament-Size" in params:
            self._tournament_size = params[ "Tournament-Size" ]

        self._crossover_probability = 0.8
        if "Crossover-Probability" in params:
            self._crossover_probability = params[ "Crossover-Probability" ]

        self._mutation_probability = 0.5
        if "Mutation-Probability" in params:
            self._mutation_probability = params[ "Mutation-Probability" ]

        self._number_of_generations       = 5
        if "Number-of-Generations" in params:
            self._number_of_generations = params[ "Number-of-Generations" ]

    def calc_weights( self, solution ):
        weights = self._problem_instance.decision_variables["Weights"]

        weight = 0
        for  i in range(0, len( weights )):
            if solution.representation[ i ] == 1:
                weight += weights[ i ]

        return weight