# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
The0s1sProblem (Toy Example)
----------------------------

Content: 

▶ class The0s1sProblem(ProblemTemplate)

▶ def the0s1s_get_decimal_neighbors

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0
"""
# -------------------------------------------------------------------------------------------------
# import
from copy import deepcopy
from random import randint

from cifo.problem.problem_template import ProblemTemplate, ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
# C O D E
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# -------------------------------------------------------------------------------------------------
# Class: The0s1sProblem Template
# -------------------------------------------------------------------------------------------------
class The0s1sProblem(ProblemTemplate):
    """
    It is a single-objective maximization problem.

    Objective: find in an interval of numbers in the decimal system, which number in binary representation has the largest number of 1s. The number of 1s in the binary system will be used to calculate the fitness of the solution. Therefore, the best solution == biggest number of 1s (binary system).

    (!) Remark: It is a toy example just for instructional purposes. 
   
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables = { "Numbers" : [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ]}, constraints={}, encoding_rule = {}):
        """
        The 0s1s Problem Constructor

        Excepted Decision Variables
        {
            "Numbers" = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ]
        }
        """
        # encoding
        # the0s1s_encoding_rule = {
        #    "Size"         : 1,
        #    "Is ordered"   : False,
        #    "Can repeat"   : True,
        #    "Data"         : [ 1, 15 ],
        #    "Data Type"    : "Interval"
        # }
        
        # super constructor to keep the member variables of base class and define the encoding
        super().__init__(
            decision_variables = decision_variables, 
            constraints = {}, 
            encoding_rule = {
                "Size"         : 1,
                "Is ordered"   : False,
                "Can repeat"   : True,
                "Data"         : [ 1, 15 ],
                "Data Type"    : "Interval"
            }
        )

        # Name of the problem
        self._name = "The 0s and 1s Problem"

        # set the objective
        self._objective_function_list = [ self._objective_function1 ]
        self._objective_list          = [ ProblemObjective.Maximization ] 
    
    # Build Solution Function - build_solution()
    #----------------------------------------------------------------------------------------------
    def build_solution(self):       
        # empty linear solution
        solution_representation = []
        # size = self._encoding.size
        data = self._encoding.encoding_data

        max = data[-1] 
        min = data[0]

        solution_representation.append( randint( min, max) )
        
        solution = LinearSolution(representation = solution_representation, encoding_rule = self._encoding_rule)
        
        return solution    
    
    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristics 
        """
        The solution of mastermind problem does not have a constraint. So, it always will return True
        """
        return True 
    
    # Objective Function 1
    #----------------------------------------------------------------------------------------------
    def _objective_function1(self, solution, decision_variables, feedback = None):  
        fitness = 0

        binary_rep = decimal_to_binary( solution.representation )

        for digit in binary_rep:
            fitness += digit # fitness = fitness + digit
    
        return fitness
    


# -------------------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------------------
# helper funtion : binary_to_decimal
def binary_to_decimal( solution ):
    solution_str = ''
    
    for digit in solution:
        solution_str += str(digit)
    
    i = int(solution_str, 2) 

    return i

# helper function : binary_to_decimal
def decimal_to_binary ( solution ):
    binary_rep = list(bin(solution[0])[2:]) # bin(n).replace("0b","") 
    return [int(digit) for digit in binary_rep]

# -------------------------------------------------------------------------------------------------
# Neighborhood Functions
# -------------------------------------------------------------------------------------------------
def the0s1s_get_decimal_neighbors( solution, problem, neighborhood_size = 0 ): 
    
    numbers = problem.decision_variables["Numbers"]

    new_solution1 = deepcopy(solution)
    new_solution2 = deepcopy(solution)

    neighbors = []

    if solution.representation[0] == numbers[0]:
        new_solution1.representation[0] =  solution.representation[0] + 1
        neighbors.append( new_solution1 )
    elif solution.representation[0] == numbers[ -1 ]:
        new_solution1.representation[0] = solution.representation[0] - 1
        neighbors.append( new_solution1 )
    else:
        new_solution1.representation[0] = solution.representation[0] - 1
        new_solution2.representation[0] = solution.representation[0] + 1
        neighbors.append( new_solution1 )
        neighbors.append( new_solution2 )
    
    return neighbors



