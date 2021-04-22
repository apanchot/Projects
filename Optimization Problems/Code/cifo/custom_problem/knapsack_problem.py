# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Knapsack Problem
-------------------
Content

 ▶ class KnapsackProblem(ProblemTemplate)

 ▶ dv_mastermind_template = {}
 
 ▶ Knapsack_encoding1 = {} << Elements cannot be repeated
 
 ▶ mastermind_encoding2 = {} << Elements can be repeated
 
 ▶ def mastermind_get_neighbors

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0

"""
# -------------------------------------------------------------------------------------------------
from copy import deepcopy
from random import choice, randint

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding


knapsack_encoding_rule = {
    "Size"         : -1, # It must be defined by the size of DV (Number of products)
    "Is ordered"   : False,
    "Can repeat"   : True,
    "Data"         : [0,1],
    "Data Type"    : "Choices"
}

knapsack_decision_variables_example = {
    "Values"    : [15,16,13,9,24], #<< Number, Mandatory
    "Weights"   : [12,11,19,18,11], #<< Number, Mandatory
    "Item-Name" : ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"] # << String, Optional, The names of the items, can be used to present the solution.
}

knapsack_constraints_example = {
    "Max-Weight" : 30
}

class KnapsackProblem( ProblemTemplate ):
    """
    The goal of the knapsack problem is to pack a set of items, with given weights and values, into a "knapsack" with a maximum capacity. The objective  is to maximize the total value of the packed items.
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints , encoding_rule = knapsack_encoding_rule):
        """
            Knapsack Problem Template CONSTRUCTOR
        
            Parameters:

            @decision_variables
            
            Expected Decision Variables, so the dictionary must have the following keys and values of them must be lists:
            
            e.g:
            
            decision_variables_example = {

                "Values"    : [10,11,13,9,13,18,12,21,12,25], #<< Number, Mandatory
                
                "Weights"   : [12,11,16,9,9,15,11,20,13,23], #<< Number, Mandatory
                
                "Item-Name" : ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"] 
                # << String, Optional, The names of the items, can be used to present the solution.
            
            }
            
            @constraints
            
            The capacity (maximum weight of the knapsack)
            e.g.:
            
            constraints = {

                "Max-Weight" = 45 << Number, Mandatory - The capacity (maximum weight of the knapsack)
            
            } 

            @encoding_rule

            knapsack_encoding_rule = {
                
                "Size"         : -1, # This number must be redefined using the size of DV (Number of products contained in the instance of the problem)
                
                "Is ordered"   : False,
                
                "Can repeat"   : True,
                
                "Data"         : [0,1],
                
                "Data Type"    : "Choices"
            }
            
        """
        # optimize the access to the decision variables
        self._values = []
        if "Values" in decision_variables:
            self._values = decision_variables["Values"]

        self._weights = []
        if "Weights" in decision_variables:
            self._weights = decision_variables["Weights"]

        self._capacity = 0 # "Max-Weight"
        if "Max-Weight" in constraints:
            self._capacity = constraints["Max-Weight"]

        encoding_rule["Size"] = len( self._values )

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables = decision_variables, 
            constraints = constraints, 
            encoding_rule = encoding_rule
        )

        self._name = "Knapsack Problem"
        
        # Problem Objective
        self._objective = ProblemObjective.Maximization

    # Build Solution for Knapsack Problem
    #----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        Builds a linear solution for Knapsack with same size of decision variables with 0s and 1s.
        
        Where: 
            
            0 indicate that item IS NOT in the knapsack 
            
            1 indicate that item IS in the knapsack 
        """
        solution_representation = []
        encoding_data = self._encoding.encoding_data

        for _ in range(0, self._encoding.size):
            solution_representation.append( choice(encoding_data) )
        
        solution = LinearSolution(
            representation = solution_representation, 
            encoding_rule = self._encoding_rule
        )
        
        return solution

    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristic 
        """
        Check if the solution is admissible, considering the weight and capacity (maximum weight) 
        """
        weights = self._weights

        weight = 0
        for  i in range(0, len( weights )):
            if solution.representation[ i ] == 1:
                weight += weights[ i ]

        result = weight <= self._capacity
        return result 

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        Calculate the "value" of the products that are in the knapsack
        """
        values = self._values

        fitness = 0
        
        for  i in range(0, len( values )):
            if solution.representation[ i ] == 1:
                fitness += values[ i ]
        
        solution.fitness = fitness

        return solution        

# -------------------------------------------------------------------------------------------------
# Knapsack Neighborhood Function [get_neighbors()]
# -------------------------------------------------------------------------------------------------
def knapsack_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    """
    This is a bitflip approach to implement a neighborhood function for Knapsack
    """
    neighbors = []

    # Generate all neighbors considering a bit flip
    for position in range(0, len(solution.representation)):
        n = deepcopy(solution) # solution.clone()
        if n.representation[ position ]  ==  1 : 
            n.representation[ position ] = 0
        else: 
            n.representation[ position ] = 1
        
        neighbors.append(n)

    # return all neighbors
    if neighborhood_size == 0:
        return neighbors
    # return a RANDOM subset of all neighbors (in accordance with neighborhood size)    
    else:     
        subset_neighbors = []
        indexes = list( range( 0, len( neighbors ) ) )
        for _ in range(0, neighborhood_size):
            selected_index = choice( indexes )

            subset_neighbors.append( neighbors[ selected_index ] )
            indexes.remove( selected_index )

        return subset_neighbors    