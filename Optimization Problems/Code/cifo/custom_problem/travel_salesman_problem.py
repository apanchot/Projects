from copy import deepcopy
from random import choice, randint

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

import numpy as np

tsp_encoding_rule = {
    "Size"         : -1, # It must be defined by the size of DV (Number of products)
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [0,0], # must be defined by the data
    "Data Type"    : ""
}


# REMARK: There is no constraint

# -------------------------------------------------------------------------------------------------
# TSP - Travel Salesman Problem
# -------------------------------------------------------------------------------------------------
class TravelSalesmanProblem( ProblemTemplate ):
    """
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints , encoding_rule = tsp_encoding_rule):
        """
        """
        # optimize the access to the decision variables
        # ...
        self.decision_variabless=decision_variables
        self._distances = []
        if "data" in decision_variables:
            self._distances = decision_variables["data"]

        encoding_rule["Data"] = list(range(len(self._distances)))       
        
        encoding_rule["Size"] = len(self._distances)

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables = decision_variables, 
            constraints = constraints, 
            encoding_rule = encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "TSP Problem"
        
        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Minimization

    # Build Solution for Knapsack Problem
    #----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        """
        permutation = list(np.random.permutation(self._encoding_rule["Data"]))

        solution = LinearSolution(
            representation = permutation, 
            encoding_rule = self._encoding_rule
        )

        return solution

    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristic 
        """
        """
        return True

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        """
        dist=[ self._distances[solution.representation[i]][solution.representation[i+1]] for i in range(0, len(self.encoding_rule["Data"])-1)]
        dist.append( self._distances[solution.representation[-1]][solution.representation[0]])
        fitness = sum(dist)

        solution.fitness = fitness

        return solution    


    def hill_climbing( self, solution ): 
        """
        Find a local maximum given an initial random solution
        """
        best_solution = self.evaluate_solution(solution)
        
        max_iterations = 100
        iterations = 0
        changed = True
        
        while (iterations < max_iterations) and (changed == True):
            
            changed = False
        
            neighbors = self.neighborhood_function(best_solution.representation)
            for i in range(len(neighbors)):
                new_solution = LinearSolution(
                                representation = neighbors[i], 
                                encoding_rule = self._encoding_rule
                            )
                new_solution = self.evaluate_solution (new_solution)
                if new_solution.fitness < best_solution.fitness:
                    best_solution = new_solution
                    changed = True
            
            iterations +=1

        return best_solution

    def neighborhood_function(self, x):
        x = list(x)
        neighbors = []
        for i in range(len(x)):
            y = x.copy()
            if (i<len(x)-1):
                y[i], y[i+1] = y[i+1], y[i]
            else:
                y[i], y[0] = y[0], y[i]
            neighbors.append(y)                   
        return neighbors

# -------------------------------------------------------------------------------------------------
# OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def tsp_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    pass