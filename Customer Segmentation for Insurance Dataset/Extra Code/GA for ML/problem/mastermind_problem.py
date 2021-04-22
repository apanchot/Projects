# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Mastermind Problem
-------------------
Content

 ▶ class MastermindProblem(ProblemTemplate)

 ▶ dv_mastermind_template = {}
 
 ▶ mastermind_encoding1 = {} << Elements cannot be repeated
 
 ▶ mastermind_encoding2 = {} << Elements can be repeated
 
 ▶ def mastermind_get_neighbors

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0

"""
# -------------------------------------------------------------------------------------------------

# import
from problem.problem_template import ProblemTemplate, ProblemObjective
from problem.solution import LinearSolution, Encoding
from copy import deepcopy
from random import choice

# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
# C O D E
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# -------------------------------------------------------------------------------------------------
# Auxiliary variables - Encoding and Default Decision Variables 
# This DVs template can also be used in tests), but in the DVs are problem-specific, 
# so each problem instance will provide different values for the DVs.
#
# DVs - Decision Variables
#--------------------------------------------------------------------------------------------------

# template (example) decision variables for mastermind
dv_mastermind_template = {
    "Colors"        : ['Blue', 'Red', 'Yellow', 'Green', 'White', 'Magenta'],
    "Colors-Id"     : [1, 2, 3, 4, 5, 6],
    "Colors-Code"   : ["da_blue", "da_red", "da_yellow", "da_white", "da_magenta" ] # Dark_<color> Sty colors
}

# encoding 1: The elements cannot be repeated
mastermind_encoding1 = {
    "Size"         : 4,
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [], # in the constructor must get the data from the Dvs (as it is DV dependent)
    "Data Type"    : "Choices"
} 

# encoding 2: The element can be repeated
mastermind_encoding2 = {
    "Size"         : 4,
    "Is ordered"   : True,
    "Can repeat"   : True,
    "Data"         : [], # in the constructor must get the data from the Dvs (as it is DV dependent)
    "Data Type"     : "Choices" # MinMax
} 

# -------------------------------------------------------------------------------------------------
# Class: Mastermind Problem Template
# -------------------------------------------------------------------------------------------------
class MastermindProblem(ProblemTemplate):
    """
    Expected Decision Variables, so deliver:
    decision_variables = {
        "Color-Label"   : [...] << List of String
        "Color-Id"      : [...] << List of Numbers 
        "Colors-Code"   : [...] << List of Strings (Sty background colors)
    }
    
    if decision_variable IS EMPTY it will be used default_decision_variables defined internally for tests

    @ constraints = {} this problem does not have constraints
    @ encoding_rule = {} there two types of encoding defined for this problem: mastermind_encoding1 and mastermind_encoding2mastermind_encoding1 is the default if it was not defined an encoding it will be used
            
    """
    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables = [], constraints={}, encoding_rule = mastermind_encoding1):
        """
            Mastermind Problem Template CONSTRUCTOR

            Parameters:

            @ decision_variables
            Expected Decision Variables, so deliver:
            decision_variables = {
                "Color-Label" : [...] << String
                "Color-Id"    : [...] << Number 
                "Color-Code"  : [...] << String (Sty color)
            }

            if decision_variable IS EMPTY it will be used default_decision_variables defined internally for tests

            @ constraints = {} this problem does not have constraints

            @ encoding_rule = {} there two types of encoding defined for this problem: mastermind_encoding1 and mastermind_encoding2
            mastermind_encoding1 is the default if it was not defined an encoding it will be used
            
        """
        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
             decision_variables, 
             constraints, 
             encoding_rule)

        self._name = "Mastermind Problem"

        if not decision_variables: 
            # DEFAULT DECISION VARIABLE, if the problem instance does not define a specific decision variables
            # It will be used the default_decision_variables defined in this file
            self._decision_variables = dv_mastermind_template
        
        # Mastermind Problem will use "Color-Id" to create the encoding
        if "Colors-Id" in self._decision_variables:
            self._encoding_rule["Data"]  = self._decision_variables["Colors-Id"]
            self._encoding.encoding_data = self._decision_variables["Colors-Id"]

        # Problem Objective
        self._objective_function_list = [ self._objective_function1 ]
        self._objective_list          = [ ProblemObjective.Maximization ] 


    # Build Solution Function - build_solution()
    #----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        Builds a mastermind guess based on the configured encoding
        """
        # empty linear solution
        solution_representation = []
        size = self._encoding.size
        data = self._encoding.encoding_data

        # if elements can be repeated
        if self._encoding.can_repeat_elements:
            for _ in range(0, size):
                solution_representation.append( choice( data ) )

            solution = LinearSolution(representation = solution_representation, encoding_rule = self._encoding_rule)
            return solution
        # if elements cannot be repeated
        else:   
            encoding_data_temp = deepcopy( data )

            for _ in range(0, size):
                element = choice( encoding_data_temp )
                solution_representation.append( element )
                encoding_data_temp.remove( element ) 

            solution = LinearSolution(representation = solution_representation, encoding_rule = self._encoding_rule)
            return solution

    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristic
        """
        The solution of mastermind problem does not have a constraint. So, it always will return True
        """
        return True

    # Objective Function 1
    #----------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    # Objective Function Signature: objective_function(self, solution, decision_variables, feedback = None)
    def _objective_function1(self, solution, decision_variables, feedback = None): #<< use this signature in the sub classes, the meta-heuristics
        """
        Objective Function of Mastermind Problem,
        It calculates the fitness of a guess (candidate solution)
        """
        fitness = 0

        if feedback :
    
            computed_solution_position  = [False] * len(solution.representation)
            computed_feedback_position = [False] * len(solution.representation)

            for i in range( 0, len(solution.representation) ):
                if ( solution.representation[i] == feedback.representation[i] ):
                    fitness += 10
                    computed_solution_position[i] = True
                    computed_feedback_position[i] = True

            for i in range( 0, len(feedback.representation) ):
                if not computed_solution_position[i]:
                    for j in range( 0, len(solution.representation) ):
                        if not computed_feedback_position[j] and solution.representation[i] == feedback.representation[j]: 
                                fitness += 3 
                                computed_solution_position[i] = True
                                computed_feedback_position[j] = True
                                break               
            
            return fitness
        else: return 0 

# -------------------------------------------------------------------------------------------------
# Neighborhood Function [get_neighbors()]
# -------------------------------------------------------------------------------------------------

def mastermind_get_neighbors( solution, problem, neighborhood_size = 0 ):
    """
    Generate the neighbors of a solution in accordance with neighborhood structure defined in this function 
    solution  : 1, 3, 4, 3
    neighbors : *, 3, 4, 3 | 1, *, 4, 3 | 1, 3, *, 3 | 1, 3, 4, *
    """

    neighbors = []

    # if elements can be repeated
    if solution.encoding.can_repeat_elements:
        for i in range(0, len(solution.representation)):
            next_choices = deepcopy( solution.encoding.encoding_data )
            neighbor     = deepcopy( solution )
            
            next_choices.remove( solution.representation[i] ) # remove the current position only, to avoid repeat this element in this position
            
            neighbor.representation[i]  = choice( next_choices) # for DataType = Choice

            neighbors.append( neighbor )
    
        return neighbors
    # if elements cannot be repeated
    else:   
        for i in range(0, len(solution.representation)):
            next_choices = deepcopy( solution.encoding.encoding_data )
            neighbor     = deepcopy( solution )
            
            current_element = solution.representation[i]
            next_choices.remove( current_element )
            new_element = choice( next_choices ) # for DataType = Choice
            
            z = -1
            for j in range(0, len(neighbor.representation)):
                if neighbor.representation[j] == new_element: 
                    z = j
                    break

            # if the element is not in other positions accept new_element
            if z == -1 : neighbor.representation[i] = new_element
            # else, the new_element is in another position, swap them.
            else: 
                 neighbor.representation[i] = neighbor.representation[z]
                 neighbor.representation[z] = current_element

            neighbors.append( neighbor )
    
        return neighbors

     
