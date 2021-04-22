# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Simulated Annealing Meta-Heuristic
----------------------------------

Content: 

▶ class Simulated Annealing

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0
"""
# -------------------------------------------------------------------------------------------------

# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
# C O D E
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/


# -------------------------------------------------------------------------------------------------
# Class: Simulated Annealing
# -------------------------------------------------------------------------------------------------
class SimulatedAnnealing:
    """
    Classic Implementation of Simulated Annealing with some improvements.

    Improvements:
    ------------
    1. Memory - avoiding to lose the best
    2. C / Minimum C Calibration

    Algorithm:
    ---------
    1: Initialize
    2: Repeat while Control Parameter >= Minimum Control Parameter 
        2.1. Internal Looping
            2.1.1: Get the best neighbor
            2.1.2: Select the best, between the current best and current best neighbor
            2.1.3: Check stop condition ***
        2.2. Update C (Control Parameter)
        2.3: Check stop condition ***
    3: Return the Solution
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, problem_instance, neighborhood_function, feedback = None, config = {}):
        
        """
        Simulated Annealing Constructor

        parameters:
        -----------
        * neighborhood_function - it is expected a function that must follow the signature:
           
        <<neighborhood_function>>( solution, problem, neighborhood_size = 0 )
            
        where:
        - <neighborhood_function>  it is the name of the neighborhood function implemented for the problem_instance
        """
        self._problem_instance = problem_instance
        self._get_neighbors    = neighborhood_function
        self._feedback         = feedback


        # memory (to avoid lost the best)
        self._best_solution     = None

    # Search
    #----------------------------------------------------------------------------------------------
    def search(self):
        """
        Simulated Annealing Search Method
        ----------------------------------

        Algorithm:

        1: Initialize
        2: Repeat while Control Parameter >= Minimum Control Parameter 
            2.1. Internal Looping
                2.1.1: Get the best neighbor
                2.1.2: Select the best, between the current best and current best neighbor
                2.1.3: Check stop condition ***
            2.2. Update C (Control Parameter)
            2.3: Check stop condition ***
        3: Return the Solution

        """
        pass

    # Constructor
    #----------------------------------------------------------------------------------------------
    def _initialize(self):
        """
        Initialize the initial solution, start C and Minimum C
        """
        pass

    # Constructor
    #----------------------------------------------------------------------------------------------
    def _select(self): 
        """
        Select the solution for the next iteration
        """
        pass

    # Constructor
    #----------------------------------------------------------------------------------------------
    def _get_random_neighbor(self, solution):
        """
        Get a random neighbor of the neighborhood (internally it will call the neighborhood provided to the algorithm)
        """
        pass
    
        # Constructor
    #----------------------------------------------------------------------------------------------
    def _initialize_C(self):
        """
        Use one of the available approaches to initialize C and Minimum C
        """
        pass

# -------------------------------------------------------------------------------------------------
# Class: Simulated Annealing
# -------------------------------------------------------------------------------------------------
def initialize_C_approach1():
    return 0, 0