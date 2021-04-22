from copy import deepcopy
from random import choice, randint

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

import numpy as np
import pandas as pd
from iteround import saferound
from math import sqrt
from random import seed
from random import randint
from random import shuffle
from random import choice

pip_encoding_rule = {
    "Size"         : 503, # It must be defined by the size of DV (Number of products)
    "Is ordered"   : False,
    "Can repeat"   : False,
    "Data"         : [0,100],
    "Data Type"    : "Assets Percentage"
}

pip_constraints = {
    "Risk-Tolerance" : 1, #must be 1.0 or higher
    "Budget": 10000, #not used in this version
    "Maximum-Portfolio-Size" : 20,
    "Risk-Free-Return" : 1.56
}

#1st way of generating the portfolio distribution (weights)
#----------------------------------------------------------------------------------------------

def portfolioDistributionV1(portfolio_size):
    cut_positions = [0]
    for p in range(portfolio_size-1):
        position = randint(1,100)
        while position in cut_positions:
            position = randint(1,100)
        cut_positions.append(position)
    cut_positions.append(100)
    cut_positions.sort()
    
    weights = []
    for asset in range(portfolio_size):
        weights.append(cut_positions[asset+1]-cut_positions[asset])
    shuffle(weights)
        
    return weights

#2nd way of generating the portfolio distribution (weights)
#----------------------------------------------------------------------------------------------

def portfolioDistributionV2(portfolio_size):
    weights = []
    for asset in range(portfolio_size-1):
        weights.append(randint(1,100 - sum(weights) - (portfolio_size-1) + asset))
    weights.append(100 - sum(weights))
    shuffle(weights)
    
    return weights

# -------------------------------------------------------------------------------------------------
# PIP - Portfolio Investment Problem 
# -------------------------------------------------------------------------------------------------
class PortfolioInvestmentProblem(ProblemTemplate):
    """
    Given the SP500 stocks expected return rates, prices
    and standard deviations, the goal of the PIP is to find
    the portfolio with the highest return rate under a
    fixed risk tolerance index. The Sharpe Ratio is used to
    calculate the portfolio's risk and the last 12 week's
    standard deviations matrix is needed.
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints , encoding_rule = pip_encoding_rule):
        """
        """
        # optimize the access to the decision variables
        self._sp500 = pd.DataFrame()
        if "sp500" in decision_variables:
            self._sp500 = decision_variables["sp500"]

        self._stdDevMatrix = pd.DataFrame()
        if "stdDevMatrix" in decision_variables:
            self._stdDevMatrix = decision_variables["stdDevMatrix"]

        #Calcutate the Assets Correlation Matrix
        if "assetsCorrelation" in decision_variables:
            self._assetsCorrelation = decision_variables["assetsCorrelation"]
        else:
            self._assetsCorrelation = self._stdDevMatrix.corr()

        # optimize the access to constraints
        self._riskTolerance = 1 # "Minimum Sharpe Ratio"
        if "Risk-Tolerance" in constraints:
            self._riskTolerance = constraints["Risk-Tolerance"]
        
        self._maxPortfolioSize = 1 # "Minimum Sharpe Ratio"
        if "Maximum-Portfolio-Size" in constraints:
            self._maxPortfolioSize = constraints["Maximum-Portfolio-Size"]
        
        self._riskFreeReturn = 1.56 # "US Bond return"
        if "Risk-Free-Return" in constraints:
            self._riskFreeReturn = constraints["Risk-Free-Return"]

        encoding_rule["Size"] = len(self._sp500.index)

        # Call the Parent-class constructor to store these values and to execute any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables = decision_variables, 
            constraints = constraints, 
            encoding_rule = encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "Portfolio Investment Problem"
        
        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Maximization
    
    # Build Solution for Portfolio Investment Problem
    #----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        """
        #generate a list of zeros with length equal to the number of of possible assets
        portfolio = [0] * self._encoding_rule["Size"]

        # seed random number generator
        #seed(1)

        # generate a random int to set the portfolio size (number of assets in the portfolio)
        portfolio_size = randint(1, self._maxPortfolioSize)

        #select the SP500 assets that will be in the portfolio (positions in the _portfolio_ variable)
        assets_in_portfolio = np.random.choice(self._encoding_rule["Size"]-1, portfolio_size, replace=False)

        #generating the portfolio distribution (weights) using both portfolio distribution algorithms
        distribution_functions = [portfolioDistributionV1, portfolioDistributionV2]
        weights = choice(distribution_functions)(portfolio_size)

        for i, asset in enumerate(assets_in_portfolio):
            portfolio[asset] = weights[i]
        
        solution = LinearSolution(
            representation = portfolio, 
            encoding_rule = self._encoding_rule
        )
        
        self.evaluate_solution(solution)

        return solution

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        1) Calculates the return of the portfolio
        2) Calculates the standard deviation of the portfolio
        3) Calculates the sharpe ratio for the portfolio
        """
        portfolioReturn = 0
        for i, asset in enumerate(solution.representation):
            if asset != 0:
                portfolioReturn += (asset/100)*self._sp500.at[i,'exp_return_3m']
               
        portfolioStdDev = 0
        for i, assetA in enumerate(solution.representation):
            if assetA != 0:
                portfolioStdDev += ((assetA/100)**2)*(self._sp500.loc[i]['standard_deviation']**2)
                for j, assetB in enumerate(solution.representation):
                    if (assetB != 0) and (i != j):
                        portfolioStdDev += (assetA/100)*(assetB/100)*self._assetsCorrelation.at[self._sp500.loc[i]['symbol'],self._sp500.loc[j]['symbol']]*self._sp500.loc[i]['standard_deviation']*self._sp500.loc[j]['standard_deviation']

        _ = (portfolioReturn - float(self._riskFreeReturn))/portfolioStdDev
        
        solution.fitness = float("%.2f" % portfolioReturn)
        solution.stddev = sqrt(portfolioStdDev)
        solution.sharperatio = _

        return solution

    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible(self, solution): #<< use this signature in the sub classes, the meta-heuristic 
        """
        This function tests the solution for 3 conditions:
        1) number of assets smaller than the maximum portfolio size
        2) sum of ratios is equal to 100%
        3) sharpe ratio is higher than 1.0
        """
        #sharpe ratio is higher than the minimum sharpe ratio
        if (solution.sharperatio >= self._riskTolerance):
            #sum of ratios is equal to 100%
            if (sum(solution.representation) == 100):
                #number of assets is smaller than maximum_portfolio_size
                if (np.count_nonzero(np.asarray(solution.representation)) <= self._maxPortfolioSize):
                    return True
        return False 
    
    # Hill Climbing initialization - hill_climbing() and neighborhood_function()
    #----------------------------------------------------------------------------------------------
    def hill_climbing( self, solution ): 
        """
        Find a local maximum given an initial random solution
        """
        best_solution = self.evaluate_solution(solution)
        #print(best_solution.fitness)
        
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
            if (x[i] != 0):
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
def pip_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    pass
