from cifo.problem.objective import ProblemObjective

# -------------------------------------------------------------------------------------------------
# Population Class
# -------------------------------------------------------------------------------------------------
class Population:
    """
    Population - 
    """
    # ---------------------------------------------------------------------------------------------
    def __init__( self, problem , maximum_size, solution_list):
        self._problem   = problem
        self._objective = problem.objective
        self._max_size  = maximum_size
        self._list      = solution_list
        self._fittest   = None
        self._sorted    = False

    # ---------------------------------------------------------------------------------------------     
    @property
    def fittest(self):
        self.sort()
        if len(self._list) > 0 :
            return self._list[ -1 ]
        return None

    @property
    def least_fit(self):
        self.sort()
        if len(self._list) > 0 :
            return self._list[ 0 ]
        return None

    def replace_leastfit(self, solution ):
        self.sort()
        self._list[ 0 ] = solution

    @property
    def size(self):
        return len(self._list)
    
    @property
    def has_space(self):
        return len(self._list) < self._max_size

    @property
    def is_full(self):
        return len(self._list) >= self._max_size


    def add(self, solution):
        self._list.append( solution )    

    def get(self, index):
        """
        It returns a solution of the population according to the index
        """
        if index >= 0  and index < len(self._list):
            return self._list[ index ]
        else: 
            return None 

    @property
    def solutions(self):
        """
        Solution list (of the population)
        """
        return self._list

    # 
    def sort(self):
        """
        it sorts the population in ascending order of fittest solution in accordance with the objective

        @ objective
        - Maximization 
        - Minimization
        - Multi-objective { set of objectives }
        """
    

        if self._objective == ProblemObjective.Maximization :
            for i in range (0, len( self._list )) :
                for j in range (i, len (self._list )) :
                    if self._list[ i ].fitness > self._list[ j ].fitness:
                        swap = self._list[ j ]
                        self._list[ j ] = self._list[ i ]
                        self._list[ i ] = swap
                        
        elif self._objective == ProblemObjective.Minimization :    
            for i in range (0, len( self._list )) :
                for j in range (i, len (self._list )) :
                    if self._list[ i ].fitness < self._list[ j ].fitness:
                        swap =self._list[ j ]
                        self._list[ j ] = self._list[ i ]
                        self._list[ i ] = swap                  

        self._sorted = True