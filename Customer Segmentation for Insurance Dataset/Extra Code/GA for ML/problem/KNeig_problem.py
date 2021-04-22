from problem.problem_template import ProblemTemplate
from problem.objective import ProblemObjective
from problem.solution import LinearSolution, Encoding
from copy import deepcopy
from random import choice, randint
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score
from  math import sqrt
KN_constraints = {
    "Max-Neighbors" : 30
}


lista = []
for i in range(1,KN_constraints["Max-Neighbors"]):
    lista.append(i)

KN_encoding_rule = {
    "Size"         : 20,
    "Is ordered"   : False,
    "Can repeat"   : False,
    "Data"         : [lista ,[0,1],[0,1,2,3],[0,1,2,3,4,5],[0,1]],
    "Data Type"    : "Choices"
}



dv_KN_template = {'n_neighbors':list(range(1,KN_constraints["Max-Neighbors"])),
            'weights':['uniform','distance'],
            'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p':[1,2,3,4,5],
            'metric':['minkowski', 'euclidean']
            }

[11, 1, 0, 0, 1]

class KNeigProblem( ProblemTemplate ):
    def __init__(self, decision_variables=dv_KN_template, constraints=KN_constraints , encoding_rule = KN_encoding_rule):
        if 'n_neighbors' in decision_variables:
            self._n_neighbors = decision_variables["n_neighbors"]
        if 'weights' in decision_variables:
            self._weights = decision_variables["weights"]
        if 'algorithm' in decision_variables:
            self._algorithm = decision_variables["algorithm"]
        if 'p' in decision_variables:
            self._p = decision_variables["p"]
        if 'metric' in decision_variables:
            self._metric = decision_variables["metric"]
        self._encoding_rule = encoding_rule
        self._encoding = encoding_rule
        self._decision_variables = dv_KN_template
        self._name = "KNeighbors Problem"

        # Problem Objective
        self._objective_function_list = [ self.objective_function ]
        self._objective_list          = [ ProblemObjective.Maximization ]

    def objective_function(self,solution,column_index):
        data = pd.read_csv('C:/ML/file.csv')
        complete, _ = split_dataframes(data)


        i = complete.columns[column_index]
        X_train, X_test, y_train, y_test = train_test_split(complete.loc[:,complete.columns != i].values,
                                                            complete.loc[:,i].values, test_size = 0.2, random_state = 0)

        regressor = KNeighborsRegressor(n_neighbors=self.decision_variables['n_neighbors'][solution.representation[0]-1],
                                       weights = self.decision_variables['weights'][solution.representation[1]],
                                       algorithm = self.decision_variables['algorithm'][solution.representation[2]],
                                       p = self.decision_variables['p'][solution.representation[3]],
                                       metric = self.decision_variables['metric'][solution.representation[4]])


        [15, 1, 1, 0, 0]
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)

        fitness = sqrt(mean_squared_error(y_test, y_pred)) #rootMSE
        #fitness = r2_score(y_test, y_pred) #R2
        #fitness = mean_squared_error(y_test, y_pred) #MSE

        return fitness

     #[200,1,3,1,1]
    def build_solution(self):
        solution_representation = []

        solution_representation.append(randint(1,self._n_neighbors[-1]))
        solution_representation.append(randint(0,1))
        solution_representation.append(randint(0,3))
        solution_representation.append(randint(0,1))
        solution_representation.append(randint(0,1))

        solution = LinearSolution(representation = solution_representation, encoding_rule = self._encoding_rule)

        return solution

    def is_admissible(self,solution):
        decision_variables = self._decision_variables
        neig = False
        weig = False
        alg = False
        p = False
        met = False

        if solution.representation[0] in list(range(1, len(decision_variables['n_neighbors'])+1)):
            neig = True
        if solution.representation[1] in list(range(0,len(decision_variables['weights'])+1)):
            weig = True
        if solution.representation[2] in list(range(0,len(decision_variables['algorithm'])+1)):
            alg = True
        if solution.representation[3] in list(range(0,len(decision_variables['p'])+1)):
            p = True
        if solution.representation[4] in list(range(0,len(decision_variables['metric'])+1)):
            met = True

        if neig & weig & alg & p & met:
            return True
        else:
            return False

def split_dataframes(data_insurance, reset_index = False):
    data_insurance_incomplete = data_insurance[data_insurance.isna().any(axis=1)]
    if reset_index:
        data_insurance_incomplete.reset_index(inplace=True)
        data_insurance_incomplete.drop('index', axis=1, inplace=True)

    data_insurance_complete = data_insurance[~data_insurance.isna().any(axis=1)]
    return data_insurance_complete, data_insurance_incomplete
