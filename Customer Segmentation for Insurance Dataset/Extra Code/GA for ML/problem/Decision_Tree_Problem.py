from problem.problem_template import ProblemTemplate
from problem.objective import ProblemObjective
from problem.solution import LinearSolution, Encoding
from copy import deepcopy
from random import choice, randint
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KN_constraints = {
    "min_sample_split" : [0,301],
    "min_samples_leaf" : [0,301],
    "max_features": [0,10],
    "max_depth" : [0,30]
}


lista_samples = []
lista_leafs = []
lista_max_features = []
lista_depth = []

for i in range(KN_constraints["min_sample_split"][0],KN_constraints["min_sample_split"][-1]):
    lista_samples.append(i)
for i in range(KN_constraints["min_samples_leaf"][0],KN_constraints["min_samples_leaf"][-1]):
    lista_leafs.append(i)
for i in range(KN_constraints["max_depth"][0],KN_constraints["max_depth"][-1]):
    lista_depth.append(i)
for i in range(KN_constraints["max_features"][0],KN_constraints["max_features"][-1]):
    lista_max_features.append(i)


DT_encoding_rule = {
    "Size"         : 20,
    "Is ordered"   : False,
    "Can repeat"   : False,
    "Data"         : [[0,1],lista_samples ,lista_leafs,lista_max_features,lista_depth],
    "Data Type"    : "Choices"
}



decision_variables_DT = {'criterion': ['mse', 'friedman_mse', 'mae'],
            'min_sample_split':lista_samples,
            'min_samples_leaf':lista_leafs,
            'max_features': lista_max_features,
            'max_depth': lista_depth
            }



class DecisionTreeProblem( ProblemTemplate ):
    def __init__(self,column_index, decision_variables=decision_variables_DT, constraints=KN_constraints , encoding_rule = DT_encoding_rule, ):
        if 'criterion' in decision_variables:
            self._criterion = decision_variables["criterion"]
        if 'min_sample_split' in decision_variables:
            self._min_sample_split = decision_variables["min_sample_split"]
        if 'min_samples_leaf' in decision_variables:
            self._min_samples_leaf = decision_variables["min_samples_leaf"]
        if 'max_features' in decision_variables:
            self._max_features = decision_variables["max_features"]
        if 'max_depth' in decision_variables:
            self._max_depth = decision_variables['max_depth']

        self._column_index = column_index
        self._encoding_rule = encoding_rule
        self._encoding = encoding_rule
        self._decision_variables = decision_variables_DT
        self._name = "KNeighbors Problem"

        # Problem Objective
        self._objective_function_list = [ self.objective_function ]
        self._objective_list          = [ ProblemObjective.Maximization ]

    def objective_function(self,solution,column_index):
        data = pd.read_csv('C:/ML/data_insurance.csv')
        complete, _ = split_dataframes(data)

        #Encoding
        columns_list = ['Basic','High School', 'BSc/MSc','PhD','Area 1','Area 2','Area 3','Area 4','No Kids','Have Kids']
        for i in complete.columns:
            if i not in ['Educational Degree','Geographic Living Area','Has Children (Y=1)']:
                columns_list.append(i)

        onehotencoder = OneHotEncoder(categorical_features = [1,3,4])
        complete = pd.DataFrame(onehotencoder.fit_transform(complete).toarray())

        #Give the column names back
        complete.columns = columns_list

        #Drop identity not needed anymore
        complete.drop('Customer Identity', axis=1, inplace=True)
        columns_list.remove('Customer Identity')


        #Standardizing
        scaler = StandardScaler()

        scaled_df = scaler.fit_transform(complete.loc[:,'Gross Monthly Salary':])
        scaled_df = pd.DataFrame(scaled_df)

        scaled_data = pd.concat([complete.loc[:,:'Have Kids'],scaled_df], axis=1)  

        scaled_data.columns = columns_list

        complete = scaled_data
        #complete.to_csv("C:/ML/scaled.csv", index=False)
        i = complete.columns[column_index]
        X_train, X_test, y_train, y_test = train_test_split(complete.loc[:,complete.columns != i].values,
                                                            complete.loc[:,i].values, test_size = 0.35, random_state = 1)

        regressor = DecisionTreeRegressor(criterion=self._decision_variables['criterion'][solution.representation[0]],
                                        min_samples_split =self._decision_variables['min_sample_split'][solution.representation[1]]+1,
                                        min_samples_leaf=self._decision_variables['min_samples_leaf'][solution.representation[2]],
                                        max_features=self._decision_variables['max_features'][solution.representation[3]],
                                        max_depth=self._decision_variables['max_depth'][solution.representation[4]],
                                        random_state=1,
                                        )


        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)

        fitness = r2_score(y_test, y_pred)
        #MSE = mean_squared_error(y_test, y_pred)

        return fitness # MSE

     #[200,1,3,1,1]

    def build_solution(self):

        solution_representation = []

        solution_representation.append(randint(0,1))
        solution_representation.append(self._min_sample_split[randint(1,len(self._min_sample_split)-1)])
        solution_representation.append(self._min_samples_leaf[randint(1,len(self._min_samples_leaf)-1)])
        solution_representation.append(self._max_features[randint(1,len(self._max_features)-1)])
        solution_representation.append(self._max_depth[randint(1,len(self._max_depth)-1)])

        solution = LinearSolution(representation = solution_representation, encoding_rule = self._encoding_rule)

        return solution

    def is_admissible(self,solution):
        decision_variables = self._decision_variables
        crit = False
        mss = False
        msl = False
        mf = False
        md = False

        if self._decision_variables['criterion'][solution.representation[0]] in self._decision_variables['criterion']:
            crit = True
        if solution.representation[1] in list(range(KN_constraints['min_sample_split'][0], KN_constraints['min_sample_split'][-1]+1)):
            mss = True
        if solution.representation[2] in list(range(KN_constraints['min_samples_leaf'][0],KN_constraints['min_samples_leaf'][-1]+1)):
            msl = True
        if solution.representation[3] in list(range(KN_constraints['max_features'][0],KN_constraints['max_features'][-1]+1)):
            mf = True
        if solution.representation[4] in list(range(KN_constraints['max_depth'][0],KN_constraints['max_depth'][-1]+1)):
            md = True
        if crit & mss & msl & mf & md:
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
