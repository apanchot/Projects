from algorithm.genetic_algorithm import GeneticAlgorithm
from problem.KNeig_problem import KNeigProblem
from problem.Decision_Tree_Problem import DecisionTreeProblem
import pandas as pd
#knapsack_problem_instance = KnapsackProblem(
#    decision_variables = knapsack_decision_variables_example,
#    constraints = knapsack_constraints_example
#)
import warnings
warnings.filterwarnings("ignore")
column_indexes = [15,16,17,18,19]

resultado = []
kn_problem_instance = KNeigProblem()
dt_problem_instance = DecisionTreeProblem(column_index=13)



for j in range(10):
    ga1 = GeneticAlgorithm(
        problem_instance = dt_problem_instance,
        params = {
            "Population-Size" : 30,
            "Number-of-Generations" : 50,
            "Tournament-Size" : 5,
            "Crossover-Probability" : 0.8,
            "Mutation-Probability" : 0.5
        } )
    id, representation, fitness = ga1.search()
    temp_list = [id,str(representation), fitness]
    resultado.append(temp_list)



#resultado.head()
df = pd.DataFrame(columns=['ID', 'REP', 'FITNESS'])
for i in range(len(resultado)):
    df.loc[i] = resultado[i]

df.to_excel("C:/ML/GA_Test/coluna13.xlsx", index=False)


