# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------

from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.knapsack_problem import (
    KnapsackProblem, knapsack_decision_variables_example, knapsack_constraints_example, 
    knapsack_bitflip_get_neighbors
)
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import (
    initialize_randomly,
    RouletteWheelSelection, RankSelection, TournamentSelection, 
    singlepoint_crossover,
    single_point_mutation,
    elitism_replacement, standard_replacement 
)    
from cifo.util.terminal import Terminal, FontColor
from cifo.util.observer import GeneticAlgorithmObserver
from random import randint

def plot_performance_chart( df ):
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    x = df["Generation"] 
    x_rev = x[::-1]
    y1 = df["Fitness_Mean"] 
    y1_upper = df["Fitness_Lower"]
    y1_lower = df["Fitness_Upper"]



    # line
    trace1 = go.Scatter(
        x = x,
        y = y1,
        line=dict(color='rgb(0,100,80)'),
        mode='lines',
        name='Fair',
    )

    trace2 = go.Scatter(
        x = x,
        y = y1_upper,
        fill='tozerox',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Fair',
    )

    trace3 = go.Scatter(
        x = x,
        y = y1_lower,
        fill='tozerox',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Fair',
    )

    data = [trace1]
    
    layout = go.Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=dict(
            gridcolor='rgb(255,255,255)',
            range=[1,10],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# Problem
#--------------------------------------------------------------------------------------------------
# Decision Variables
dv = {
    "Values"    : [360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147, 
    78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28, 87, 73, 78, 15, 
    26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276, 312], 

    "Weights"   : [7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13]
}

# Problem Instance
knapsack_problem_instance = KnapsackProblem( 
    decision_variables = dv,
    constraints = { "Max-Weight" : 400 })


# Configuration
#--------------------------------------------------------------------------------------------------
# parent selection object
parent_selection = TournamentSelection()
#parent_selection = RouletteWheelSelection()

params = {
        # params
        "Population-Size"           : 10,
        "Number-of-Generations"     : 10,
        "Crossover-Probability"     : 0.8,
        "Mutation-Probability"      : 0.8,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection.select,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : singlepoint_crossover,
        "Mutation-Aproach"          : single_point_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

log_name = "mp0-8"

number_of_runs = 30

# Run the same configuration many times
#--------------------------------------------------------------------------------------------------
for run in range(1,number_of_runs + 1):
    # Genetic Algorithm
    ga = GeneticAlgorithm( 
        problem_instance = knapsack_problem_instance,
        params =  params,
        run = run,
        log_name = log_name )

    ga_observer = GeneticAlgorithmObserver( ga )
    ga.register_observer( ga_observer )
    ga.search()    
    ga.save_log()

# Consolidate the runs
#--------------------------------------------------------------------------------------------------

# save the config

# consolidate the runs information
from os import listdir, path, mkdir
from os.path import isfile, join
from pandas import pandas as pd
import numpy as np

log_dir   = f"./log/{log_name}" 

log_files = [f for f in listdir(log_dir) if isfile(join(log_dir, f))]
print(log_files)

fitness_runs = []
columns_name = []
counter = 0
generations = []

for log_name in log_files:
    if log_name.startswith('run_'):
        df = pd.read_excel( log_dir + "/" + log_name)
        fitness_runs.append( list ( df.Fitness ) )
        columns_name.append( log_name.strip(".xslx") )
        counter += 1

        if not generations:
            generations = list( df["Generation"] )
        
#fitness_sum = [sum(x) for x in zip(*fitness_runs)]   

df = pd.DataFrame(list(zip(*fitness_runs)), columns = columns_name)

fitness_sd   = list( df.std( axis = 1 ) )
fitness_mean = list( df.mean( axis = 1 ) )

#df["Fitness_Sum"] = fitness_sum
df["Generation"]  = generations
df["Fitness_SD"]  = fitness_sd
df["Fitness_Mean"]  = fitness_mean
df["Fitness_Lower"]  = df["Fitness_Mean"] + df["Fitness_SD"]
df["Fitness_Upper"]  = df["Fitness_Mean"] - df["Fitness_SD"]


if not path.exists( log_dir ):
    mkdir( log_dir )

df.to_excel( log_dir + "/all.xlsx", index = False, encoding = 'utf-8' )

plot_performance_chart( df )




#[sum(sublist) for sublist in itertools.izip(*myListOfLists)]



