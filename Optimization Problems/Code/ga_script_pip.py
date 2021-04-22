# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
import multiprocessing as mp
from functools import partial
from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.portfolio_investment_problem import (
    PortfolioInvestmentProblem,
    portfolioDistributionV1,
    portfolioDistributionV2,
    pip_encoding_rule,
    pip_constraints,
    pip_bitflip_get_neighbors
)
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import (
    RankSelection,
    RouletteWheelSelection,
    TournamentSelection,
    cycle_crossover,
    elitism_replacement,
    initialize_randomly,
    initialize_using_hc,
    initialize_using_sa,
    insert_mutation,
    inversion_mutation,
    order_crossover,
    pipArithmeticCrossover,
    pipHeuristicCrossover,
    pmx_crossover,
    edge_crossover,
    scramble_mutation,
    single_point_mutation,
    singlepoint_crossover,
    singlepoint_crossover_pip,
    two_point_crossover_pip,
    standard_replacement,
    swap_mutation,
    swap_mutation_pip
)    
from cifo.util.terminal import Terminal, FontColor
from cifo.util.observer import GeneticAlgorithmObserver
import numpy as np
import pandas as pd
from iteround import saferound
from math import sqrt
from random import seed
from random import randint
from random import shuffle
from random import choice
from datetime import datetime

from os import listdir, path, mkdir
from os.path import isfile, join


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
data = './data/sp500_gen.xlsx'
sp500 = pd.read_excel(data)

data = './data/sp_12_weeks_gen.xlsx'
stdDevMatrix=pd.read_excel(data)

assetsCorrelation = stdDevMatrix.corr()

dv = {
    "sp500" : sp500, 
    "stdDevMatrix" : stdDevMatrix,
    "assetsCorrelation" : assetsCorrelation
}

# Problem Instance
pip_instance = PortfolioInvestmentProblem( 
    decision_variables = dv,
    constraints = pip_constraints
    )

# Configuration
#--------------------------------------------------------------------------------------------------
# parent selection object
parent_selection = TournamentSelection()
#parent_selection = RouletteWheelSelection()

cross20_mut80_singlepoint_crossover_300runs = {
        # params
        "Population-Size"           : 40,
        "Number-of-Generations"     : 1000,
        "Crossover-Probability"     : 0.20,
        "Mutation-Probability"      : 0.80,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection.select,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : singlepoint_crossover,
        "Mutation-Aproach"          : swap_mutation_pip,
        "Replacement-Approach"      : elitism_replacement
    }

cross20_mut80_singlepoint_crossover_hc = {
        # params
        "Population-Size"           : 40,
        "Number-of-Generations"     : 100,
        "Crossover-Probability"     : 0.20,
        "Mutation-Probability"      : 0.80,
        # operators / approaches
        "Initialization-Approach"   : initialize_using_hc,
        "Selection-Approach"        : parent_selection.select,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : singlepoint_crossover,
        "Mutation-Aproach"          : swap_mutation_pip,
        "Replacement-Approach"      : elitism_replacement
    }

params = { 
           'cross20_mut80_singlepoint_crossover_300runs': cross20_mut80_singlepoint_crossover_300runs
           #'cross20_mut80_singlepoint_crossover_hc': cross20_mut80_singlepoint_crossover_hc,
           #'cross20_mut80_singlepoint_crossover_tourn20': cross20_mut80_singlepoint_crossover_tourn20
         }

number_of_runs = 30

def multirun(run,param):
    key,value=param
    log_name=key
    # Genetic Algorithm
    ga = GeneticAlgorithm(
        problem_instance=pip_instance,
        params=value,
        run=run,
        log_name=log_name)

    ga_observer = GeneticAlgorithmObserver(ga)

    ga.register_observer(ga_observer)
    ga.search()
    ga.save_log()

def main():
    for (key,value) in params.items():
        log_name=key
        param=(key,value)

        part=partial(multirun,param=param)
        pool=mp.Pool(processes=int(mp.cpu_count()))
        pool.map(part,range(1,number_of_runs+1))
        pool.close()

     

        # Consolidate the runs
        #--------------------------------------------------------------------------------------------------

        # save the config

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

        if not path.exists(log_dir):
            mkdir(log_dir)

        df.to_excel(log_dir + "/all.xlsx", index = False, encoding = 'utf-8')

        parameters = open(log_dir+"/ga_script_params.txt","w+")
        parameters.write("Selection Algorithm: ")
        parameters.write(str(parent_selection))
        parameters.write('\n')
        parameters.write(str(key))
        parameters.write('\n')
        for _key, _value in value.items():
            parameters.write(str(_key))
            parameters.write(str(_value))
            parameters.write('\n')
        parameters.close()

        #plot_performance_chart( df )
if __name__ == "__main__":
    main()


    #[sum(sublist) for sublist in itertools.izip(*myListOfLists)]
