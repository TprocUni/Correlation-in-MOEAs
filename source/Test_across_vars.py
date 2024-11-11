#Test

#Correlation in a problem space


#imports
import numpy as np
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pymoo.util.plotting import plot
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
import pandas as pd

from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.indicators.hv import Hypervolume
from pymoo.core.callback import Callback




#get problems
#get objective functions for problems

#generate set of random input values 

#acquire points

#try WFG, CEC UF and CF

# Define the problems
problems = [f"dtlz{i}" for i in range(1, 8)]

AllAveragesCorrs = {problem: [] for problem in problems}


# Initialize storage for results
results = {j: [] for j in range(2,101)}

# Simulation parameters
numIterations = 1000
x_size = 10  # Adjust this based on the problem dimension



def corr4Prob(problem, numIterations=1000):
    results = {}
    correlations = []

    for i in range(30, 101):
        if problem in ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']:
            prob = get_problem(problem, n_var=i)
            x_size = i
        elif problem == 'zdt5':
            prob = get_problem(problem)
            x_size = prob.n_var
        elif problem in [f"dtlz{i}" for i in range(1, 8)]:
            prob = get_problem(problem, n_var = i, n_obj = 3)
            x_size = i

        results[i] = []

        for _ in range(numIterations):
            x = np.random.rand(x_size) if problem != 'zdt5' else np.random.randint(2, size=x_size)
            obj_values = prob.evaluate(x)
            results[i].append(obj_values)

    # Correlation calculation and tracking
    correlationValues = []

    # Pairwise correlation calculation
    for i in results:
        data = np.array(results[i])
        if data.shape[1] >= 2:  # Check if at least two objectives
            for j in range(data.shape[1]):
                for k in range(j + 1, data.shape[1]):
                    corr, _ = pearsonr(data[:, j], data[:, k])
                    correlationValues[(j, k, i)] = corr

    return correlationValues





def corr4Prob2(problem, numIterations=1000):
    results = {}
    pairwiseCorrelations = {}

    for i in range(30, 101):
        if problem in ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']:
            prob = get_problem(problem, n_var=i)
            x_size = i
        elif problem == 'zdt5':
            prob = get_problem(problem)
            x_size = prob.n_var
        elif problem in [f"dtlz{i}" for i in range(1, 8)]:
            prob = get_problem(problem, n_var = i, n_obj = 3)
            x_size = i

        results[i] = []

        for _ in range(numIterations):
            x = np.random.rand(x_size) if problem != 'zdt5' else np.random.randint(2, size=x_size)
            obj_values = prob.evaluate(x)
            results[i].append(obj_values)


    # Pairwise correlation calculation
    for i in results:
        data = np.array(results[i])
        if data.shape[1] >= 2:  # Check if at least two objectives
            for j in range(data.shape[1]):
                for k in range(j + 1, data.shape[1]):
                    corr, _ = pearsonr(data[:, j], data[:, k])
                    pairwiseCorrelations[(j, k, i)] = corr  # Store correlation

    o12 = 0
    o23 = 0
    o31 = 0

    for k,v in pairwiseCorrelations.items():
        if k[0] == 0 and k[1] == 1:
            o12 += v
        elif k[0] == 1 and k[1] == 2:
            o23 += v
        else:
            o31 += v

    return (o12/k[2], o23/k[2], o31/k[2])


def generateData(problemName, nVar, nPoints=10000):
    problem = get_problem(problemName, n_var=nVar)
    X = np.random.rand(nPoints, nVar)
    F = problem.evaluate(X)
    return X, F

    
def plotProblemWithPareto(problemName, nVar):
    X, F = generateData(problemName, nVar)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot evaluated points
    plt.scatter(F[:, 0], F[:, 1], label='Evaluated Points', color='blue', s=10)

    # Plot Pareto front as a line (if available)
    if problemName in ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']:
        problem = get_problem(problemName)
        paretoFront = problem.pareto_front(n_pareto_points=100)  # Specify number of points
        plt.plot(paretoFront[:, 0], paretoFront[:, 1], label='Pareto Front', color='red', linewidth=2)

    plt.title(f'{problemName.upper()} Problem - Evaluated Points and Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def plotProblemWithPareto2(problemName, nVar):
    X, F = generateData(problemName, nVar)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot evaluated points
    plt.scatter(F[:, 0], F[:, 1], label='Evaluated Points', color='blue', s=10)

    # Plot Pareto front as a line (if available)
    if problemName in [f"dtlz{i}" for i in range(1, 8)]:
        problem = get_problem(problemName, n_var = nVar, n_obj = 3)
        paretoFront = problem.pareto_front(n_pareto_points=100)  # Specify number of points
        plt.plot(paretoFront[:, 0], paretoFront[:, 1], label='Pareto Front', color='red', linewidth=2)

    plt.title(f'{problemName.upper()} Problem - Evaluated Points and Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.grid(True)
    plt.show()



'''
# Example Usage
#plotProblemWithPareto('zdt1', 30)
# Assuming 'problems' is a list of problem names
for prob in problems:
    corr= corr4Prob2(prob)
    print(f"Correlation for {prob} is: {corr}")

'''

#-------------------------------------------------------------------------------------------------------------------------

def plotResultsWithParetoFront(df, problemName, nObj):
    # Extract objective values from DataFrame
    objectives = [f"Objective_{i+1}" for i in range(nObj)]
    data = df[objectives].values

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], label='Evaluated Points', color='blue', s=10)

    # Plot Pareto front (if available and applicable)
    if problemName in ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']:
        problem = get_problem(problemName)
        paretoFront = problem.pareto_front(n_pareto_points=100)
        #print(f"paretoFront is {paretoFront}")
        plt.plot(paretoFront[:, 0], paretoFront[:, 1], label='Pareto Front', color='red', linewidth=2)
    elif problemName in [f"DTLZ{i}" for i in range(1, 8)]:
        problem = get_problem(problemName, n_var=len(df.columns), n_obj=nObj)
        paretoFront = problem.pareto_front(n_pareto_points=100)
        plt.plot(paretoFront[:, 0], paretoFront[:, 1], label='Pareto Front', color='red', linewidth=2)

    # Finalizing plot
    plt.title(f'{problemName.upper()} Problem - Evaluated Points and Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
# Assuming df is the DataFrame you got from final()
# plotResultsWithParetoFront(df, 'zdt1', 2)


class RecordHypervolume(Callback):
    def __init__(self, ref_point):
        super().__init__()
        self.ref_point = ref_point
        self.hv = Hypervolume(ref_point=ref_point)
        self.hv_history = []

    def notify(self, algorithm):
        pop = algorithm.pop

        # Check for MOEAD or SPEA2
        if isinstance(algorithm, (MOEAD, SPEA2)):
            # For MOEAD and SPEA2, consider all solutions
            all_solutions = pop.get("F")
            hv_value = self.hv.do(all_solutions)
            self.hv_history.append(hv_value)
        else:
            # Existing logic for other algorithms
            non_dominated_solutions = pop[np.isin(pop.get("rank"), [0])]
            if len(non_dominated_solutions) > 0:
                hv_value = self.hv.do(non_dominated_solutions.get("F"))
                self.hv_history.append(hv_value)





# Function to generate an initial population
def generateInitialPopulation(problem, popSize, seed = 1):
    np.random.seed(seed)
    X = np.random.random((popSize, problem.n_var))
    pop = Population.new("X", X)
    Evaluator().eval(problem, pop)
    return pop



def calculateTotalHypervolume(problem, refPoint):
    # Generate or get the Pareto front for the problem
    paretoFront = problem.pareto_front()

    # Sort the Pareto front points by the first objective
    paretoFrontSorted = sorted(paretoFront, key=lambda x: x[0])

    # Calculate the total hypervolume
    totalHypervolume = 0
    for i in range(len(paretoFrontSorted) - 1):
        xDiff = paretoFrontSorted[i+1][0] - paretoFrontSorted[i][0]
        yMin = min(refPoint[1], paretoFrontSorted[i][1])
        totalHypervolume += xDiff * (refPoint[1] - yMin)

    # Add the volume from the last Pareto point to the reference point
    lastParetoPoint = paretoFrontSorted[-1]
    totalHypervolume += (refPoint[0] - lastParetoPoint[0]) * (refPoint[1] - lastParetoPoint[1])

    return totalHypervolume






def runAlgo(problem, algorithm, generations, seed, ref):
    #initialise the callback function
    #saveCallback = SaveGenerationCallback()
    #hypervolume stuff here
    hv_callback = RecordHypervolume(ref)
    # Extract decision variables from the initial population
    #run the optimisation
    res = minimize(problem, algorithm, ('n_gen', generations), seed=seed, verbose=False, save_history=True, callback=hv_callback)
    print("worked")
    #extract objective values
    objective_values = res.F
    df = pd.DataFrame(objective_values, columns=[f"Objective_{i+1}" for i in range(objective_values.shape[1])])
    return df, hv_callback.hv_history


def plotAlgoPerformance(algosPerformance, totalHV, probName):
    plt.figure(figsize=(12, 8))

    for algo, data in algosPerformance.items():
        generations = range(len(data['hvStats']))
        avg_hv = [avg for _, avg, _ in data['hvStats']]
        min_hv = [min_ for min_, _, _ in data['hvStats']]
        max_hv = [max_ for _, _, max_ in data['hvStats']]

        # Plot the average line
        plt.plot(generations, avg_hv, label=f"{algo} Avg")

        # Create shaded area for min and max range
        plt.fill_between(generations, min_hv, max_hv, alpha=0.2, label=f"{algo} Range")

    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.title(f'Algorithm Performance Over Generations for {probName}')
    plt.legend()
    plt.grid(True)
    plt.show()






def final():
    #finished ones "ZDT4", "ZDT2","DTLZ7","DTLZ3","DTLZ1" 
    actProbs = ["DTLZ5"] #DTLZ 5 is 3D
    algos = ["NSGA2", "SPEA2", "MOEAD", "NSGA3", "SMSEMOA"]
    generations = 100
    popSize = 100
    nOfObj = 2
    elimDupes = False
    margin = 1.0

    ref_dirs = get_reference_directions("das-dennis", nOfObj , n_partitions = (popSize-1))


    #run algorithms
    for problem in actProbs:
        #acquire problem and format
        if problem in ["ZDT4"]:
            prob = get_problem(problem)
        elif problem == "ZDT2":
            prob = get_problem(problem, n_var = 75)
        else:
            prob = get_problem(problem, n_var = 30, n_obj = 2)
        print(prob)

        #Generate static initial pop for problem, to ensure fair comparisons
        initialPopulation = generateInitialPopulation(prob, popSize, seed=1)
        algorithms = {
        # Domination based MOEAs
        "NSGA2": NSGA2(pop_size=popSize, eliminate_duplicates=elimDupes, sampling=initialPopulation),
        "SPEA2": SPEA2(pop_size=popSize, eliminate_duplicates=elimDupes, sampling=initialPopulation),
        # Decomposition based MOEAs
        "MOEAD": MOEAD(ref_dirs=ref_dirs, sampling=initialPopulation),
        "NSGA3": NSGA3(ref_dirs=ref_dirs, sampling=initialPopulation),
        # Indicator-based MOEAs
        "SMSEMOA": SMSEMOA(pop_size=popSize, eliminate_duplicates=elimDupes, sampling=initialPopulation), #hypervolume
        }


        # Evaluate the initial population to get objective values
        F = np.array([prob.evaluate(ind.X) for ind in initialPopulation])
        # Find the maximum value for each objective
        maxValues = np.max(F, axis=0)
        # Add a margin to these maximum values to define the reference point
        ref = maxValues + margin

        print("Reference Point for Hypervolume:", ref)

        algosPerformance = {algo: {'hvRes': [[] for _ in range(generations)], 'df': []} for algo in algos}
        #run algorithms
        for algo in algos:
            for seed in range(5):
                print(f"algo is {algo}")
                #get algo from dict
                algorithm = algorithms[algo]
                #run algorithm
                #df contains the non-dominated results
                df, hvRes = runAlgo(prob, algorithm, generations, seed, ref)
                # Inside your runAlgo function, use the initial_population
                # Store the data frame of the last run (optional)
                algosPerformance[algo]['df'].append(df)
                # Append hypervolume results for each generation
                for gen in range(generations):
                    algosPerformance[algo]['hvRes'][gen].append(hvRes[gen] if gen < len(hvRes) else None)

            #save data
            for algo, data in algosPerformance.items():
                hv_stats = []
                for gen_hv in data['hvRes']:
                    if gen_hv:  # Check if the list is not empty
                        min_hv = min(gen_hv)
                        max_hv = max(gen_hv)
                        avg_hv = sum(gen_hv) / len(gen_hv)
                    else:
                        min_hv = max_hv = avg_hv = None  # Or some default value
                    hv_stats.append((min_hv, avg_hv, max_hv))
                data['hvStats'] = hv_stats

        #totalHV = calculateTotalHypervolume(prob, ref)
        totalHV = 0
        plotAlgoPerformance(algosPerformance, totalHV, problem)








final()








#IMPLEMENT WFG
#           real problem test suite - or similar
