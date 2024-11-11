#imports
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import json



class SaveGenerationCallback:
    def __init__(self):
        self.data = []

    def __call__(self, algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop
        X, F = pop.get("X", "F")
        self.data.append((gen, X.copy(), F.copy()))



class RunMOEA():
    def __init__(self, problem, algoName, popSize, generations, seed, elimDupes = False):
        #assign values to attributes
        self.problem = get_problem(problem, n_obj=2)
        self.algoName = algoName
        self.seed = seed
        self.generations = generations
        self.popSize = popSize
        self.pf = self.problem.pareto_front()
        #calculate reference directors
        ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions = popSize)
        self.algorithms = {
            # Domination based MOEAs
            "NSGA2": NSGA2(pop_size=popSize, eliminate_duplicates=elimDupes),
            "SPEA2": SPEA2(pop_size=popSize, eliminate_duplicates=elimDupes),
            # Decomposition based MOEAs
            "MOEAD": MOEAD(ref_dirs=ref_dirs),
            "NSGA3": NSGA3(ref_dirs=ref_dirs),
            # Indicator-based MOEAs
            "SMSEMOA": SMSEMOA(pop_size=popSize, eliminate_duplicates=elimDupes), #hypervolume
        }
        self.algorithm = self.algorithms[algoName]
        self.generationalData = {}
        self.df = pd.DataFrame()
        self.correMat = pd.DataFrame()


    def spearmanCorrAllGenerations(self):
        # Initialize a dictionary to store correlation matrices for each generation
        correMatrices = {}

        # Iterate over each generation's data
        for gen, data in self.generationalData.items():
            # Convert generational data to a DataFrame
            df = pd.DataFrame(data, columns=[f"Objective_{i+1}" for i in range(data.shape[1])])
            # Calculate the Spearman correlation matrix for this generation
            correMat = df.corr(method="spearman")
            # Store the correlation matrix in the dictionary
            correMatrices[gen] = correMat

        # Now you have a dictionary of correlation matrices for each generation
        # If you want a single DataFrame for all generations (assuming the same shape for all matrices):
        # Concatenate the correlation matrices along the third axis
        allCorreMat = pd.concat(correMatrices, axis=0)

        # Print or return the new DataFrame with all correlation matrices
        return allCorreMat


    def spearmanCorr(self, data):
        # Assuming data is a list of lists where each inner list has the objective values for a solution
        # Convert data to a DataFrame
        df = pd.DataFrame(data, columns=[f"Objective_{i+1}" for i in range(len(data[0]))])
        # Calculate the Spearman correlation matrix for the DataFrame
        correMat = df.corr(method="spearman")
        # Print the correlation matrix
        return correMat


    def pearsonCorr(self, data):
        df = pd.DataFrame(data, columns=[f"Objective_{i+1}" for i in range(len(data[0]))])
        correMat = df.corr(method="pearson")
        return correMat


    def kendallsTauCorr(self, data):
        df = pd.DataFrame(data, columns=[f"Objective_{i+1}" for i in range(len(data[0]))])
        correMat = df.corr(method="kendall")
        return correMat


    def makeGraph(self):
        #get pareto front
        pf = self.problem.pareto_front()
        #print values and populate graph
        plt.scatter(self.df['Objective_1'], self.df['Objective_2'])
        plt.xlabel('Objective 1')
        plt.plot(pf[:, 0], pf[:, 1], color = "red" , label = "pareto front")
        plt.ylabel('Objective 2')
        plt.title('Objective Space')
        #show graph
        plt.show()


    def makeGraphAllPoints2D(self):
        # Get the Pareto front
        pf = self.problem.pareto_front()

        # Set up the plot and axes
        fig, ax = plt.subplots()

        # Prepare a custom colormap for generations from yellow to blue
        cmap = plt.cm.YlGnBu  # This colormap transitions from light yellow-green-blue to dark blue

        # Define the number of generations
        numGenerations = len(self.generationalData)

        # Normalize the color range for the number of generations
        norm = mcolors.Normalize(vmin=0, vmax=numGenerations-1)

        # Plot each generation with colors from yellow to blue
        for i, (gen, data) in enumerate(sorted(self.generationalData.items())):
            FArray = np.array(data)  # Ensure F is a NumPy array
            ax.scatter(FArray[:, 0], FArray[:, 1], color=cmap(norm(numGenerations - 1 - i)), s=20, alpha=0.5)  # Adjust size and transparency

        # Plot the Pareto front
        if pf is not None:
            ax.plot(pf[:, 0], pf[:, 1], color="red", label="Pareto front")

        # Set labels and title
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title(f'Objective Space Across {self.generations} Generations for {self.algoName} on {self.problem.__class__.__name__}, Population of {self.popSize}')

        # Create a gradient legend for generations
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Set the array to an empty list

        # Determine the tick marks for the colorbar based on generation numbers
        if numGenerations % 2 == 0:
            # If there's an even number of generations, take the one before the exact middle
            middle_gen = (numGenerations // 2) - 1
        else:
            # If there's an odd number of generations, take the exact middle
            middle_gen = numGenerations // 2
        cbar_ticks = [0, middle_gen, numGenerations-1]
        
        cbar = plt.colorbar(sm, ax=ax, ticks=cbar_ticks)
        cbar.set_label('Generation')

        # Set the tick labels for the colorbar
        cbar.ax.set_yticklabels([str(numGenerations - 1 - t) for t in cbar_ticks])

        # Show the plot
        plt.show()


    def makeGraphAllPoints3D(self):
        # Get the Pareto front
        pf = self.problem.pareto_front()

        # Set up the plot and axes for 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Prepare a custom colormap for generations from yellow to blue
        cmap = plt.cm.YlGnBu  # This colormap transitions from light yellow-green-blue to dark blue

        # Define the number of generations
        numGenerations = len(self.generationalData)

        # Normalize the color range for the number of generations
        norm = mcolors.Normalize(vmin=0, vmax=numGenerations-1)

        # Plot each generation with colors from yellow to blue in 3D
        for i, (gen, data) in enumerate(sorted(self.generationalData.items())):
            FArray = np.array(data)  # Ensure F is a NumPy array
            # Assume that FArray has three columns, one for each objective
            ax.scatter(FArray[:, 0], FArray[:, 1], FArray[:, 2], color=cmap(norm(numGenerations - 1 - i)), s=20, alpha=0.5)

        # Plot the Pareto front in 3D
        if pf is not None and pf.shape[1] == 3:  # Check if the Pareto front is 3D
            ax.plot(pf[:, 0], pf[:, 1], pf[:, 2], color="red", label="Pareto front")

        # Set labels and title
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title(f'Objective Space Across {self.generations} Generations for {self.algoName} on {self.problem.__class__.__name__}, Population of {self.popSize}')

        # Create a gradient legend for generations
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Set the array to an empty list

        # Determine the tick marks for the colorbar based on generation numbers
        if numGenerations % 2 == 0:
            # If there's an even number of generations, take the one before the exact middle
            middle_gen = (numGenerations // 2) - 1
        else:
            # If there's an odd number of generations, take the exact middle
            middle_gen = numGenerations // 2
        cbar_ticks = [0, middle_gen, numGenerations-1]

        cbar = plt.colorbar(sm, ax=ax, ticks=cbar_ticks, shrink=0.5)
        cbar.set_label('Generation')
        cbar.ax.set_yticklabels([str(numGenerations - 1 - t) for t in cbar_ticks])

        # Show the plot
        plt.show()


    def runAlgo(self):
        #initialise the callback function
        saveCallback = SaveGenerationCallback()
        #run the optimisation
        res = minimize(self.problem, self.algorithm, ('n_gen', self.generations), seed=self.seed, callback=saveCallback, verbose=False)
        #extract generational data
        for gen, X, F in saveCallback.data:
            #save to generational data dict
            self.generationalData[gen] = F
            #print(f"generation_{gen} has the following number of points: {len(F)}, with popSize {self.popSize}")
        
        #extract objective values
        objective_values = res.F
        self.df = pd.DataFrame(objective_values, columns=[f"Objective_{i+1}" for i in range(objective_values.shape[1])])


    def runAlgoTest(self):
        #initialise the callback function
        saveCallback = SaveGenerationCallback()
        #run the optimisation
        res = minimize(self.problem, self.algorithm, ('n_gen', self.generations), seed=self.seed, verbose=False)
        #extract objective values
        objective_values = res.F
        self.df = pd.DataFrame(objective_values, columns=[f"Objective_{i+1}" for i in range(objective_values.shape[1])])


    def convergenceMetric(self):
        """
        Convergence Metric (CM): Measures the closeness of the solutions to the Pareto-optimal front.
        Lower values indicate better convergence.

        :param F: Array of objective values of the final solution set.
        :param pf: Theoretical Pareto front.
        :return: Convergence metric value.
        """
        F = self.df.to_numpy()
        pf = self.pf
        # Calculate the Euclidean distance from each solution in F to the closest solution in the Pareto front
        distances = np.min(np.sqrt(((F[:, np.newaxis, :] - pf[np.newaxis, :, :]) ** 2).sum(axis=2)), axis=1)
        # Calculate the average of these distances
        cm = np.mean(distances)
        return cm


    def spread(self):
        """
        Spread: Measures the extent of spread achieved among the solutions in the objective space.
        A higher spread indicates better diversity.

        :param F: Array of objective values of the final solution set.
        :return: Spread metric value.
        """
        # Calculate the Euclidean distances between consecutive solutions in the sorted array
        F = self.df.to_numpy()
        sorted_F = np.sort(F, axis=0)
        distances = np.sqrt(((np.diff(sorted_F, axis=0)) ** 2).sum(axis=1))
        # Calculate the average distance (d_bar) and the standard deviation of these distances
        d_bar = np.mean(distances)
        spr = np.std(distances) / d_bar
        return spr





#returns generational data transferred into a numpy array
def convertGDtoArr(dict):
    allData = []
    for gen_data in dict.values():
        for data_point in gen_data:
            allData.append(data_point)
    return np.array(allData)



def convertDataFrame(data):
    """
    Convert a pandas DataFrame into a nested dictionary or list of lists.
    """
    if isinstance(data, pd.DataFrame):
        return data.values.tolist()  # Convert DataFrame to list of lists
    else:
        return data


def saveDataToFile(data, algo, problem, extension='json'):
    """
    Save data, including converted DataFrames (correlation matrices), to a JSON file.
    """
    # Convert DataFrames in the data dictionary to a serializable format
    for key in data:
        data[key] = [convertDataFrame(item) for item in data[key]]

    # Create a directory for the algorithm if it doesn't exist
    directory = os.path.join(os.getcwd(), algo)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the filename
    filename = f"{algo}_{problem}_2D.{extension}"
    filePath = os.path.join(directory, filename)

    # Save the data to the file
    with open(filePath, 'w') as file:
        json.dump(data, file, indent=4)



'''
#ZDTs DTLZ2"
b = RunMOEA("DTLZ2", "NSGA2", 1000, 5, 2, False)
b.runAlgo()

# Get data from generationalData and combine all generations into one list of lists
allDataa = convertGDtoArr(b.generationalData)

# Convert allData to a NumPy array
allDataArray = np.array(allDataa)

# Now call spearmanCorr with the NumPy array
print("spearmans")
b.spearmanCorr(allDataArray)
print("pearsons")
b.pearsonCorr(allDataArray)
print("kendalls")
b.kendallsTauCorr(allDataArray)


b.makeGraphAllPoints3D()
cM = b.convergenceMetric()
s = b.spread()
print(f"CM = {cM}, s = {s}")
'''

#ROTATE LINE or do corre across 
problem = "DTLZ1"
algo = "NSGA2"
b = RunMOEA(problem, algo, 100, 40, 1, False)
b.runAlgo()
convertedData = convertGDtoArr(b.generationalData)
print("Spearmans correlations")
print(b.spearmanCorr(convertedData))
b.makeGraphAllPoints2D()


input()
#problem list:
algos = ["NSGA2", "SPEA2", "MOEAD", "NSGA3", "SMSEMOA"]
problemTag = "DTLZ"

pre = 10

for algo in algos:
    for prob in range(1,8):
        #create problem name
        problem = problemTag+str(prob)
        print(problem)

        dataDict = {
            "problem": problem,
            "algo": algo,
            "sp": [],
            "ps": [],
            "kt": [],
            "cM": [],
            "s": []
            }

        #average over 5 runs
        for i in range(pre):
            works = False
            try:
                #generate run
                b = RunMOEA(problem, algo, 100, 40, i+1, False)
                #run the algorithm
                b.runAlgo()
                convertedData = convertGDtoArr(b.generationalData)
                #get correlations
                sp = b.spearmanCorr(convertedData)
                ps = b.pearsonCorr(convertedData)
                kt = b.kendallsTauCorr(convertedData)
                #get quality measures
                cM = b.convergenceMetric()
                s = b.spread()
                #save values to dict
                dataDict["sp"].append(sp)
                dataDict["ps"].append(ps)
                dataDict["kt"].append(kt)
                dataDict["cM"].append(cM)
                dataDict["s"].append(s)
                works=True
            except:
                print(f"combination {algo} and {prob} failed")
        if works==True:
            #save dict to file
            saveDataToFile(dataDict, algo, problem)
            print("iteration completed")
    


#currently all runs are for 3d.
#do them all for 2d, 
#sample some in 5d



'''
input()
#visualise
plt.figure(figsize=(10, 8))
sns.heatmap(correMat, annot=True, cmap='coolwarm')
plt.show()'''


'''
TODO:
1. get 5 algos /
2. test each works /
3. implement callback feature to save generations /
5. visually represent one run                /
4. implement correlation measures   /
5. experiment

get more problems easily accessible
consider execution times

get measures of quality

alt:
1. implement alternate correlation measures
2. experiment
'''
