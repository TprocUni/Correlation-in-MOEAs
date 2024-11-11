#Correlation in a problem space


#imports
import numpy as np
from pymoo.factory import get_problem
from scipy.stats import pearsonr



#get problems
#get objective functions for problems

#generate set of random input values 

#acquire points

#show points on scatter graph





# Define the problems
problems = [f"zdt{i}" for i in range(1, 7)]

# Initialize storage for results
results = {problem: [] for problem in problems}

# Simulation parameters
numIterations = 1000
x_size = 10  # Adjust this based on the problem dimension

# Simulation loop
for problem in problems:
    prob = get_problem(problem, n_var=x_size)
    x = np.random.rand(x_size)  # Initial random solution

    for _ in range(num_iterations):
        # Evaluate the objective functions
        obj_values = prob.evaluate(x)
        results[problem].append(obj_values)

        # Increment/change x
        x = np.random.rand(x_size)  # This can be adjusted as needed


# Correlation analysis (example using Pearson correlation)
for problem in results:
    data = np.array(results[problem])
    if data.shape[1] == 2:  # If there are two objectives
        corr, _ = pearsonr(data[:, 0], data[:, 1])
        print(f"Correlation in {problem}: {corr}")
    else:
        print(f"{problem} has more than two objectives. Further analysis needed.")
