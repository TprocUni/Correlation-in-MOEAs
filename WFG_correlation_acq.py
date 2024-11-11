from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from scipy.stats import pearsonr
import numpy as np

# Choose a WFG problem, for example, WFG1
problem = get_problem("wfg1", n_var=10, n_obj=3)

# Define the algorithm
algorithm = NSGA2(pop_size=100)

# Define the termination criterion
termination = MultiObjectiveDefaultTermination()

# Perform the optimization
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

# Extract the objective values
F = res.F

# Compute Spearman's rank correlation for each pair of objectives
correlations = {}
for i in range(F.shape[1]):
    for j in range(i+1, F.shape[1]):
        corr, _ = pearsonr(F[:, i], F[:, j])
        correlations[(i, j)] = corr

# Print the correlations
for k, v in correlations.items():
    print(f"Correlation between objective {k[0]} and {k[1]}: {v}")
