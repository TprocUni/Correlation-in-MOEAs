# Correlation-in-MOEAs
The paper investigates how correlations among objectives impact the performance of Multi-Objective Evolutionary Algorithms (MOEAs) and seeks to guide MOEA selection based on these correlations. Key findings reveal that:

Algorithm Types and Correlation:

Decomposition-based algorithms (e.g., MOEA/D) perform best in anti-correlated objective spaces, but they struggle with complex Pareto fronts.
Domination-based algorithms (like NSGA-II and SPEA2) demonstrate consistent performance, making them ideal for scenarios with unknown Pareto fronts.
Indicator-based algorithms (e.g., SMS-EMOA) generally underperform, particularly in negatively correlated scenarios.
Empirical Tests: The study tests MOEAs on problem sets (ZDT, DTLZ) representing various correlation scenarios. MOEA/D excelled in simpler objective spaces but faltered with disconnected and convex Pareto fronts. NSGA-III and SPEA2 showed stability across different correlation conditions.

Experimental Methodology:

Problem correlation is calculated using Pearson’s correlation, with random sampling to approximate the objective space.
Multiple runs are conducted to ensure robust results, with convergence and spread as primary performance metrics.
Conclusion and Recommendations:

Decomposition-based methods suit anti-correlated objectives without complex front shapes.
Domination-based methods are versatile and recommended for unknown or complex objective spaces.
Future research should explore high-dimensional objective spaces and custom problem generation.
This study highlights the importance of objective correlation in MOEA selection, suggesting specific algorithms for distinct correlation scenarios.












