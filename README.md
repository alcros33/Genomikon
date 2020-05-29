# Genomikon
Python Genetic algorithms library with focus on easy of implementation and proof of concepts. Based on the architecture of fast.ai deep-learning library.

## Example
Below is an example on how to solve the traveling salesman problem using Genomikon.
```python
from functools import partial
import Genomikon as gen
from Genomikon.utils import traveling_salesman_objective

data = [[ 0,12,29,22,13,24],
        [12, 0,19, 3,25, 6],
        [29,19, 0,21,23,28],
        [22, 3,21, 0, 4, 5],
        [13,25,23, 4, 0,16],
        [24, 6,28, 5,16, 0]]
# Define objective, negative because genomikon maximizes
objective_func = lambda x: -traveling_salesman_objective(x, data)

## Initial population
population = (gen.PermutationGenome.generator(6) # 6 Citys
       .evaluate(objective_func) # Objective function
       .cross(gen.PermutationOrderCross(0.5)) # Crossbreed op
       .mutate(gen.PermutationInsertMutator(0.3)) # Mutation op
       .population(10)) # 10 initial solutions

## Define execution
AG = gen.Algorithm(population, # Initial population
                   gen.UniversalStochasticSelector(10), # Parent selection policy
                   gen.MergeGenerationSelector(10), # Survivor selection policy
                   metrics=[gen.std_fitness, gen.mean_fitness] # Metrics to log
                   )
result = AG.run(50) # Execute for 50 generations
```
**Output:** [2, 1, 5, 3, 4, 0] fitness=-76

## Reference
*TODO*

## Future plans
* Build a reference
* Implement an OpenMP and/or CUDA backend to achieve performance
