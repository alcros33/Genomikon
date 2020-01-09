#Example case of use 3
import sys
from functools import partial
sys.path.append("../")
import Genomikon as gen
from Genomikon.utils import traveling_salesman_objective

data = [[ 0,12,29,22,13,24],
        [12, 0,19, 3,25, 6],
        [29,19, 0,21,23,28],
        [22, 3,21, 0, 4, 5],
        [13,25,23, 4, 0,16],
        [24, 6,28, 5,16, 0]]

## Define objective function
objective_func = lambda x: -traveling_salesman_objective(x, data)

## Define the population
# This time we are gonna use PermutationGenome
population = (gen.PermutationGenome.generator(6)
       .evaluate(objective_func)
       .cross(gen.PermutationOrderCross(0.5))
       .mutate(gen.PermutationInsertMutator(0.3))
       .population(10))

## Define an algorithm like last time
AG = gen.Algorithm(population, gen.UniversalStochasticSelector(10), gen.MergeGenerationSelector(10),
                    metrics=[gen.std_fitness, gen.mean_fitness],
                    callbacks=[partial(gen.CSVLogger, file_name="Report3.csv")])

r = AG.run(50)
print("Actual Optimum:")
opt = [0,4,3,5,1,2]
print(f"{opt} Fitness = {objective_func(opt)}")
print("Found Optimum:")
print(r)
