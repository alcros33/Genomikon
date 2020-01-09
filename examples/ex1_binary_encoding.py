#Example case of use 1
import sys
from functools import partial
sys.path.append("../")
import Genomikon as gen
from Genomikon.utils import bit_encoding_size, bits_to_array

## Objective function
def beale_function(x):
    Res = (1.5 - x[0] * (1 - x[1])) ** 2 + (2.25 - x[0] * (1 - x[1] ** 2)) ** 2
    Res += (2.625 - x[0] * (1 - x[1] ** 3)) ** 2
    return Res

## Define objective with decode func
bit_size = bit_encoding_size(2, -4.5, 4.5)
decode_func = partial(bits_to_array, n=2, minv=-4.5, maxv=4.5)
objective_func = lambda x: - beale_function(decode_func(x))

## Define the population
population = (gen.BinaryGenome.generator(bit_size)
       .evaluate(objective_func)
       .cross(gen.BinaryTwoPointCross(0.5))
       .mutate(gen.BinaryUniformMutator(0.05))
       .population(200))

# First define a generator, whih takes the same arguments as the .random() method does
# A generator is a proxy in which operators can be set
# We set the evaluation operator to be the objective function
# We set the crossover operator to be BinaryTwoPoint with a probability of 0.5
# We set the mutation operator to be BinaryUniform with a probability of 0.05
# Finally we generate a population of 200 Individuals with the above charateristics

# population is a list containing modified BinaryGenome objects
# each modified BinaryGenome has the following methods defined:
# .evaluate() calculates and returns the fitness, which can be later accesed with .fitness
# .cross(other1, other2, ...) performs the cross defined and returns a list with children
# .mutate() performs mutation defined on self.value and returns self


## Define an algorithm which is the main excecution class
AG = gen.Algorithm(population, gen.ProportionalSelector(200), gen.MergeGenerationSelector(200),
                    metrics=[gen.std_fitness, gen.mean_fitness],
                    callbacks=[partial(gen.CSVLogger, file_name="Report.csv")])

# An algorithm is constructed using a population, a parent selector and a survivor selector.
# We set the parent selector to be ProportionalSelector and set it to select 200 parents each gen
# We set the survivor selector to be MergeGeneration and set it to select 200 individuals each gen
# Optional parameters are metrics and callbacks
# Metrics is a list containing functions that calculate metrics from the population each gen
# Callbakcs is a list containing callback objects

r = AG.run(50)
print("Actual Optimum:")
print(f"[3, 0.5] Fitness = f{beale_function([3.0, 0.5])}")
print("Found Optimum:")
print(decode_func(r.value), f"Fitness = {r.fitness}")
