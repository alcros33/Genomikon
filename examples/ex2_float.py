#Example case of use 2
import sys
from functools import partial
sys.path.append("../")
import Genomikon as gen

## Objective function
def beale_function(x):
    Res = (1.5 - x[0] * (1 - x[1])) ** 2 + (2.25 - x[0] * (1 - x[1] ** 2)) ** 2
    Res += (2.625 - x[0] * (1 - x[1] ** 3)) ** 2
    return Res

## Define objective with decode func
bounds = [-4.5, 4.5]
objective_func = lambda x: - beale_function(x)

## Define the population
population = (gen.FloatGenome.generator(2, bounds)
       .evaluate(objective_func)
       .cross(gen.FloatSimulatedBinaryCross(0.5))
       .mutate(gen.FloatNonUniformMutator(0.3, *bounds))
       .validate(partial(gen.bounds_validator, bounds=bounds))
       .population(200))

# This time we are gonna use FloatGenome which is a numpy nd array
# We set our operators, but add an optinal one which is validate
# validate is called just after mutations and it ensures that the generated genome is valid


## Define an algorithm like last time
AG = gen.Algorithm(population, gen.TournamentSelector(200, 10), gen.MergeGenerationSelector(200),
                    metrics=[gen.std_fitness, gen.mean_fitness],
                    callbacks=[partial(gen.CSVLogger, file_name="Report2.csv")])

r = AG.run(50)
print("Actual Optimum:")
opt = [3.0, 0.5]
print(f"{opt} Fitness = {beale_function(opt)}")
print("Found Optimum:")
print(r)
