"""
Callbacks
"""
from .core import *
import Genomikon.core as core

__all__ = ["BaseCallback", "CSVLogger"]

class BaseCallback():
    "Base class for callbacks"
    order = 0
    def __init__(self, algorithm):
        self.algorithm = algorithm
    
    def on_run_begin(self):
        """To initialize constants in the callback."""
        pass
    def on_generation_begin(self):
        """At the beginning of each generation."""
        pass
    def on_selection_begin(self):
        """Just before selection operation"""
        pass
    def on_crossover_begin(self):
        """Just before crossover operation"""
        pass
    def on_mutation_begin(self):
        """Just before mutation"""
        pass
    def on_evaluation_begin(self):
        """Just before children are evaluated in objective function"""
        pass
    def on_survivor_begin(self):
        """Just before survivor selection"""
        pass
    def on_generation_end(self):
        """Called at the end of a generation"""
        pass
    def on_run_end(self):
        """Useful for cleaning up things"""
        pass

    def  __repr__(self):
        attrs = func_args(self.__init__)
        to_remove = getattr(self, 'exclude_repr', [])
        list_repr = [self.__class__.__name__] + [f'{k}: {getattr(self, k)}' for k in attrs if k != 'self' and k not in to_remove]
        return '\n'.join(list_repr)

class CSVLogger(BaseCallback):
    order = 10
    def __init__(self, algorithm, file_name: PathOrStr = "log.csv"):
        self.algorithm = algorithm
        self.file_name = file_name
        self.fieldnames = ["Generation"] + [x.__name__ for x in self.algorithm.metrics] + ["time"]
    def on_run_begin(self):
        with open(self.file_name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
    def on_generation_end(self):
        with open(self.file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            row = self.algorithm.metrics_record[-1]
            row.update({"Generation": core.CTX["GENERATION"]})
            writer.writerow(row)