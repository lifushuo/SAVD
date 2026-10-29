import sys


from exact_concorde.exact_concorde import Concorde
import lkh

class LKH(Concorde):
    def __init__(self, nodes=None, coordinates=None, dist_matrix=None):
        """
        Initialize the AdvancedConcorde TSP instance with the option to enable advanced mode.
        """
        super().__init__(nodes, coordinates, dist_matrix)
        self.solver_executor_path = 'LKH-3.0.13/LKH'

    def optimize(self, runs=10):
        instance_dim = len(self.dist_matrix)
        problem = lkh.LKHProblem.load(self.pseudo_file.name)
        self.route = lkh.solve(self.solver_executor_path, problem=problem, max_trials=instance_dim, runs=runs)[0]
        self.route = [i-1 for i in self.route]
        self.route.append(self.route[0]) # a cycle sequence with head and tail node being the same one
