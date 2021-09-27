import numpy as np
from opytimark.markers.n_dimensional import Sphere
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.science import BH
from opytimizer.spaces import SearchSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 20
n_variables = 50

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = np.full(n_variables, -10)
upper_bound = np.full(n_variables, 10)

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = BH()
function = Function(Sphere())

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=250)

# Saves the optimization task
opt.save('standard_opt_bh.pkl')
