import matplotlib.pyplot as plt
import numpy as np
import opytimizer.visualization.convergence as c
from opytimizer import Opytimizer


# Compatibility for loading an object
# which has an external fitness function
def neural_network(x):
    pass

# Loads the optimization history
opt = Opytimizer.load('history/finding_best_cnn.pkl')
# opt = Opytimizer.load('history/finding_best_neural_network.pkl')
history = opt.history

# Gathers desired variables
_, agent_0_fit = history.get_convergence('agents', 0)
_, agent_1_fit = history.get_convergence('agents', 1)
_, agent_2_fit = history.get_convergence('agents', 2)
best_pos, best_fit = history.get_convergence('best_agent')
print(best_pos[:, -1])

# Plots convergence graph
c.plot(agent_0_fit, agent_1_fit, agent_2_fit, best_fit,
       labels=['$x_0$', '$x_1$', '$x_2$', '$x^*$'],
       title='Agents and best agent fitness convergence')
