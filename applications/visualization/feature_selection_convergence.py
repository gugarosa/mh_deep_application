import matplotlib.pyplot as plt
import numpy as np
import opytimizer.visualization.convergence as c
from opytimizer import Opytimizer


# Compatibility for loading an object
# which has an external fitness function
def feature_selection(x):
    pass

# Loads the optimization history
opt = Opytimizer.load('history/finding_best_features.pkl')
history = opt.history

# Gathers desired variables
_, agent_0_fit = history.get_convergence('agents', 0)
_, agent_1_fit = history.get_convergence('agents', 1)
_, agent_2_fit = history.get_convergence('agents', 2)
best_pos, best_fit = history.get_convergence('best_agent')

# Plots convergence graph
c.plot(agent_0_fit, agent_1_fit, agent_2_fit, best_fit,
       labels=['$x_0$', '$x_1$', '$x_2$', '$x^*$'],
       title='Agents and best agent fitness convergence')

# Plots features comparison
fig, ax = plt.subplots(1, 3)
ax[0].imshow(np.reshape(best_pos[:, 0], (8, 8)), cmap='Greys', interpolation='nearest')
ax[0].set_title('Iteration: 1')
ax[1].imshow(np.reshape(best_pos[:, 9], (8, 8)), cmap='Greys', interpolation='nearest')
ax[1].set_title('Iteration: 10')
ax[2].imshow(np.reshape(best_pos[:, -1], (8, 8)), cmap='Greys', interpolation='nearest')
ax[2].set_title('Iteration: 100')
plt.setp(ax, xticks=[], yticks=[])
plt.show()
