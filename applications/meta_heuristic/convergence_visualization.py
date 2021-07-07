import opytimizer.visualization.convergence as c
from opytimizer import Opytimizer

# Loads optimization models (PSO and GA)
pso = Opytimizer.load('standard_opt_pso.pkl')
ga = Opytimizer.load('standard_opt_ga.pkl')

# Gathers best agents' positions and fitnesses (PSO and GA)
pso_best_agent_pos, pso_best_agent_fit = pso.history.get_convergence('best_agent')
ga_best_agent_pos, ga_best_agent_fit = ga.history.get_convergence('best_agent')

# Plots the convergence of best agents' positions and fitnesses
c.plot(pso_best_agent_pos[0], ga_best_agent_pos[0], labels=['PSO: $x_0$', 'GA: $x_0$'])
c.plot(pso_best_agent_fit, ga_best_agent_fit, labels=['PSO', 'GA'])
