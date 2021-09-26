import opytimizer.visualization.convergence as c
from opytimizer import Opytimizer

# Loads optimization models (BH, GA and PSO)
bh = Opytimizer.load('meta_heuristic_bh.pkl')
ga = Opytimizer.load('meta_heuristic_ga.pkl')
pso = Opytimizer.load('meta_heuristic_pso.pkl')

# Gathers best agents' positions and fitnesses (BH, GA and PSO)
bh_best_agent_pos, bh_best_agent_fit = bh.history.get_convergence('best_agent')
ga_best_agent_pos, ga_best_agent_fit = ga.history.get_convergence('best_agent')
pso_best_agent_pos, pso_best_agent_fit = pso.history.get_convergence('best_agent')

# Plots the convergence of best agents' positions and fitnesses
c.plot(bh_best_agent_pos[0], ga_best_agent_pos[0], pso_best_agent_pos[0],
       labels=['BH', 'GA', 'PSO'],
       title='Best agents $x_0$ (first variable) convergence')

c.plot(bh_best_agent_fit, ga_best_agent_fit, pso_best_agent_fit,
       labels=['BH', 'GA', 'PSO'],
       title='Best agents fitness convergence')
