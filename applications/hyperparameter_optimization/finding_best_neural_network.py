import sys

sys.path.append('applications/deep_learning')

import numpy as np
import torch
import torchvision
from models import NeuralNetwork
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace
from torch.utils.data import DataLoader


def neural_network(x):
    # Gathers current optimization parameters
    n_hidden = int(x[0][0])
    lr = x[1][0]

    # Input transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # Loads training and validation sets
    train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    val = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    # Creates training and validation loaders
    train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=0)

    # Instantiates the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(n_input=784, n_hidden=n_hidden, n_output=10, lr=lr, device=device)

    # Fits the model
    _, _, _, val_acc = model.fit(train_loader, val_loader, epochs=3)

    return 1 - val_acc

# Random seeds for experimental consistency
torch.manual_seed(0)
np.random.seed(0)

# Number of agents and decision variables
n_agents = 5
n_variables = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [32, 0.00001]
upper_bound = [512, 0.1]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(neural_network)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=10)
