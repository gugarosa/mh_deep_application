import sys

sys.path.append('applications/deep_learning')

import numpy as np
import torch
import torchvision
from models import ResNet
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.evolutionary import GA
from opytimizer.spaces import SearchSpace
from torch.utils.data import DataLoader


def neural_network(x):
    # Gathers current optimization parameters
    kernel_size = int(x[0][0])
    stride = int(x[1][0])
    padding = int(x[2][0])

    # Input transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # Loads training and validation sets
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train, val = torch.utils.data.random_split(train, [40000, 10000])

    # Creates training and validation loaders
    train_loader = DataLoader(train, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=100, shuffle=False, num_workers=0)

    # Instantiates the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(n_channels=3, n_output=10, lr=0.001, kernel_size=kernel_size,
                   stride=stride, padding=padding, device=device)

    # Fits the model
    _, _, _, val_acc = model.fit(train_loader, val_loader, epochs=3)

    return 1 - val_acc

# Random seeds for experimental consistency
torch.manual_seed(0)
np.random.seed(0)

# Number of agents and decision variables
n_agents = 5
n_variables = 3

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [1, 1, 1]
upper_bound = [7, 5, 3]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = GA()
function = Function(neural_network)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=True)

# Runs the optimization task
opt.start(n_iterations=10)

# Saves the optimization task
opt.save('finding_best_cnn.pkl')
