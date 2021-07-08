import torch
import torchvision
from models import NeuralNetwork
from torch.utils.data import DataLoader

# Input transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Loads training and testing sets
train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

# Creates training and testing loaders
train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)

# Instantiates the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(n_input=784, n_hidden=392, n_output=10, lr=0.001, device=device)

# Fits the model
model.fit(train_loader, test_loader, epochs=10)
