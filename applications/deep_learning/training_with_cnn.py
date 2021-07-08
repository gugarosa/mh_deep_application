import torch
import torchvision
from models import ResNet
from torch.utils.data import DataLoader

# Input transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Loads training and testing sets
train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Creates training and testing loaders
train_loader = DataLoader(train, batch_size=100, shuffle=True, num_workers=0)
test_loader = DataLoader(test, batch_size=100, shuffle=False, num_workers=0)

# Instantiates the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(n_channels=3, n_output=10, lr=0.001, device=device)

# Fits the model
model.fit(train_loader, test_loader, epochs=10)
