import torch.nn as nn
import torchvision as tv
from models.model import Model


class ResNet(Model):
    """A ResNet class implements a ResNet18 learning architecture.

    """

    def __init__(self, n_channels=3, n_output=10, lr=0.001,
                 kernel_size=3, stride=1, padding=1, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_channels (int): Number of input channels.
            n_output (int): Number of output units.
            lr (float): Learning rate.
            kernel_size (int): Size of first convolutional kernel.
            stride (int): First convolutional layer stride.
            padding (int): First convolutional layer padding.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Override its parent class
        super(ResNet, self).__init__(init_weights, device)

        # Loads base model from torchvision
        self.model = tv.models.resnet18()

        # Replaces first convolutional layer with smaller kernel, stride and padding
        # (n_input, n_output, kernel_size, stride, padding)
        self.model.conv1 = nn.Conv2d(n_channels, 64, kernel_size, stride, padding, bias=False)

        # Replaces fully-connected layer with proper number of classes
        # (n_input, n_output)
        self.model.fc = nn.Linear(512, n_output)

        # Compiles the network
        self._compile(lr)

    def forward(self, x):
        """Performs the forward pass.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            Output tensor.

        """

        # Passes down the model
        preds = self.model(x)
        
        return preds
