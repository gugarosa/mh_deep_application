import torch.nn as nn
from models import Model


class LSTM(Model):
    """An LSTM class implements a Long Short-Term Memory learning architecture.

    """

    def __init__(self, n_input=512, n_embedding=256, n_hidden=128, n_output=2, lr=0.001,
                 init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_input (int): Number of input units.
            n_embedding (int): Number of embedding units.
            n_hidden (int): Number of hidden units.
            n_output (int): Number of output units.
            lr (float): Learning rate.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Override its parent class
        super(LSTM, self).__init__(init_weights, device)

        # Embedding layer
        # (n_input, n_embedding)
        self.emb = nn.Embedding(n_input, n_embedding)

        # Recurrent layer
        # (n_embedding, n_hidden)
        self.rnn = nn.LSTM(n_embedding, n_hidden)

        # Linear layer
        # (n_input, n_output)
        self.fc = nn.Linear(n_hidden, n_output)

        # Compiles the network
        self._compile(lr)

    def forward(self, x):
        """Performs the forward pass.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            Output tensor.

        """

        # Passes down through the network
        x = self.emb(x)
        _, (h, _) = self.rnn(x)
        preds = self.fc(h[-1])

        return preds
