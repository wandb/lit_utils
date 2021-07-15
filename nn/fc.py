import torch


class FullyConnected(torch.nn.Module):
    """Applies a dense matrix to the inputs.

    Also known as a "dense", "linear", or "perceptron" layer.

    Args:
        in_features: int. Number of entries in the input feature vector.
        out_features: int. Number of entries in the output feature vector.
        activation: Callable or None. (Typically nonlinear) activation function for layer.
                    If None, the identity function is applied.
        batchnorm: String or None. If not None, specifies whether to apply batchnorm
                   before ("pre") or after ("post") the activation function.
        dropout: float or None. If not None, adds dropout layer after the activation function
                 using the provided value as the dropout probability.
    """

    def __init__(self, in_features, out_features, activation=None,
                 batchnorm=None, dropout=None):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

        preactivation = []
        if batchnorm == "pre":
            preactivation.append(torch.nn.BatchNorm1d(out_features))

        if activation is None:
            activation = torch.nn.Identity()

        if not callable(activation):
            raise ValueError(f"activation must be callable, was {activation}")

        postactivation = []
        if dropout is not None:
            postactivation.append(torch.nn.Dropout(dropout))
        if batchnorm == "post":
            postactivation.append(torch.nn.BatchNorm1d(out_features))

        self.preactivation = torch.nn.Sequential(*preactivation)
        self.activation = activation
        self.postactivation = torch.nn.Sequential(*postactivation)

    def forward(self, x):
        return self.postactivation(self.activation(self.preactivation(self.linear(x))))
