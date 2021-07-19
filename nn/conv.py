import torch


class Convolution2d(torch.nn.Module):
    """Applies 2D convolution to the inputs with torch.nn.Conv2d.

    Also can apply nonlinear activations and other components of convolutional layers.

    Args:
        in_channels: int. Number of channels in the input tensor.
        out_channels: int. Number of channels in the output tensor.
        kernel_size: int or tuple. Shape of convolutional kernel. If int, promoted to (int, int).
        activation: Callable or None. (Typically nonlinear) activation function for layer.
                    If None, the identity function is applied.
        batchnorm: String or None. If not None, specifies whether to apply batchnorm
                   before ("pre") or after ("post") the activation function.
        dropout: float or None. If not None, apply dropout with this probability.
        kwargs:  keyword arguments for torch.nn.Conv2d.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=None,
                 batchnorm=None, dropout=None, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

        preactivation = []
        if batchnorm == "pre":
            preactivation.append(torch.nn.BatchNorm2d(out_channels))

        if activation is None:
            activation = torch.nn.Identity()

        if not callable(activation):
            raise ValueError(f"activation must be callable, was {activation}")

        postactivation = []
        if dropout is not None:
            postactivation.append(torch.nn.Dropout2d(dropout))
        if batchnorm == "post":
            postactivation.append(torch.nn.BatchNorm2d(out_channels))

        self.preactivation = torch.nn.Sequential(*preactivation)
        self.activation = activation
        self.postactivation = torch.nn.Sequential(*postactivation)

    def forward(self, x):
        return self.postactivation(self.activation(self.preactivation(self.conv(x))))


class ConvolutionTranspose2d(torch.nn.Module):
    """Applies 2D tranposed convolution to the inputs.

    Also can apply nonlinear activations and other components of convolutional layers.

    Args:
        in_channels: int. Number of channels in the input tensor.
        out_channels: int. Number of channels in the output tensor.
        kernel_size: int or tuple. Shape of convolutional kernel. If int, promoted to (int, int).
        activation: Callable or None. (Typically nonlinear) activation function for layer.
                    If None, the identity function is applied.
        batchnorm: String or None. If not None, specifies whether to apply batchnorm
                   before ("pre") or after ("post") the activation function.
        dropout: float or None. If not None, apply dropout with this probability.
        kwargs:  keyword arguments for torch.nn.ConvTranspose2d.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=None, batchnorm=None, dropout=None, **kwargs):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size. **kwargs)

        preactivation = []
        if batchnorm == "pre":
            preactivation.append(torch.nn.BatchNorm2d(out_channels))

        if activation is None:
            activation = torch.nn.Identity()

        if not callable(activation):
            raise ValueError(f"activation must be callable, was {activation}")

        postactivation = []
        if dropout is not None:
            postactivation.append(torch.nn.Dropout2d(dropout))
        if batchnorm == "post":
            postactivation.append(torch.nn.BatchNorm2d(out_channels))

        self.preactivation = torch.nn.Sequential(*preactivation)
        self.activation = activation
        self.postactivation = torch.nn.Sequential(*postactivation)

    def forward(self, x):
        return self.postactivation(self.activation(self.preactivation(self.conv(x))))
