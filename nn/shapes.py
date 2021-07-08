from math import floor

import torch


def sequential_output_shape(self, shape):
    """Computes the output shape of a torch.nn.Sequential.

    Optimistically assumes any layer without method does not change shape.
    """
    for element in self:
        for cls, method in output_shape_methods.items():
            if isinstance(element, cls):
                shape = method(element, shape)
                break

    return shape


def sequential_feature_dim(self):
    """Computes the feature dimension of a torch.nn.Sequential.

    Returns None if feature dimension cannot be determined.
    """
    feature_dim = None
    for element in reversed(self):
        for cls, method in feature_dim_methods.items():
            if isinstance(element, cls):
                feature_dim = method(element)
                if feature_dim is not None:
                    return feature_dim


def conv2d_output_shape(module, h_w):
    """Computes the output shape of 2d convolutional operators."""
    # grab operator properties
    props = module.kernel_size, module.stride, module.padding, module.dilation
    # diagonalize into tuples as needed
    props = [tuple((p, p)) if not isinstance(p, tuple) else p for p in props]
    # "transpose" operator properties -- list indices are height/width rather than property id
    props = list(zip(*props))

    h = conv1d_output_shape(h_w[0], *props[0])  # calculate h from height parameters of props
    w = conv1d_output_shape(h_w[1], *props[1])  # calculate w from width parameters of props

    assert (h > 0) & (w > 0), "Invalid parameters"

    return h, w


def conv1d_output_shape(lngth, kernel_size, stride, padding, dilation):
    """Computes the change in dimensions for a 1d convolutional operator."""
    return floor( ((lngth + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)  # noqa


def convtranspose2d_output_shape(*args, **kwargs):
    raise NotImplementedError


output_shape_methods = {  # order is important here; torch.nn.Module must be last
    torch.nn.Sequential: sequential_output_shape,
    torch.nn.Conv2d: conv2d_output_shape,
    torch.nn.MaxPool2d: conv2d_output_shape,
    torch.nn.Linear: lambda module, shape: module.out_features,
    torch.nn.AdaptiveAvgPool2d: lambda module, shape: module.output_size,
    torch.nn.Module: lambda module, shape: shape,
    }

feature_dim_methods = {
    torch.nn.Sequential: sequential_feature_dim,
    torch.nn.Conv2d: lambda module: module.out_channels,
    torch.nn.ConvTranspose2d: lambda module: module.out_channels,
    torch.nn.Linear: lambda module: module.out_features,
    }
