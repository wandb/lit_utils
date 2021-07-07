"""Utilities for working with model pruning."""
import copy
import math

import torch


def make_prune_config(base_prune_config, network=None, n_epochs=None):
    """Builds a config dictionary for a pl.callbacks.ModelPruning callback.

    Aside from the keyword arguments to that class, this dictionary
    may contain the keys "target_sparsity"

    target_sparsity is combined with n_epochs to determine the value of the
    "amount" keyword argument to ModelPruning, which specifies how much pruning to
    do on each epoch.

    The key "parameters" can be None, "conv", or "linear". It is used to fetch the
    parameters which are to be pruned from the provided network. See
    get_parameters_to_prune for details. Note that None corresponds to pruning
    all parameters.
    """
    prune_config = copy.copy(base_prune_config)
    if "target_sparsity" in prune_config.keys():
        target = prune_config.pop("target_sparsity")
        if n_epochs is None:
            raise ValueError("when specifying target sparsity, must provide number of epochs")
        prune_config["amount"] = compute_iterative_prune(target, n_epochs)

    if "amount" not in prune_config.keys():
        raise ValueError("must specify stepwise pruning amount or target in base_prune_config")

    if "parameters" in prune_config.keys():
        parameters = prune_config.pop("parameters")
        if parameters is not None:
            if network is None:
                raise ValueError("when specifying parameters, must provide network")
        prune_config["parameters_to_prune"] = get_parameters_to_prune(parameters, network)

    if "parameters_to_prune" not in prune_config.keys():
        raise ValueError("must specify which parameters_to_prune in base_prune_config, "
                         "use None for global pruning")

    return prune_config


def get_parameters_to_prune(parameters, network):
    """Return the weights of network matching the parameters value.

    Parameters must be one of "conv" or "linear", or None,
    in which case None is also returned.
    """
    if parameters == "conv":
        return [(layer, "weight") for layer in network.modules()
                if isinstance(layer, torch.nn.Conv2d)]
    elif parameters == "linear":
        return [(layer, "weight") for layer in network.modules()
                if isinstance(layer, torch.nn.Linear)]
    elif parameters is None:
        return
    else:
        raise ValueError(f"could not understand parameters value: {parameters}")


def compute_iterative_prune(target_sparsity, n_epochs):
    return 1 - math.pow(1 - target_sparsity, 1 / n_epochs)
