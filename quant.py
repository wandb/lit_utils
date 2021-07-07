"""Utilities for working with quantized networks."""
import torch


def run_static_quantization(network, xs, qconfig="fbgemm"):
    """Return a quantized version of supplied network.

    Runs forward pass of network with xs, so make sure they're on
    the same device. Returns a copy of the network, so watch memory consumption.

    Note that this uses torch.quantization, rather than PyTorchLightning.

    Args:
        network: torch.Module, network to be quantized.
        xs: torch.Tensor, valid inputs for network.forward.
        qconfig: string, "fbgemm" to quantize for server/x86, "qnnpack" for mobile/ARM
    """
    # set up quantization
    network.qconfig = torch.quantization.get_default_qconfig(qconfig)
    network.eval()

    # attach methods for collecting activation statistics to set quantization bounds
    qnetwork = torch.quantization.prepare(network)

    # run inputs through network, collect stats
    qnetwork.forward(xs)

    # convert network to uint8 using quantization statistics
    qnetwork = torch.quantization.convert(qnetwork)

    return qnetwork
