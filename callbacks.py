"""Lightning Callbacks for logging to Weights & Biases."""
import os
from pathlib import Path
import tempfile

import numpy as np
import pytorch_lightning as pl
import torch
import wandb


try:
    import torchviz
    has_torchviz = True
except ImportError:
    has_torchviz = False


def get_weights(module):
    weights = [parameter for name, parameter in module.named_parameters()
               if "weight" in name.split(".")[-1]]
    masks = [mask for name, mask in module.named_buffers()
             if "weight_mask" in name.split(".")[-1]]
    if masks:
        with torch.no_grad():
            weights = [mask * weight for mask, weight in zip(masks, weights)]

    return weights


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_params_nonzero(module):
    """Counts the total number of non-zero parameters in a module.

    For compatibility with networks with active torch.nn.utils.prune methods,
    checks for _mask tensors, which are applied during forward passes and so
    represent the actual sparsity of the networks.
    """
    suffix = "_mask"
    if module.named_buffers():
        masks = {name[:-len(suffix)]: mask_tensor for name, mask_tensor in module.named_buffers()
                 if name.endswith(suffix)}
    else:
        masks = {}

    nparams = 0
    with torch.no_grad():
        for name, tensor in module.named_parameters():
            if name[:len(suffix)] in masks.keys():
                nparams += int(torch.sum(tensor != 0))

    return nparams


def fraction_nonzero(module):
    """Gives the fraction of parameters that are non-zero in a module."""
    return count_params_nonzero(module) / count_params(module)


class FilterLogCallback(pl.Callback):
    """PyTorch Lightning Callback for logging the "filters" of a PyTorch Module.

    Filters are weights that touch input or output, and so are often interpretable.
    In particular, these weights are most often interpretable for networks that
    consume or produce images, because they can be viewed as images.

    This Logger selects the input and/or output filters (set by log_input and
    log_output boolean flags) for logging and sends them to Weights & Biases as
    images.
    """
    def __init__(self, image_size=None, log_input=False, log_output=False):
        super().__init__()
        if len(image_size) == 2:
            image_size = [1] + list(image_size)
        self.image_size = image_size
        self.log_input, self.log_output = log_input, log_output

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.log_input:
            input_filters = self.fetch_filters(pl_module, reversed=False,
                                               output_shape=self.image_size)
            self.log_filters(input_filters, "filters/input", trainer)

        if self.log_output:
            output_filters = self.fetch_filters(pl_module, reversed=True,
                                                output_shape=self.image_size)
            self.log_filters(output_filters, "filters/output", trainer)

    def log_filters(self, filters, key, trainer):
        trainer.logger.experiment.log({
            key: wandb.Image(filters.cpu()),
            "global_step": trainer.global_step
        })

    def fetch_filters(self, module, reversed=False, output_shape=None):
        weights = get_weights(module)
        assert len(weights), "could not find any weights"

        if reversed:
            filter_weights = torch.transpose(weights[-1], -2, -1)
        else:
            filter_weights = weights[0]

        filters = self.extract_filters(filter_weights, output_shape=output_shape)

        return filters

    def extract_filters(self, filter_weights, output_shape=None):
        is_convolutional = filter_weights.ndim == 4
        if is_convolutional:
            channel_count = filter_weights.shape[1]
            if channel_count not in [1, 3]:
                raise ValueError("convolutional filters must have 1 (L) or 3 (RGB) channels, " +
                                 f"but had {channel_count}")
            return filter_weights
        else:
            if filter_weights.ndim != 2:
                raise ValueError("filter_weights must have 2 or 4 dimensions, " +
                                 f"but had {filter_weights.ndim}")
            if output_shape is None:
                raise ValueError("output_shape must be provided when final weights are linear")
            filter_weights = self.reshape_linear_weights(filter_weights, output_shape)
            return filter_weights

    @staticmethod
    def reshape_linear_weights(filter_weights, output_shape):
        if len(output_shape) < 2:
            raise ValueError("output_shape must be at least H x W")
        if np.prod(output_shape) != filter_weights.shape[1]:
            raise("shape of filter_weights did not match output_shape")
        return torch.reshape(filter_weights, [-1] + list(output_shape))


class ImageLogCallback(pl.Callback):
    """Logs the input and output images produced by a module to Weights & Biases.

    Useful in combination with, e.g., an autoencoder architecture,
    a convolutional GAN, or any image-to-image transformation network.
    """
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, _ = val_samples
        self.val_imgs = self.val_imgs[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        outs = pl_module(val_imgs)

        mosaics = torch.cat([outs, val_imgs], dim=-2)
        caption = "Top: Output, Bottom: Input"
        trainer.logger.experiment.log({
            "test/examples": [wandb.Image(mosaic, caption=caption)
                              for mosaic in mosaics],
            "global_step": trainer.global_step
            })


class ModelSizeLogCallback(pl.Callback):
    """Logs information about model size to Weights & Biases."""

    def __init__(self, count_nonzero=False):
        super().__init__()
        self.count_nonzero = count_nonzero

    def on_fit_end(self, trainer, module):
        summary = {}
        summary["size_mb"] = self.get_model_disksize(module)
        summary["nparams"] = count_params(module)
        if self.count_nonzero:
            summary["nonzero_params"] = count_params_nonzero(module)

        trainer.logger.experiment.summary.update(summary)

    @staticmethod
    def get_model_disksize(module, print_size=True):
        """Temporarily save model file to disk and return (and optionally print) model size in MB."""
        with tempfile.NamedTemporaryFile() as f:
            torch.save(module.state_dict(), f)
            size_mb = os.path.getsize(f.name) / 1e6
        if print_size:
            print(f"{round(size_mb, 2)} MB")
        return size_mb

    def on_fit_start(self, trainer, module):
        print(f"Parameter Count: {count_params(module)}")


class GraphLogCallback(pl.Callback):
    """Logs a compute graph to Weights & Biases."""

    def __init__(self):
        super().__init__()
        self.graph_logged = False
        assert has_torchviz, "GraphLogCallback requires torchviz installation"

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):
        if not self.graph_logged:
            self.log_graph(trainer, module, outputs["loss"])
            self.graph_logged = True

    @staticmethod
    def log_graph(trainer, module, outputs):
        params_dict = dict(list(module.named_parameters()))
        graph = torchviz.make_dot(outputs, params=params_dict)
        graph.format = "png"
        fname = Path(trainer.logger.experiment.dir) / "graph"
        graph.render(fname)
        wandb.save(str(fname.with_suffix("." + graph.format)))


class SparsityLogCallback(pl.Callback):
    """PyTorch Lightning Callback for logging the sparsity of weight tensors in a PyTorch Module."""

    def on_validation_epoch_end(self, trainer, module):
        self.log_sparsities(trainer, module)

    def get_sparsities(self, module):
        weights = get_weights(module)
        names = [".".join(name.split(".")[:-1]) for name, _ in module.named_parameters()
                 if "weight" in name.split(".")[-1]]
        sparsities = [torch.sum(weight == 0) / weight.numel() for weight in weights]

        return {"sparsity/" + name: sparsity for name, sparsity in zip(names, sparsities)}

    def log_sparsities(self, trainer, module):
        sparsities = self.get_sparsities(module)
        sparsities["sparsity/total"] = 1 - fraction_nonzero(module)
        sparsities["global_step"] = trainer.global_step
        trainer.logger.experiment.log(sparsities)


class MagicCallback(pl.Callback):
    """Attempts to infer metadata about a module and log to Weights & Biases."""

    def on_fit_start(self, trainer, module):
        wandb.run.config["batchnorm"] = self.detect_batchnorm(module)
        wandb.run.config["dropout"] = self.detect_dropout(module)
        wandb.run.config["loss_fn"] = self.detect_loss_fn(module)
        wandb.run.config["optimizer"] = self.detect_optimizer(module)

    def on_train_batch_start(self, trainer, module, batch, batch_idx, dataloader_idx):
        if "x_range" not in wandb.run.config.keys():
            wandb.run.config["x_range"] = self.detect_x_range(batch)

    @staticmethod
    def detect_batchnorm(module):
        for module in module.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                return True
        return False

    @staticmethod
    def detect_dropout(module):
        dropout_cfg_dict = {}
        dropout_ct, dropout2d_ct = 0, 0
        for module in module.modules():
            if isinstance(module, torch.nn.Dropout):
                dropout_cfg_dict[f"dropout.{dropout_ct}.p"] = module.p
                dropout_ct += 1
            if isinstance(module, torch.nn.Dropout2d):
                dropout_cfg_dict[f"dropout.{dropout2d_ct}.p"] = module.p
                dropout2d_ct += 1
        return dropout_cfg_dict

    @staticmethod
    def detect_loss_fn(module):
        try:
            classname = module.loss.__class__.__name__
            if classname in ["method", "function"]:
                return "unknown"
            else:
                return classname
        except AttributeError:
            return

    @staticmethod
    def detect_optimizer(module):
        try:
            return module.optimizers().__class__.__name__
        except AttributeError:
            return

    @staticmethod
    def detect_x_range(batch):
        with torch.no_grad():
            xs = batch[0]
            x_range = [torch.min(xs), torch.max(xs)]
        return x_range


class ImagePredLogCallback(pl.Callback):

    def __init__(self, max_images_to_log=32, labels=None, on_train=False):
        super().__init__()
        self.max_images_to_log = min(max(max_images_to_log, 0), 32)
        self.labels = labels
        self.on_train = on_train

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and dataloader_idx == 0:
            images_with_predictions = self.package_images_predictions(outputs, batch)
            trainer.logger.experiment.log({"validation/predictions": images_with_predictions})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.on_train and batch_idx == 0 and dataloader_idx == 0:
            images_with_predictions = self.package_images_predictions(outputs, batch)
            trainer.logger.experiment.log({"train/predictions": images_with_predictions})

    def package_images_predictions(self, outputs, batch):
        xs, ys = batch
        xs, ys = xs[:self.max_images_to_log], ys[:self.max_images_to_log]
        preds = self.preds_from_y_hats(outputs["y_hats"][:self.max_images_to_log])

        if self.labels is not None:
            preds = [self.labels[int(pred)] for pred in preds]
            ys = [self.labels[int(y)] for y in ys]

        images_with_predictions = [
            wandb.Image(x, caption=f"Pred: {pred}, Target: {y}")
            for x, pred, y in zip(xs, preds, ys)
            ]

        return images_with_predictions

    @staticmethod
    def preds_from_y_hats(y_hats):
        if y_hats.shape[-1] == 1:  # handle single-class case
            preds = torch.greater(y_hats, 0.5)
            preds = [bool(pred) for pred in preds]
        else:  # assume we are in the typical one-hot case
            preds = torch.argmax(y_hats, 1)
        return preds
