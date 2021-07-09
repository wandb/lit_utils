"""Basic Lightning Modules plus Weights & Biases features."""
import pytorch_lightning as pl
import torch
import torchmetrics


class LoggedLitModule(pl.LightningModule):
    """LightningModule plus wandb features and simple training/val steps.

    By default, assumes that your training loop involves inputs (xs)
    fed to .forward to produce outputs (y_hats)
    that are compared to targets (ys)
    by a loss and by metrics,
    where each batch == (xs, ys).

    If this is not true, overwrite _train_forward
    and optionally _val_forward and _test_forward.
    """

    def __init__(self):
        super().__init__()

        self.training_metrics = torch.nn.ModuleList([])
        self.validation_metrics = torch.nn.ModuleList([])
        self.test_metrics = torch.nn.ModuleList([])

        self.graph_logged = False

    def training_step(self, xys, idx):
        xs, ys = xys
        y_hats = self._train_forward(xs)
        loss = self.loss(y_hats, ys)

        logging_scalars = {"loss": loss}
        for metric in self.training_metrics:
            self.add_metric(metric, logging_scalars, y_hats, ys)

        self.do_logging(xs, ys, idx, y_hats, logging_scalars)

        return {"loss": loss, "y_hats": y_hats}

    def validation_step(self, xys, idx):
        xs, ys = xys
        y_hats = self._val_forward(xs)
        loss = self.loss(y_hats, ys)

        logging_scalars = {"loss": loss}
        for metric in self.validation_metrics:
            self.add_metric(metric, logging_scalars, y_hats, ys)

        self.do_logging(xs, ys, idx, y_hats, logging_scalars, step="validation")

        return {"loss": loss, "y_hats": y_hats}

    def test_step(self, xys, idx):
        xs, ys = xys
        y_hats = self._test_forward(xs)
        loss = self.loss(y_hats, ys)

        logging_scalars = {"loss": loss}
        for metric in self.test_metrics:
            self.add_metric(metric, logging_scalars, y_hats, ys)

        self.do_logging(xs, ys, idx, y_hats, logging_scalars, step="tes")

        return {"loss": loss, "y_hats": y_hats}

    def do_logging(self, xs, ys, idx, y_hats, scalars, step="training"):
        self.log_dict(
            {step + "/" + name: value for name, value in scalars.items()})

    def on_pretrain_routine_start(self):
        print(self)

    def add_metric(self, metric, logging_scalars, y_hats, ys):
        metric_str = metric.__class__.__name__.lower()
        value = metric(y_hats, ys)
        logging_scalars[metric_str] = value

    def _train_forward(self, xs):
        """Overwrite this method when module.forward doesn't produce y_hats."""
        return self.forward(xs)

    def _val_forward(self, xs):
        """Overwrite this method when training and val forward differ."""
        return self._train_forward(xs)

    def _test_forward(self, xs):
        """Overwrite this method when val and test forward differ."""
        return self._val_forward(xs)


class LoggedImageClassifierModule(LoggedLitModule):
    """LightningModule for ImageClassification with Weights and Biases logging."""
    def __init__(self):

        super().__init__()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.training_metrics.append(self.train_acc)
        self.validation_metrics.append(self.valid_acc)
        self.test_metrics.append(self.test_acc)

    def add_metric(self, metric, logging_scalars, y_hats, ys):
        metric_str = metric.__class__.__name__.lower()
        if metric_str == "accuracy":
            preds = self.preds_from_y_hats(y_hats)
            value = metric(preds, ys)
        else:
            value = metric(y_hats, ys)
        logging_scalars[metric_str] = value

    @staticmethod
    def preds_from_y_hats(y_hats):
        if y_hats.shape[-1] == 1:  # handle single-class case
            preds = torch.greater(y_hats, 0.5)
            preds = [bool(pred) for pred in preds]
        else:  # assume we are in the typical one-hot case
            preds = torch.argmax(y_hats, 1)
        return preds
