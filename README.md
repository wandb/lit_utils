These utilities are intended for use with W&B educational materials
and are not guaranteed to have a stable API outside of that context.

# Structure

### `datamodules`

This module includes `pl.LightningDataModule`s for
a variety of datasets.

The API is only moderately standardized.
It might be further standardized
if/when we add "dataset callbacks".

### `nn`

`lu.nn` includes `pl.LightningModule`s and `torch.nn.Module`s.
The core module is the `LoggedLitModule`,
which abstracts logging, metrics, and training/validation/test steps
in a manner that's suitable for many basic DNNs.

### `callbacks`

Callbacks includes `pl.Callback`s
that log to Weights & Biases.
The `lu.callbacks.WandbCallback` is designed to work
with any `lu.nn.LoggedLitModule`
and should be included in all educational Colabs.

Others are specific to particular DNN problems,
like image classification or autoencoding.

### `utils`

This is a grab-bag of utilities,
like a run name generator for use with the W&B YOLOv5 integration.

# Installation

These utilities are used in Colab and are "installed" via git.

The following snippet installs the requirements that are not
included in Colab:

```python
%%capture
!pip install pytorch-lightning==1.3.8 torchviz wandb
!git clone https://github.com/wandb/lit_utils
!cd "/content/lit_utils" && git pull
```

The library is imported as `lu`:
```python
import lit_utils as lu
```

And you should use the
```python
lu.utils.filter_warnings()
```

For local development,
also invoke `!pip install -r requirements-dev.txt`.
