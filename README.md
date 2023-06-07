# visgator

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

## Installation

Requires Python 3.10 (or higher) and [pdm](https://github.com/pdm-project/pdm).

```bash
git clone https://github.com/FrancescoGentile/visgator.git

cd visgator

pdm install
```

## Usage

Example command to train the baseline model on the RefCOCOg dataset:

```bash
python -m visgator --phase train --config config/example.yaml
```

Example command to test the baseline model on the RefCOCOg dataset:

```bash
python -m visgator --phase eval --config config/example.yaml
```

Example command to preprocess a dataset (e.g., RefCOCOg) if preprocessing is required:

```bash
python -m visgator --phase generate --config config/example.yaml
```

The `--debug` flag can be used to run the code in debug mode. In debug mode, the code will run on a small subset of the data (max 100 samples per split) and all pytorch operations will be deterministic if possible.

### Configuration file

The training and testing parameters can be easily specified in a (YAML or JSON) configuration file. An example configuration file can be found [here](config/example.yaml). The structure of the configuration file for training is the following:

```yaml
wandb:
  enabled: bool # Whether to use wandb for logging, default: true
  project: str | None # Name of the wandb project, default: visual grounding
  entity: str | None # Name of the wandb entity, default: None
  job_type: str | None # Name of the job type, default: "train"
  name: str | None # Name of the run, default: None
  tags: list[str] | None # List of tags, default: None
  notes: str | None # Notes, default: None
  id: str | None # ID of the run to resume, default: None. Such parameter is necessary if you want to resume a run from a remote checkpoint. If you want to resume a run from a local directory (with a wandb subfolder), you can simply specify the output directory of the run.
  save: bool # Whether to save also the checkpoints and models in wandb, default: false
dir: str # Output directory, default: "output"
debug: bool # Whether to run in debug mode, default: false. This flag will be overwritten if the --debug flag is passed to the script.
num_epochs: int # Number of epochs
train_batch_size: int # Batch size for training
eval_batch_size: int # Batch size for evaluation
seed: int # Seed, default: 3407
compile: bool # Whether to compile the model, default: false
eval_interval: int # Interval between evaluations, default: 1
checkpoint_interval: int # Interval between checkpoints, default: 1
gradient_accumulation_steps: int # Gradient accumulation steps, default: 1
device: str | None # Device, default: None. If None, the device will be automatically selected (CUDA if available, CPU otherwise).
mixed_precision: bool # Whether to use mixed precision, default: true
max_grad_norm: float | None # Max gradient norm, default: None
dataset:
  module: str # Name of the dataset module
  ...
model:
  module: str # Name of the model module
  ...
optimizer:
  module: str # Name of the optimizer module
  ...
lr_scheduler:
  module: str # Name of the lr_scheduler module
  ...
```
