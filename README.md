# visgator

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

### Installation

Requires Python 3.10 (or higher) and [pdm](https://github.com/pdm-project/pdm).

```bash
git clone https://github.com/FrancescoGentile/visgator.git

cd visgator

pdm install
```

### Usage

Example command to evaluate a model on the test set:

```bash
python -m visgator --phase eval --config config/example.json
```
