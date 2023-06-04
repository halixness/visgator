# visgator

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

### Installation

Requires Python 3.10 (or higher) and [pdm](https://github.com/pdm-project/pdm).

```bash
git clone https://github.com/FrancescoGentile/visgator.git

cd visgator

pdm install
```

NOTE: if you want to train or test the models that use Grounding DINO (i.e., SceneGraphGrounder and ERPA), you also need to manually install the grounding dino wheel. To do this, download the wheel from [here](https://drive.google.com/file/d/16ugtOF8kmeOPfhFzbKUsR83uvepfA2ya/view?usp=sharing) and install it with `pip install <path_to_wheel>`. At the moment, we provide the wheel only for Python 3.10. For the configuration and the weights of Grounding DINO, please refer to the [official repository](https://github.com/IDEA-Research/GroundingDINO).

### Usage

Example command to train the baseline model on the RefCOCOg dataset:

```bash
python -m visgator --phase train --config config/example.yaml
```

Example command to test the baseline model on the RefCOCOg dataset:

```bash
python -m visgator --phase eval --config config/example.yaml
```
