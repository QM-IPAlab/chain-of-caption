<div align="center">
<h2><center> Chain-of-Caption: Training-free improvement of multimodal large language model on referring expression comprehension </h2>

<a href='https://arxiv.org/abs/2602.08211'><img src='https://img.shields.io/badge/ArXiv-2412.14803-red'></a> 
<a href='https://qm-ipalab.github.io/chain-of-caption/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

This is the official repository for "Chain-of-Caption: Training-free improvement of multimodal large language model on referring expression comprehension" published in ICASSP 2026. This repository contains code for evaluating NVILA-8B with Chain-of-Caption on the RefCOCO dataset.

## Installation

Follow the instructions at [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to set up the conda environment and environment variables.

## Run evaluation

See [`examples/models/vila.sh`](examples/models/vila.sh) for the main evaluation script and [`lmms_eval/models/vila.py`](lmms_eval/models/vila.py) for all the prompts used.

First set up environment variables for downloading models and datasets.

```bash
export HF_HOME="~/.cache/huggingface"
export HF_TOKEN="your_token_here"
```

Then, select options for the model

**Inference options** (optional env vars):

| Variable       | Values / type | Description                                      |
|----------------|---------------|--------------------------------------------------|
| `DESC_MODE`    | `none`, `all`, or integer (e.g. `5`) | Grounded description mode   |
| `DRAW_BBOX`    | `true` / `false` | Draw bounding boxes (default: `false`)       |
| `COC`          | `true` / `false` | Chain-of-Caption (default: `false`)           |
| `CROP_AND_ZOOM`| `true` / `false` | Crop and zoom (default: `false`)            |

**Debug visualization** (optional):

| Variable                 | Description                                      |
|--------------------------|--------------------------------------------------|
| `DEBUG_OUTPUT_PATH`      | Directory to save debug images (empty = disabled)|
| `DEBUG_DRAW_TEXT`        | `true` / `false` — overlay text on debug images  |
| `DEBUG_DRAW_BOUNDING_BOX`| `true` / `false` — draw boxes on debug images    |

Example: run with 5 objects in grounded description and enable Chain-of-Caption:

```
COC="true"
DESC_MODE="5"
DRAW_BBOX="false"
CROP_AND_ZOOM="false"
```

Finally, run the script
```bash
bash examples/models/vila.sh
```

# Acknowledgements
This code is an extension of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
