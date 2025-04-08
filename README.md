# M2_Coursework

A coursework project for the M2 module: Deep Learning. This project investigates the use of LoRA (Low-Rank Adaptation) for fine-tuning Qwen2.5-0.5B-Instruct, a Large Language Model, on forecasting tasks involving Lotka–Volterra predator-prey systems. The work builds on the LLMTIME approach, converting numerical time series into tokenised sequences for autoregressive prediction.

## Author
- Yilan Xu

## Project Structure


```
src/
├── qwen.py              # Loads Qwen2.5 model
├── lora_skeleton.py     
├── lora_implemented.py  # User-implemented LoRA module
├── preprocessor.py      # LLMTIME preprocessing
├── FLOPS.py             # Floating point operation counter
└── __init__.py

Notebooks/
├── 2(b)3(a).ipynb        # Task 2(b)3(a): model training experiment and evaluation for question 2(b)3(a)
├── 3(b)3(c).ipynb        # Task 3(b)3(c): model training experiment and evaluation for question 3(b)3(c)
├── FLOPS_count.ipynb     # FLOPS accounting
└── Preprocessing.ipynb   # LLMTIME preprocessing exploration

Training Log/
├── results_3a/           # Logs and outputs for Task 3a
├── results_3b/           # Logs and outputs for Task 3b
└── results_3c/           # Logs and outputs for Task 3c

plots/                    # Figures used in the report

reports/
└── M2_report.pdf         # Main report

data/
└── lotka_volterra_data.h5  # Lotka–Volterra predator-prey dataset
```


## Installation

To install required Python packages:

```bash
pip install -r requirements.txt
```

## Install the package for the project

In the root directory:

```bash
pip install -e .
```

## Usage Example

### Load Model and Losses from a Previous Run

To reload a previously trained LoRA model and its corresponding training/validation loss history:

```python
import torch
import numpy as np
from lora_implemented import prepare_model

# Set up model config
model, tokenizer, _ = prepare_model(lora_rank=4, learning_rate=1e-5)

# Define the prefix for saved experiment
filename_prefix = "Training Log/results_3a/experiment" # check Training Log folder for all logs

# Load model weights
lora_weights = torch.load(f"{filename_prefix}_lora_weights.pth")
_ = model.load_state_dict(lora_weights, strict=False)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load training and validation losses
train_losses_3a = np.load(f"{filename_prefix}_train_losses.npy")
val_losses_3a = np.load(f"{filename_prefix}_val_losses.npy")
```

This snippet restores the model state and related loss arrays for further evaluation, plotting, or inference.




## All the other necessary details are included in the report

## Declaration
During the development of this project, I made use of Github Copilot’s autocompletion feature to assist in coding and generating docstrings for functions. Additionally, ChatGPT was used to support debugging. For the final report, ChatGPT helped in generating LaTeX code in desired formats and organising/polishing written languages.

