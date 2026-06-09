# Conditional Autoregressive Diffusion for Discrete Level Generation

A deep learning framework for generating Super Mario Bros levels using conditional diffusion models with autoregressive patch generation.

## Overview

This project implements a two-stage approach for controllable level generation:

1. **Autoencoder**: Learns a compressed latent representation of level patches (14×16 tile grids)
2. **Conditional Diffusion Model**: Generates new level patches in latent space, conditioned on:
   - Previous patches (autoregressive context)
   - Target difficulty score (0.0 - 1.0)

The system uses **Classifier-Free Guidance (CFG)** to enable control over the difficulty of generated levels.

## Project Structure

```
├── config/                  # Configuration files
│   ├── model_config.py      # Model architecture settings
│   ├── training_config.py   # Training hyperparameters
│   ├── generation_config.yaml  # Generation settings
│   └── eval_config.yaml     # Evaluation settings
├── data/                    # Data processing modules
│   ├── parser.py            # Level parsing utilities
│   ├── dataset.py           # PyTorch dataset classes
│   └── extractor.py         # Patch extraction from levels
├── models/                  # Neural network architectures
│   ├── autoencoder.py       # Patch encoder/decoder
│   ├── diffusion.py         # Diffusion U-Net with CFG
│   ├── embeddings.py        # Time/difficulty embeddings
│   ├── noise_scheduler.py   # Diffusion noise schedule
│   └── latent_normalizer.py # Latent space normalization
├── training/                # Training modules
│   ├── autoencoder_trainer.py
│   └── diffusion_trainer.py
├── generation/              # Level generation
│   ├── sampler.py           # Diffusion sampling with CFG
│   └── stitcher.py          # Patch-to-level assembly
├── evaluation/              # Evaluation metrics
│   ├── difficulty_evaluator.py         # Difficulty scoring
│   ├── model_performance_evaluator.py  # CFG diagnostic & difficulty comparison
│   ├── difficulty_prediction_comparison.py  # Compare predicted vs actual difficulty
│   └── astar_agent.py       # A* pathfinding agent for playability testing
├── scripts/                 # Runnable scripts
│   ├── prepare_data.py      # Data preprocessing
│   ├── augment_dataset.py   # Dataset augmentation and balancing
│   ├── analyze_distribution.py  # Analyze difficulty distribution
│   ├── train_autoencoder.py # Autoencoder training
│   ├── prepare_latents.py   # Encode patches to latents
│   ├── train_diffusion.py   # Diffusion model training
│   ├── generate_levels.py   # Level generation
│   ├── evaluate_model_performance.py  # Model evaluation
│   └── compare_with_baseline.py  # Compare with baseline model (MarioGPT)
├── checkpoints/             # Saved model weights
├── output/                  # Generated outputs
├── dataset/                 # Raw level data
├── demo.py                  # Interactive desktop GUI application
├── setup.py                 # Package installation script
└── requirements.txt         # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Conditional Autoregressive diffusion for discreet level generation"

# Create virtual environment (recommended)
conda create -n diffusion python=3.10
conda activate diffusion

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if using GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Step 1: Prepare Data

```bash
python -m scripts.prepare_data
```

This extracts patches and computes difficulty scores, saving to `output/processed/`.

### Step 2: Augment Dataset (Optional but Recommended)

```bash
python -m scripts.augment_dataset
```

This balances the difficulty distribution and enhances patch diversity:

- **Platform Addition**: Adds elevated platforms to 80% of flat patches (any difficulty)
- **Enemy Placement**: Converts easy patches to medium/hard by adding enemies with smart placement:
  - Maximum 2 enemies per distinct platform
  - Enemies spread across left/center/right zones
  - Only modifies patches with existing features (obstacles, gaps, platforms)
- **Conversion Flow**: Easy → Medium → Hard (no direct easy-to-hard)

**Difficulty Classes:**
| Class | Enemy Count | Score |
|-------|-------------|-------|
| Easy | 0-1 | 0.0 |
| Medium | 2-3 | 0.5 |
| Hard | 4-5 | 1.0 |

### Step 3: Train Autoencoder

```bash
python -m scripts.train_autoencoder
```

Trains the patch autoencoder. Checkpoint saved to `checkpoints/autoencoder.pth`.

### Step 4: Prepare Latents

```bash
python -m scripts.prepare_latents
```

Encodes all patches to latent space and fits the latent normalizer.

### Step 5: Train Diffusion Model

```bash
python -m scripts.train_diffusion
```

Trains the conditional diffusion model. Best checkpoint saved to `checkpoints/diffusion_best.pth`.

### Step 6: Generate Levels

Edit `config/generation_config.yaml` to set generation parameters:

```yaml
generation:
  num_levels: 5           # Number of levels to generate
  patches_per_level: 20   # Patches per level (affects length)
  difficulty_target: 0.7  # Difficulty target (0.0=easy, 1.0=hard)
  temperature: 0.5        # Sampling temperature
  guidance_scale: 3.0     # CFG strength (higher = stronger conditioning)
```

Then generate:

```bash
python -m scripts.generate_levels
```

You can also use command-line arguments to override config settings:

```bash
python -m scripts.generate_levels --difficulty 0.8 --temperature 0.5 --guidance 3.0 --patches 10
```

**Available CLI Arguments:**
- `--difficulty` — Difficulty target (0.0-1.0)
- `--temperature` — Sampling temperature
- `--guidance` — CFG guidance scale
- `--patches` — Number of patches per level
- `--num_levels` — Number of levels to generate

Generated levels are saved to `output/generated_levels/`.

### Step 7: Evaluate Model Performance

Evaluate the quality of the trained diffusion model:

```bash
python -m scripts.evaluate_model_performance
```

This runs two evaluation tests:

1. **CFG Diagnostic Test**: Tests the Classifier-Free Guidance signal strength at the model level by comparing conditional vs unconditional noise predictions across different timesteps and difficulty values. Diagnoses whether CFG is working properly.

2. **Difficulty Evaluation**: Generates patches for various target difficulties and measures how well the actual generated difficulty matches the target. Reports Mean Absolute Error (MAE) and correlation coefficient.

Configuration options in `config/eval_config.yaml`:

```yaml
evaluation:
  target_difficulties: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  
  num_samples_per_target: 5
  guidance_scale: 3.0
```

Results are saved to:
- `output/visualizations/difficulty_evaluation.png` — Difficulty comparison plots

### Step 8: Compare with Baseline Model

Compare the diffusion model against a baseline (MarioGPT):

```bash
python -m scripts.compare_with_baseline
```

This evaluates both models on three metrics:

| Metric | Description |
|--------|-------------|
| **Controllability** | How well the model follows difficulty targets (MAE, accuracy, correlation) |
| **Playability** | Percentage of levels completable by an A* agent simulating Mario physics |
| **Diversity** | Tile distribution entropy (higher = more varied levels) |

The A* agent (`evaluation/astar_agent.py`) simulates:
- Walking left/right
- Jumping (up to 4 tiles high, 5 tiles horizontal)
- Falling with gravity
- Collision detection with solid tiles and enemies

Results are saved to:
- `output/visualizations/model_comparison.png` — Side-by-side comparison plots

### Step 9: Interactive Demo (Desktop App)

For an interactive experience, use the desktop GUI application:

```bash
python demo.py
```

## License
MIT License

Copyright (c) 2026 Ahmed Ebrahim (C1earVision)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
