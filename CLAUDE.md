# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research simulation codebase implementing "Meta-Learning-Based Fronthaul Compression for Cloud Radio Access Networks" (IEEE Transactions on Wireless Communications, March 2024). The goal is to learn transformation matrices `W` that compress uplink/downlink signals to fit fronthaul capacity constraints in a 19-cell hexagonal CRAN.

## Running Scripts

No build system. All scripts are run directly with Python from within either the `uplink/` or `downlink/` directory:

```bash
python gen_test_UELocs.py   # Generate test dataset (run first)
python LocalCSI_DNN.py      # Stage 1: train FCNN on local CSI
python Meta_GRU.py          # Stage 2: meta-learning GRU refinement
python EVD.py               # EVD baseline
python Global_GD.py         # Global CSI gradient descent upper bound
python SingleCellProcess.py # Single-cell benchmark
```

Plot scripts live in `plot_result/` subdirectories and read pre-saved `.mat` result files.

**Dependencies** (no requirements.txt — install manually):
```bash
pip install torch numpy scipy matplotlib
```

GPU is used automatically when available. TF32 is explicitly disabled for numerical precision (`torch.backends.cuda.matmul.allow_tf32 = False`).

## Architecture

### Two parallel pipelines
`uplink/` and `downlink/` are near-mirrors. Key difference: uplink uses `UEpow=23 dBm`, downlink uses `UEpow=30 dBm`. Both pipelines share the same method scripts and utility structure.

### Data flow
1. `gen_test_UELocs.py` generates user locations and channel matrices, saved as `.mat` files.
2. Each method script loads the `.mat` dataset and runs its own training/evaluation loop.
3. Results are saved back as `.mat` files for plotting scripts to consume.

### System parameters (hardcoded in each script)
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `Ball` | 19 | Total cells (RRHs) |
| `M` | 32 | Antennas per RRH |
| `Nc` | 2 | Users per cell |
| `Nall` | 38 | Total users |
| `B` | 17 | RRHs in cluster per user |
| `K` | 2 | Compression dimensions |

### Core modules

**`funcs.py`** (~550 lines) — all signal processing utilities: channel generation, rate computation, Gram-Schmidt orthonormalization, EVD baseline, quantization-aware rate calculation, and bit allocation.

**`funcs_autograd.py`** — PyTorch autograd wrapper to differentiate the sum-rate objective w.r.t. transformation matrices `W`.

### Neural network methods

**Stage 1 — `LocalCSI_DNN.py`**: A fully connected network per RRH.
- Input: flattened local channel matrix (32×38 = 1216 dims)
- Layers: 1216→2048 (tanh) → 512 (tanh) → K×M (64 outputs)
- Output rows are unit-normalized then Gram-Schmidt orthogonalized to produce valid `W`
- One independent FCNN instance per RRH (19 models), sharing the same architecture

**Stage 2 — `Meta_GRU.py`**: Refines Stage 1 output with a meta-learning GRU.
- Takes current `W` and its gradient w.r.t. sum rate as input
- GRU hidden state: 2×K×M = 128; followed by 128→256→64 dense layers
- Runs 7 iterative refinement steps per sample
- Trained jointly with the Stage 1 FCNN (loaded from `saved_model/`)

### Model persistence
Checkpoints are saved to `saved_model/`. Setting `epochs=0` in a script skips training and loads the pretrained checkpoint instead.

## Key Design Decisions

- **No shared weights across RRHs**: Each of the 19 RRHs trains its own FCNN independently; parameter sharing is not used.
- **`.mat` files for all I/O**: Test sets and results use MATLAB format via `scipy.io`, enabling reproducibility and cross-tool compatibility.
- **Optional quantization**: `use_quant` flag in training scripts switches between ideal and practical (quantized) rate computation without changing the rest of the pipeline.
- **Validation/testing inside training loops**: There is no separate test runner; `val_size` and `test_size` batches are evaluated within each script's training loop.
