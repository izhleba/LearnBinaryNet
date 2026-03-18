# Binary Neural Network Experiments

This file documents the experiments in `binarynet.py`, which explores binary neural networks (BNNs) on the MNIST digit classification task.

## Overview

The script trains two models on MNIST and then compares weight storage formats and inference backends:

1. **Full-precision MLP** -- a standard two-layer network with float32 weights.
2. **Binary MLP** -- a two-layer network where weights and hidden activations are constrained to {+1, -1}.

After training, it saves the binary model in two formats, compares their file sizes, and benchmarks PyTorch GPU inference against a pure NumPy CPU implementation.

## Architecture

### Full-precision MLP

A simple baseline: `Linear(784, 256) -> ReLU -> Linear(256, 10)`. All weights and activations are float32.

### Binary MLP

```
Input (float32, 784)
  -> BinaryLinear(784, 256)    # weights binarized via sign() in forward pass
  -> BatchNorm1d(256)
  -> sign()                    # activations binarized to {+1, -1}
  -> BinaryLinear(256, 10)     # weights binarized via sign() in forward pass
  -> Logits (10)
```

Key components:

- **BinaryLinear**: Stores latent float32 weights but binarizes them with `sign()` during the forward pass. Gradients flow through via the Straight-Through Estimator (STE).
- **Straight-Through Estimator (STE)**: During backpropagation, `sign()` has zero gradient almost everywhere. The clipped STE approximation passes the gradient through unchanged where |x| <= 1, and blocks it otherwise. This allows standard optimizers (Adam) to update the latent weights.
- **BatchNorm**: Placed after the first binary linear layer and before the activation binarization. It re-centers and re-scales the outputs, which is critical for BNNs because the binary matmul produces integer-valued outputs with a limited dynamic range.

## Training

Both models are trained for 5 epochs with:

- **Optimizer**: Adam, learning rate 1e-3
- **Loss**: Cross-entropy
- **Batch size**: 128
- **Seed**: 42 (for reproducibility)

The full-precision model typically reaches ~97-98% test accuracy. The binary model reaches ~95-97%, demonstrating that binarization introduces only a modest accuracy drop on this task.

## Experiments

### Experiment 1: Weight storage comparison

After training, the binary model's weights are saved in two formats:

| Format | Description | Expected size |
|--------|-------------|---------------|
| `.pth` (PyTorch) | Standard `state_dict` with float32 tensors | ~830 KB |
| `.npz` (bit-packed) | Weights packed at 1 bit per value via `np.packbits`; biases and BatchNorm params remain float32 | ~30 KB |

The weight matrices contain 256 x 784 + 10 x 256 = 203,264 parameters. In float32, that is ~793 KB. Bit-packed, it is ~24.8 KB (a **~32x** reduction for the weight data alone). The `.npz` also includes ~2 KB of float32 overhead for biases and BatchNorm parameters.

#### Bit-packing scheme

1. Binarize weights: `w_bin = sign(w)` producing {+1, -1}.
2. Map to bits: `+1 -> 1`, `-1 -> 0`.
3. Pack 8 values per byte with `np.packbits`.
4. Store the original shape alongside for unpacking.

### Experiment 2: Inference backend comparison

The script benchmarks two inference paths on the full MNIST test set (10,000 images):

| Backend | Device | Description |
|---------|--------|-------------|
| PyTorch `BinaryMLP` | GPU (CUDA) | Standard PyTorch forward pass, batched through `DataLoader` |
| NumPy `numpy_inference` | CPU | Pure NumPy implementation: matrix multiplications, vectorized BatchNorm, `np.sign` |

The NumPy inference function replicates the BinaryMLP forward pass exactly:

1. Flatten input to (batch, 784).
2. `x = x @ W1.T + b1` -- first binary linear layer (weights are {+1, -1} stored as float32 after unpacking).
3. BatchNorm (eval mode): `x = (x - running_mean) / sqrt(running_var + eps) * gamma + beta`.
4. `x = sign(x)` -- binarize activations.
5. `x = x @ W2.T + b2` -- second binary linear layer.

Both backends should produce identical predictions (same accuracy), confirming correctness. The timing comparison shows the trade-off between GPU parallelism with framework overhead vs. CPU-only vectorized math.

## Running

```bash
python binarynet.py
```

The script will:

1. Download MNIST to `./data/` (first run only).
2. Train both models for 5 epochs each, printing train/test metrics.
3. Save weights in both formats, print file sizes and compression ratio.
4. Run the inference benchmark, print timing and accuracy for each backend.
