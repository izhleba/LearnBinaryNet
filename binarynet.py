import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# -----------------------------
# 1. Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
DATA_ROOT = "./data"

torch.manual_seed(42)

# -----------------------------
# 2. MNIST data
# -----------------------------
# torchvision now recommends transforms.v2
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # [0,255] -> [0,1]
])

train_ds = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transform,
)
test_ds = datasets.MNIST(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(DEVICE == "cuda"),
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(DEVICE == "cuda"),
)

# -----------------------------
# 3. STE for sign
# -----------------------------
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # save x for backward mask
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # clipped STE:
        # pass gradient only where |x| <= 1
        mask = (x.abs() <= 1).to(grad_output.dtype)
        return grad_output * mask

def ste_sign(x):
    return SignSTE.apply(x)

# -----------------------------
# 4. Standard two-layer network
# -----------------------------
class MLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# -----------------------------
# 5. Binary Linear
# -----------------------------
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # binarize weights only in forward
        w_bin = ste_sign(self.weight)
        return F.linear(x, w_bin, self.bias)

# -----------------------------
# 6. Two-layer binary network
# -----------------------------
class BinaryMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = BinaryLinear(28 * 28, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = BinaryLinear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # first binary layer
        x = self.fc1(x)
        x = self.bn1(x)

        # binarization of hidden activations
        x = ste_sign(x)

        # second binary layer
        x = self.fc2(x)
        return x

# -----------------------------
# 7. train / eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count

def run_experiment(model, name):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\n=== {name} ===")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, DEVICE)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} | test acc {test_acc:.4f}"
        )
    return model

# -----------------------------
# 8. Save weights
# -----------------------------
def save_weights(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch weights to {path}")


def _pack_binary_weight(weight: torch.Tensor) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    w_bin = weight.detach().cpu().sign().numpy()
    bits = (w_bin > 0).astype(np.uint8).flatten()
    packed = np.packbits(bits)
    return packed, np.array(w_bin.shape, dtype=np.int64)


def save_binary_weights(model: BinaryMLP, path: str) -> None:
    sd = model.state_dict()

    fc1_packed, fc1_shape = _pack_binary_weight(sd["fc1.weight"])
    fc2_packed, fc2_shape = _pack_binary_weight(sd["fc2.weight"])

    np.savez(
        path,
        fc1_weight_packed=fc1_packed,
        fc1_weight_shape=fc1_shape,
        fc1_bias=sd["fc1.bias"].cpu().numpy(),
        bn1_weight=sd["bn1.weight"].cpu().numpy(),
        bn1_bias=sd["bn1.bias"].cpu().numpy(),
        bn1_running_mean=sd["bn1.running_mean"].cpu().numpy(),
        bn1_running_var=sd["bn1.running_var"].cpu().numpy(),
        fc2_weight_packed=fc2_packed,
        fc2_weight_shape=fc2_shape,
        fc2_bias=sd["fc2.bias"].cpu().numpy(),
    )
    print(f"Saved bit-packed binary weights to {path}")


# -----------------------------
# 9. Compare file sizes
# -----------------------------
def compare_file_sizes(pth_path: str, npz_path: str) -> None:
    pth_size = os.path.getsize(pth_path)
    npz_size = os.path.getsize(npz_path)

    print("\n=== File size comparison ===")
    print(f"PyTorch .pth : {pth_size / 1024:.1f} KB")
    print(f"Binary  .npz : {npz_size / 1024:.1f} KB")
    print(f"Compression ratio: {pth_size / npz_size:.2f}x")


# -----------------------------
# 10. Load binary weights & numpy inference
# -----------------------------
def load_binary_weights(path: str) -> dict[str, NDArray]:
    data = np.load(path)

    params: dict[str, NDArray] = {}
    for prefix in ("fc1", "fc2"):
        packed = data[f"{prefix}_weight_packed"]
        shape = tuple(data[f"{prefix}_weight_shape"])
        total_bits = int(np.prod(shape))
        unpacked = np.unpackbits(packed)[:total_bits].astype(np.float32)
        # {1, 0} -> {+1, -1}
        params[f"{prefix}_weight"] = (unpacked * 2.0 - 1.0).reshape(shape)
        params[f"{prefix}_bias"] = data[f"{prefix}_bias"].astype(np.float32)

    for key in ("bn1_weight", "bn1_bias", "bn1_running_mean", "bn1_running_var"):
        params[key] = data[key].astype(np.float32)

    return params


def numpy_inference(params: dict[str, NDArray], x: NDArray) -> NDArray:
    x = x.reshape(x.shape[0], -1)
    x = x @ params["fc1_weight"].T + params["fc1_bias"]

    x = (x - params["bn1_running_mean"]) / np.sqrt(params["bn1_running_var"] + 1e-5)
    x = x * params["bn1_weight"] + params["bn1_bias"]

    x = np.sign(x)

    x = x @ params["fc2_weight"].T + params["fc2_bias"]
    return x


# -----------------------------
# 11. Benchmark
# -----------------------------
def benchmark(
    model: BinaryMLP,
    params: dict[str, NDArray],
    loader: DataLoader,
    device: str,
) -> None:
    images_list: list[NDArray] = []
    labels_list: list[NDArray] = []
    for x, y in loader:
        images_list.append(x.numpy())
        labels_list.append(y.numpy())

    all_images = np.concatenate(images_list)
    all_labels = np.concatenate(labels_list)

    # --- NumPy CPU ---
    t0 = time.perf_counter()
    np_logits = numpy_inference(params, all_images)
    np_time = time.perf_counter() - t0

    np_preds = np_logits.argmax(axis=1)
    np_acc = (np_preds == all_labels).mean()

    # --- PyTorch on DEVICE ---
    model.eval()
    torch_preds_list: list[NDArray] = []

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            torch_preds_list.append(logits.argmax(dim=1).cpu().numpy())

    if device == "cuda":
        torch.cuda.synchronize()
    torch_time = time.perf_counter() - t0

    torch_preds = np.concatenate(torch_preds_list)
    torch_acc = (torch_preds == all_labels).mean()

    print("\n=== Inference benchmark ===")
    print(f"NumPy  CPU : {np_time:.4f}s | accuracy {np_acc:.4f}")
    print(f"PyTorch {device.upper():4s}: {torch_time:.4f}s | accuracy {torch_acc:.4f}")

    if np_time > 0 and torch_time > 0:
        if torch_time < np_time:
            print(f"PyTorch is {np_time / torch_time:.2f}x faster")
        else:
            print(f"NumPy is {torch_time / np_time:.2f}x faster")


# -----------------------------
# 12. Run
# -----------------------------
if __name__ == "__main__":
    fp_model = MLP(hidden_dim=256)
    bin_model = BinaryMLP(hidden_dim=256)

    run_experiment(fp_model, "Full-precision 2-layer MLP")
    run_experiment(bin_model, "Binary 2-layer MLP with STE")

    pth_path = "bin_model.pth"
    npz_path = "bin_model_binary.npz"

    save_weights(bin_model, pth_path)
    save_binary_weights(bin_model, npz_path)
    compare_file_sizes(pth_path, npz_path)

    params = load_binary_weights(npz_path)
    benchmark(bin_model, params, test_loader, DEVICE)
