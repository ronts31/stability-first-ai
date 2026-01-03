import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import split_mnist_loaders, build_replay_buffer
from src.model import SimpleMLP
from src.train import (
    eval_acc,
    train_plain,
    train_stability_first_fixed,
    train_fractal_time,
    train_adaptive_time_pain,
    _l2_delta,
)
from src.vae import VAE, vae_loss

def train_vae_on_A(vae, loader_a, device, epochs=3, lr=1e-3):
    vae.train()
    opt = optim.Adam(vae.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        s = 0.0
        for x, _y in loader_a:
            x = x.to(device)
            # Undo normalization approx -> [0,1]
            x01 = x * 0.3081 + 0.1307
            x01 = torch.clamp(x01, 0.0, 1.0)

            opt.zero_grad()
            x_hat, mu, logvar = vae(x01)
            loss = vae_loss(x01, x_hat, mu, logvar)
            loss.backward()
            opt.step()
            s += float(loss.detach().cpu())
        print(f"VAE epoch {ep}/{epochs} loss={s/len(loader_a):.1f}")

@torch.no_grad()
def sample_dreams(vae, device, n):
    vae.eval()
    z = torch.randn(n, vae.z_dim, device=device)
    x01 = vae.decode(z)
    x_norm = (x01 - 0.1307) / 0.3081
    return x_norm

def run_dream_replay(
    base_model,
    ref_model,
    train_a,
    train_b,
    test_a,
    test_b,
    device,
    epochs_b=5,
    lr=1e-3,
    lambda_backbone=2000.0,
    dream_fraction=0.25,
    z_dim=16,
):
    print("\n" + "="*74)
    print("MODE: Dream replay (VAE dreams + teacher labels, no stored A replay)")
    print("="*74)

    model = copy.deepcopy(base_model).to(device)
    ref = copy.deepcopy(ref_model).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    vae = VAE(z_dim=z_dim).to(device)
    print("\n>>> Training VAE on Task A (dream generator)")
    train_vae_on_A(vae, train_a, device=device, epochs=3, lr=lr)

    backbone_p = list(model.fc1.parameters()) + list(model.fc2.parameters())
    backbone_ref = list(ref.fc1.parameters()) + list(ref.fc2.parameters())

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs_b+1):
        total = 0.0
        for xb, yb in train_b:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            loss_b = loss_fn(model(xb), yb)

            n_dream = max(1, int(xb.size(0) * dream_fraction))
            x_dream = sample_dreams(vae, device, n_dream)
            with torch.no_grad():
                y_dream = ref(x_dream).argmax(dim=1)
            loss_dream = loss_fn(model(x_dream), y_dream)

            reg = lambda_backbone * _l2_delta(backbone_p, backbone_ref)
            loss = loss_b + loss_dream + reg
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())

        print(f"epoch {ep}/{epochs_b} total={total/len(train_b):.4f}")

    acc_b = eval_acc(model, test_b, device=device, title="B after B (dream)")
    acc_a = eval_acc(model, test_a, device=device, title="A after B (dream retention)")
    return acc_a, acc_b

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    task_a = [0,1,2,3,4]
    task_b = [5,6,7,8,9]

    print("\n1) Loading Split-MNIST ...")
    train_a, test_a, train_b, test_b, idx_a_train, train_ds = split_mnist_loaders(task_a, task_b, batch_size=128)

    print("\n2) Train Task A (0-4) ...")
    base = SimpleMLP().to(device)
    train_plain(base, train_a, device=device, epochs=3, lr=1e-3, title="Task A (0-4)")
    acc_a_before = eval_acc(base, test_a, device=device, title="A before B")

    ref = copy.deepcopy(base)
    for p in ref.parameters():
        p.requires_grad = False

    replay_X, replay_Y = build_replay_buffer(train_ds, idx_a_train, k=800, seed=seed)

    results = {}

    # Baseline
    print("\n" + "="*74)
    print("MODE: Baseline (naive fine-tuning on B)")
    print("="*74)
    m = copy.deepcopy(base).to(device)
    train_plain(m, train_b, device=device, epochs=5, lr=1e-3, title="Task B (5-9), naive")
    a_after = eval_acc(m, test_a, device=device, title="A after B (baseline)")
    b_after = eval_acc(m, test_b, device=device, title="B after B (baseline)")
    results["baseline"] = (a_after, b_after)

    # Stability-First fixed
    freeze_rows = list(range(0,5))
    m = copy.deepcopy(base).to(device)
    train_stability_first_fixed(
        m, ref=ref, loader_new=train_b, device=device, epochs=5, lr=1e-3,
        title="Task B (fixed time)", lambda_backbone=2000.0,
        freeze_head_rows=freeze_rows, use_replay=True,
        replay_X=replay_X, replay_Y=replay_Y, replay_fraction=0.25,
    )
    a_after = eval_acc(m, test_a, device=device, title="A after B (fixed)")
    b_after = eval_acc(m, test_b, device=device, title="B after B (fixed)")
    results["fixed"] = (a_after, b_after)

    # Fractal time
    m = copy.deepcopy(base).to(device)
    train_fractal_time(
        m, ref=ref, loader_new=train_b, device=device, epochs=5, lr=1e-3,
        title="Task B (fractal time)", lambda_fc1=10000.0, lambda_fc2=3000.0, lambda_head=0.0,
        use_replay=True, replay_X=replay_X, replay_Y=replay_Y, replay_fraction=0.25,
    )
    a_after = eval_acc(m, test_a, device=device, title="A after B (fractal)")
    b_after = eval_acc(m, test_b, device=device, title="B after B (fractal)")
    results["fractal"] = (a_after, b_after)

    # Adaptive time
    m = copy.deepcopy(base).to(device)
    train_adaptive_time_pain(
        m, ref=ref, loader_new=train_b, device=device, epochs=5, lr=1e-3,
        title="Task B (adaptive time)", lambda_min=100.0, lambda_max=20000.0,
        replay_X=replay_X, replay_Y=replay_Y, replay_fraction=0.25,
    )
    a_after = eval_acc(m, test_a, device=device, title="A after B (adaptive)")
    b_after = eval_acc(m, test_b, device=device, title="B after B (adaptive)")
    results["adaptive"] = (a_after, b_after)

    # Dream replay
    results["dream"] = run_dream_replay(
        base_model=base,
        ref_model=ref,
        train_a=train_a,
        train_b=train_b,
        test_a=test_a,
        test_b=test_b,
        device=device,
        epochs_b=5,
        lr=1e-3,
        lambda_backbone=2000.0,
        dream_fraction=0.25,
        z_dim=16,
    )

    # Summary
    print("\n" + "#"*86)
    print(f"{'MODE':<12} | {'A before':>8} | {'A after':>8} | {'B after':>8}")
    print("-"*86)
    for k, (a_after, b_after) in results.items():
        print(f"{k:<12} | {acc_a_before:>7.2f}% | {a_after:>7.2f}% | {b_after:>7.2f}%")
    print("#"*86)

if __name__ == "__main__":
    main()
