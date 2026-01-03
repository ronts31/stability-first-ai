import torch
import torch.nn as nn
import torch.optim as optim

from src.data import sample_replay_batch

@torch.no_grad()
def eval_acc(model, loader, device, title):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    acc = 100.0 * correct / total
    print(f"[{title}] acc={acc:.2f}%")
    return acc

def train_standard(model, loader, device, epochs, lr, title):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\n--- TRAIN (standard): {title} ---")
    for ep in range(1, epochs+1):
        s = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            s += loss.item()
        print(f"epoch {ep}/{epochs} loss={s/len(loader):.4f}")

def _backbone_stability_loss(model, ref):
    loss = 0.0
    for (p, p0) in [
        (model.fc1.weight, ref.fc1.weight),
        (model.fc1.bias,   ref.fc1.bias),
        (model.fc2.weight, ref.fc2.weight),
        (model.fc2.bias,   ref.fc2.bias),
    ]:
        loss = loss + (p - p0).pow(2).sum()
    return loss

def _freeze_old_head_grads(model):
    # Preserve interface to old classes (0–4)
    if model.fc3.weight.grad is not None:
        model.fc3.weight.grad[:5].zero_()
    if model.fc3.bias.grad is not None:
        model.fc3.bias.grad[:5].zero_()

def train_stability_first_recursive_time(
    model,
    ref,
    loader_b,
    device,
    epochs,
    lr,
    title,
    lambda_backbone=2000.0,
    freeze_old_head_rows=True,
    use_replay=True,
    replay_X=None,
    replay_Y=None,
    replay_fraction=0.25,
):
    model.train()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n--- TRAIN (Stability-First, recursive-time): {title} ---")
    print(f"lambda_backbone={lambda_backbone} | freeze_old_head_rows={freeze_old_head_rows} | use_replay={use_replay}")
    if use_replay:
        assert replay_X is not None and replay_Y is not None, "Replay buffer required when use_replay=True"
        print(f"replay buffer size: {replay_X.size(0)} | replay_fraction: {replay_fraction}")

    for ep in range(1, epochs+1):
        total_s = task_s = replay_s = stab_s = 0.0

        for xb, yb in loader_b:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            loss_b = loss_fn(model(xb), yb)

            loss_r = 0.0
            if use_replay:
                n_replay = max(1, int(xb.size(0) * replay_fraction))
                xr, yr = sample_replay_batch(replay_X, replay_Y, n_replay)
                xr, yr = xr.to(device), yr.to(device)
                loss_r = loss_fn(model(xr), yr)

            loss_stab = _backbone_stability_loss(model, ref)
            loss = loss_b + loss_r + (lambda_backbone * loss_stab)

            loss.backward()

            if freeze_old_head_rows:
                _freeze_old_head_grads(model)

            opt.step()

            total_s += float(loss.detach().cpu())
            task_s += float(loss_b.detach().cpu())
            replay_s += float(loss_r.detach().cpu()) if use_replay else 0.0
            stab_s += float(loss_stab.detach().cpu())

        print(
            f"epoch {ep}/{epochs} total={total_s/len(loader_b):.4f} "
            f"taskB={task_s/len(loader_b):.4f} replay={replay_s/len(loader_b):.4f} "
            f"stab={stab_s/len(loader_b):.6f}"
        )
