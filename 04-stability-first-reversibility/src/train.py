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

def train_plain(model, loader, device, epochs, lr, title):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\n>>> Train (plain): {title}")
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

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

def _head_stability_loss(model, ref):
    loss = 0.0
    for (p, p0) in [
        (model.fc3.weight, ref.fc3.weight),
        (model.fc3.bias,   ref.fc3.bias),
    ]:
        loss = loss + (p - p0).pow(2).sum()
    return loss

def train_with_time_constraints(
    model,
    ref,
    loader,
    device,
    epochs,
    lr,
    title,
    lambda_backbone,
    lambda_head,
    use_replay=False,
    replay_X=None,
    replay_Y=None,
    replay_fraction=0.25,
    freeze_head_rows=None,
):
    model.train()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n>>> Train (time-constrained): {title}")
    print(f"    lambda_backbone={lambda_backbone} | lambda_head={lambda_head} | use_replay={use_replay}")
    if freeze_head_rows is not None:
        print(f"    freeze_head_rows={freeze_head_rows}")

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            loss_task = loss_fn(model(xb), yb)

            loss_replay = 0.0
            if use_replay:
                assert replay_X is not None and replay_Y is not None
                n_replay = max(1, int(xb.size(0) * replay_fraction))
                xr, yr = sample_replay_batch(replay_X, replay_Y, n_replay)
                xr, yr = xr.to(device), yr.to(device)
                loss_replay = loss_fn(model(xr), yr)

            loss_back = _backbone_stability_loss(model, ref)
            loss_head = _head_stability_loss(model, ref)

            loss = loss_task + loss_replay + (lambda_backbone * loss_back) + (lambda_head * loss_head)
            loss.backward()

            if freeze_head_rows is not None:
                w_grad = model.fc3.weight.grad
                b_grad = model.fc3.bias.grad
                if w_grad is not None:
                    w_grad[freeze_head_rows].zero_()
                if b_grad is not None:
                    b_grad[freeze_head_rows].zero_()

            opt.step()

def recover_head_only(model, loader, device, epochs, lr, title, freeze_backbone=True):
    print(f"\n>>> Recovery (head-only): {title}")
    if freeze_backbone:
        for p in model.fc1.parameters():
            p.requires_grad = False
        for p in model.fc2.parameters():
            p.requires_grad = False

    opt = optim.Adam(model.fc3.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
