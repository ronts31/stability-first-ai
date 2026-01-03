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
    for ep in range(1, epochs+1):
        s = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            s += float(loss.detach().cpu())
        print(f"epoch {ep}/{epochs} loss={s/len(loader):.4f}")

def _l2_delta(params, ref_params):
    s = 0.0
    for p, p0 in zip(params, ref_params):
        s = s + (p - p0).pow(2).sum()
    return s

@torch.no_grad()
def _delta_norm(params, ref_params):
    s = 0.0
    for p, p0 in zip(params, ref_params):
        d = (p - p0).detach()
        s += float(d.pow(2).sum().cpu())
    return s ** 0.5

def train_stability_first_fixed(
    model,
    ref,
    loader_new,
    device,
    epochs,
    lr,
    title,
    lambda_backbone=2000.0,
    freeze_head_rows=None,
    use_replay=True,
    replay_X=None,
    replay_Y=None,
    replay_fraction=0.25,
):
    """Fixed-time Stability-First: slow backbone + protected interface + optional replay."""
    model.train()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    backbone_p = list(model.fc1.parameters()) + list(model.fc2.parameters())
    backbone_ref = list(ref.fc1.parameters()) + list(ref.fc2.parameters())

    head_p = list(model.fc3.parameters())
    head_ref = list(ref.fc3.parameters())

    print(f"\n>>> Train (Stability-First fixed): {title}")
    print(f"    lambda_backbone={lambda_backbone} | replay={use_replay} | freeze_head_rows={freeze_head_rows}")

    for ep in range(1, epochs+1):
        total_s = task_s = rep_s = reg_s = 0.0

        for xb, yb in loader_new:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            loss_task = loss_fn(model(xb), yb)

            loss_rep = 0.0
            if use_replay:
                assert replay_X is not None and replay_Y is not None
                n = max(1, int(xb.size(0) * replay_fraction))
                xr, yr = sample_replay_batch(replay_X, replay_Y, n)
                xr, yr = xr.to(device), yr.to(device)
                loss_rep = loss_fn(model(xr), yr)

            reg_back = lambda_backbone * _l2_delta(backbone_p, backbone_ref)
            loss = loss_task + loss_rep + reg_back
            loss.backward()

            if freeze_head_rows is not None:
                w_grad = model.fc3.weight.grad
                b_grad = model.fc3.bias.grad
                if w_grad is not None:
                    w_grad[freeze_head_rows].zero_()
                if b_grad is not None:
                    b_grad[freeze_head_rows].zero_()

            opt.step()

            total_s += float(loss.detach().cpu())
            task_s += float(loss_task.detach().cpu())
            rep_s += float(loss_rep.detach().cpu()) if use_replay else 0.0
            reg_s += float(reg_back.detach().cpu())

        d_back = _delta_norm(backbone_p, backbone_ref)
        d_head = _delta_norm(head_p, head_ref)
        print(
            f"epoch {ep}/{epochs} total={total_s/len(loader_new):.4f} "
            f"task={task_s/len(loader_new):.4f} replay={rep_s/len(loader_new):.4f} reg={reg_s/len(loader_new):.4f} "
            f"| dW_back={d_back:.4f} dW_head={d_head:.4f}"
        )

def train_fractal_time(
    model,
    ref,
    loader_new,
    device,
    epochs,
    lr,
    title,
    lambda_fc1=10000.0,
    lambda_fc2=3000.0,
    lambda_head=0.0,
    use_replay=True,
    replay_X=None,
    replay_Y=None,
    replay_fraction=0.25,
):
    """Fractal / layer-wise time: different λ per layer group."""
    model.train()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    fc1_p = list(model.fc1.parameters()); fc1_ref = list(ref.fc1.parameters())
    fc2_p = list(model.fc2.parameters()); fc2_ref = list(ref.fc2.parameters())
    head_p = list(model.fc3.parameters()); head_ref = list(ref.fc3.parameters())

    print(f"\n>>> Train (Fractal time): {title}")
    print(f"    lambda_fc1={lambda_fc1} | lambda_fc2={lambda_fc2} | lambda_head={lambda_head} | replay={use_replay}")

    for ep in range(1, epochs+1):
        total_s = task_s = rep_s = reg_s = 0.0
        for xb, yb in loader_new:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            loss_task = loss_fn(model(xb), yb)

            loss_rep = 0.0
            if use_replay:
                assert replay_X is not None and replay_Y is not None
                n = max(1, int(xb.size(0) * replay_fraction))
                xr, yr = sample_replay_batch(replay_X, replay_Y, n)
                xr, yr = xr.to(device), yr.to(device)
                loss_rep = loss_fn(model(xr), yr)

            reg = (
                lambda_fc1 * _l2_delta(fc1_p, fc1_ref) +
                lambda_fc2 * _l2_delta(fc2_p, fc2_ref) +
                lambda_head * _l2_delta(head_p, head_ref)
            )

            loss = loss_task + loss_rep + reg
            loss.backward()
            opt.step()

            total_s += float(loss.detach().cpu())
            task_s += float(loss_task.detach().cpu())
            rep_s += float(loss_rep.detach().cpu()) if use_replay else 0.0
            reg_s += float(reg.detach().cpu())

        d1 = _delta_norm(fc1_p, fc1_ref); d2 = _delta_norm(fc2_p, fc2_ref); dh = _delta_norm(head_p, head_ref)
        print(
            f"epoch {ep}/{epochs} total={total_s/len(loader_new):.4f} task={task_s/len(loader_new):.4f} "
            f"replay={rep_s/len(loader_new):.4f} reg={reg_s/len(loader_new):.4f} | dW_fc1={d1:.4f} dW_fc2={d2:.4f} dW_head={dh:.4f}"
        )

def train_adaptive_time_pain(
    model,
    ref,
    loader_new,
    device,
    epochs,
    lr,
    title,
    lambda_min=100.0,
    lambda_max=20000.0,
    replay_X=None,
    replay_Y=None,
    replay_fraction=0.25,
    eps=1e-8,
):
    """Adaptive time (pain): λ_backbone depends on cosine between gradients of new vs old losses."""
    assert replay_X is not None and replay_Y is not None

    model.train()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    backbone_p = list(model.fc1.parameters()) + list(model.fc2.parameters())
    backbone_ref = list(ref.fc1.parameters()) + list(ref.fc2.parameters())

    print(f"\n>>> Train (Adaptive time / pain): {title}")
    print(f"    lambda_min={lambda_min} | lambda_max={lambda_max} | replay_fraction={replay_fraction}")

    for ep in range(1, epochs+1):
        total_s = new_s = old_s = reg_s = 0.0
        cos_s = lam_s = 0.0
        n_batches = 0

        for xb, yb in loader_new:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            n = max(1, int(xb.size(0) * replay_fraction))
            xr, yr = sample_replay_batch(replay_X, replay_Y, n)
            xr, yr = xr.to(device), yr.to(device)

            loss_new = loss_fn(model(xb), yb)
            loss_old = loss_fn(model(xr), yr)

            g_new = torch.autograd.grad(loss_new, backbone_p, retain_graph=True)
            g_old = torch.autograd.grad(loss_old, backbone_p, retain_graph=True)

            g_new_flat = torch.cat([gi.detach().flatten() for gi in g_new])
            g_old_flat = torch.cat([gi.detach().flatten() for gi in g_old])

            dot = torch.dot(g_new_flat, g_old_flat).item()
            n1 = (g_new_flat.pow(2).sum().item() ** 0.5) + eps
            n2 = (g_old_flat.pow(2).sum().item() ** 0.5) + eps
            cos = dot / (n1 * n2)

            pain = (1.0 - cos) * 0.5
            pain = max(0.0, min(1.0, pain))
            lam = lambda_min + (lambda_max - lambda_min) * pain

            reg_back = lam * _l2_delta(backbone_p, backbone_ref)

            loss = loss_new + loss_old + reg_back
            loss.backward()
            opt.step()

            total_s += float(loss.detach().cpu())
            new_s += float(loss_new.detach().cpu())
            old_s += float(loss_old.detach().cpu())
            reg_s += float(reg_back.detach().cpu())
            cos_s += cos
            lam_s += lam
            n_batches += 1

        d_back = _delta_norm(backbone_p, backbone_ref)
        print(
            f"epoch {ep}/{epochs} total={total_s/n_batches:.4f} new={new_s/n_batches:.4f} old={old_s/n_batches:.4f} "
            f"reg={reg_s/n_batches:.4f} | mean cos={cos_s/n_batches:.3f} mean lambda={lam_s/n_batches:.1f} dW_back={d_back:.4f}"
        )

def recover_head_only(model, loader, device, epochs, lr, title, freeze_backbone=True):
    """Freeze backbone and train ONLY the head."""
    print(f"\n>>> Recovery (head-only): {title}")
    if freeze_backbone:
        for p in model.fc1.parameters(): p.requires_grad = False
        for p in model.fc2.parameters(): p.requires_grad = False

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
