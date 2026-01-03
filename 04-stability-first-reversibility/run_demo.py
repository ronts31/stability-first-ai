import copy
from dataclasses import dataclass

import numpy as np
import torch

from src.data import split_mnist_loaders, get_mnist, build_replay_buffer
from src.model import SimpleMLP
from src.train import eval_acc, train_plain, train_with_time_constraints

@dataclass
class Config:
    seed: int = 42
    batch_size: int = 128
    lr: float = 1e-3
    epochs_a: int = 3
    epochs_b: int = 5

    # Stability-First params
    lambda_backbone: float = 2000.0
    lambda_head: float = 0.0  # keep 0; we protect old interface by freezing head rows

    # Episodic replay
    use_replay: bool = True
    replay_k: int = 800
    replay_fraction: float = 0.25

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    task_a = [0,1,2,3,4]
    task_b = [5,6,7,8,9]

    print("\n1) Loading Split-MNIST ...")
    train_a, test_a, train_b, test_b, idx_a_train, train_ds = split_mnist_loaders(task_a, task_b, batch_size=cfg.batch_size)

    print("\n2) Training on Task A (0-4) ...")
    base = SimpleMLP().to(device)
    train_plain(base, train_a, device=device, epochs=cfg.epochs_a, lr=cfg.lr, title="Task A (0-4)")
    acc_a_before = eval_acc(base, test_a, device=device, title="Test A before B")

    print("\n" + "="*74)
    print("SCENARIO 1: Baseline (naive fine-tuning on Task B)")
    print("="*74)
    sc1 = copy.deepcopy(base)
    train_plain(sc1, train_b, device=device, epochs=cfg.epochs_b, lr=cfg.lr, title="Task B (5-9), naive")
    acc_b_sc1 = eval_acc(sc1, test_b, device=device, title="Test B after B (baseline)")
    acc_a_sc1 = eval_acc(sc1, test_a, device=device, title="Test A after B (baseline, forgetting)")

    print("\n" + "="*74)
    print("SCENARIO 2: Stability-First (slow backbone time + protected interface + replay)")
    print("="*74)
    sc2 = copy.deepcopy(base)
    ref = copy.deepcopy(base)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    replay_X = replay_Y = None
    if cfg.use_replay:
        replay_X, replay_Y = build_replay_buffer(train_ds, idx_a_train, k=cfg.replay_k, seed=cfg.seed)
        print(f"Replay buffer: X={tuple(replay_X.shape)} Y={tuple(replay_Y.shape)}")

    freeze_rows = list(range(0, 5))  # protect old classes 0–4 in head

    train_with_time_constraints(
        sc2,
        ref=ref,
        loader=train_b,
        device=device,
        epochs=cfg.epochs_b,
        lr=cfg.lr,
        title="Task B (5-9), Stability-First",
        lambda_backbone=cfg.lambda_backbone,
        lambda_head=cfg.lambda_head,
        use_replay=cfg.use_replay,
        replay_X=replay_X,
        replay_Y=replay_Y,
        replay_fraction=cfg.replay_fraction,
        freeze_head_rows=freeze_rows,
    )

    acc_b_sc2 = eval_acc(sc2, test_b, device=device, title="Test B after B (stability-first)")
    acc_a_sc2 = eval_acc(sc2, test_a, device=device, title="Test A after B (stability-first, retained)")

    print("\n" + "#"*82)
    print(f"{'METRIC':<30} | {'Baseline':<16} | {'Stability-First':<16}")
    print("-"*82)
    print(f"{'Task A (0-4) before B':<30} | {acc_a_before:>14.2f}% | {acc_a_before:>14.2f}%")
    print(f"{'Task B (5-9) after B':<30} | {acc_b_sc1:>14.2f}% | {acc_b_sc2:>14.2f}%")
    print(f"{'Task A (0-4) after B':<30} | {acc_a_sc1:>14.2f}% | {acc_a_sc2:>14.2f}%")
    print("#"*82)

if __name__ == "__main__":
    main()
