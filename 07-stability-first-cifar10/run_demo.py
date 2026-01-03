import copy
from dataclasses import dataclass

import torch

from src.data import get_split_cifar10_loaders, build_replay_buffer_from_loader
from src.model import SimpleCNN
from src.train import train_standard, train_stability_first_recursive_time, eval_acc

@dataclass
class Config:
    batch_size: int = 64  # Reduced for CPU
    lr: float = 1e-3
    epochs_a: int = 15  # Increased for better training
    epochs_b: int = 15  # Increased for Task B

    # Slow system time for parametric memory (backbone)
    # Reduced for CIFAR-10 - too strong regularization interfered with Task B training
    lambda_backbone: float = 500.0  # Was 2000.0 - too high

    # Protect old interface (head rows 0–4)
    freeze_old_head_rows: bool = True

    # Episodic memory (replay)
    use_replay: bool = True
    replay_k: int = 1000  # More for CIFAR-10
    replay_fraction: float = 0.3  # Slightly increased

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("\n" + "="*80)
    print("STABILITY-FIRST EXPERIMENT: CIFAR-10")
    print("="*80)

    print("\n1) Loading Split-CIFAR-10 ...")
    train_a, test_a, train_b, test_b = get_split_cifar10_loaders(batch_size=cfg.batch_size)

    print("\n2) Training on Task A (0-4) ...")
    base = SimpleCNN(num_classes=10).to(device)
    train_standard(base, train_a, device=device, epochs=cfg.epochs_a, lr=cfg.lr, title="Task A (0-4)")
    acc_a_before = eval_acc(base, test_a, device=device, title="Test A before B")

    ref = copy.deepcopy(base)

    replay_X = replay_Y = None
    if cfg.use_replay:
        replay_X, replay_Y = build_replay_buffer_from_loader(train_a, k=cfg.replay_k)
        print(f"\nBuilt replay buffer from Task A: X={tuple(replay_X.shape)}, Y={tuple(replay_Y.shape)}")

    print("\n" + "="*74)
    print("SCENARIO 1: Baseline (naive fine-tuning on Task B)")
    print("="*74)
    sc1 = copy.deepcopy(base)
    train_standard(sc1, train_b, device=device, epochs=cfg.epochs_b, lr=cfg.lr, title="Task B (5-9), naive")
    acc_b_sc1 = eval_acc(sc1, test_b, device=device, title="Test B after B (baseline)")
    acc_a_sc1 = eval_acc(sc1, test_a, device=device, title="Test A after B (baseline, forgetting)")

    print("\n" + "="*74)
    print("SCENARIO 2: Stability-First (recursive-time framing)")
    print("="*74)
    sc2 = copy.deepcopy(base)
    train_stability_first_recursive_time(
        sc2,
        ref=ref,
        loader_b=train_b,
        device=device,
        epochs=cfg.epochs_b,
        lr=cfg.lr,
        title="Task B (5-9), Stability-First",
        lambda_backbone=cfg.lambda_backbone,
        freeze_old_head_rows=cfg.freeze_old_head_rows,
        use_replay=cfg.use_replay,
        replay_X=replay_X,
        replay_Y=replay_Y,
        replay_fraction=cfg.replay_fraction,
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

