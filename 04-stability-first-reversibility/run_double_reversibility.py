import copy
from dataclasses import dataclass

import numpy as np
import torch

from src.data import get_mnist, indices_for_classes, make_loader, build_replay_buffer, tensor_loader
from src.model import SimpleMLP
from src.train import eval_acc, train_plain, train_with_time_constraints, recover_head_only

@dataclass
class Config:
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3

    epochs_a: int = 3
    epochs_b: int = 3
    epochs_c: int = 3

    # Keep backbone almost fixed during B and C (slow time)
    lambda_backbone: float = 10000.0

    # Allow head drift (fast time) to destroy interfaces
    lambda_head: float = 0.0

    # Recovery buffers
    recovery_k_a: int = 50
    recovery_k_b: int = 50
    recovery_batch: int = 10
    recovery_epochs: int = 20

    # Tasks (classes)
    task_a = (0, 1, 2, 3)
    task_b = (4, 5, 6)
    task_c = (7, 8, 9)

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, test_ds = get_mnist(root="./data")

    idx_a_train = indices_for_classes(train_ds, cfg.task_a)
    idx_b_train = indices_for_classes(train_ds, cfg.task_b)
    idx_c_train = indices_for_classes(train_ds, cfg.task_c)

    idx_a_test = indices_for_classes(test_ds, cfg.task_a)
    idx_b_test = indices_for_classes(test_ds, cfg.task_b)
    idx_c_test = indices_for_classes(test_ds, cfg.task_c)

    train_A = make_loader(train_ds, idx_a_train, batch_size=cfg.batch_size, shuffle=True)
    train_B = make_loader(train_ds, idx_b_train, batch_size=cfg.batch_size, shuffle=True)
    train_C = make_loader(train_ds, idx_c_train, batch_size=cfg.batch_size, shuffle=True)

    test_A = make_loader(test_ds, idx_a_test, batch_size=1000, shuffle=False)
    test_B = make_loader(test_ds, idx_b_test, batch_size=1000, shuffle=False)
    test_C = make_loader(test_ds, idx_c_test, batch_size=1000, shuffle=False)

    # Tiny episodic buffers for recovery (A and B)
    recA_X, recA_Y = build_replay_buffer(train_ds, idx_a_train, k=cfg.recovery_k_a, seed=cfg.seed)
    recB_X, recB_Y = build_replay_buffer(train_ds, idx_b_train, k=cfg.recovery_k_b, seed=cfg.seed + 1)

    recA_loader = tensor_loader(recA_X, recA_Y, batch_size=cfg.recovery_batch, shuffle=True)
    recB_loader = tensor_loader(recB_X, recB_Y, batch_size=cfg.recovery_batch, shuffle=True)

    model = SimpleMLP().to(device)

    print("\n>>> PHASE 1: Train A")
    train_plain(model, train_A, device=device, epochs=cfg.epochs_a, lr=cfg.lr, title=f"Task A {cfg.task_a}")
    accA_1 = eval_acc(model, test_A, device=device, title="Acc A (after A)")
    accB_1 = eval_acc(model, test_B, device=device, title="Acc B (after A)")
    accC_1 = eval_acc(model, test_C, device=device, title="Acc C (after A)")

    # Reference snapshot after A (time anchor)
    ref_after_A = copy.deepcopy(model)
    ref_after_A.eval()
    for p in ref_after_A.parameters():
        p.requires_grad = False

    print("\n>>> PHASE 2: Train B (interface drift allowed)")
    train_with_time_constraints(
        model,
        ref=ref_after_A,
        loader=train_B,
        device=device,
        epochs=cfg.epochs_b,
        lr=cfg.lr,
        title=f"Task B {cfg.task_b}",
        lambda_backbone=cfg.lambda_backbone,
        lambda_head=cfg.lambda_head,
        use_replay=False,
        freeze_head_rows=None,
    )
    accA_2 = eval_acc(model, test_A, device=device, title="Acc A (after B)")
    accB_2 = eval_acc(model, test_B, device=device, title="Acc B (after B)")
    accC_2 = eval_acc(model, test_C, device=device, title="Acc C (after B)")

    print("\n>>> PHASE 3: Train C (continue interface drift)")
    train_with_time_constraints(
        model,
        ref=ref_after_A,
        loader=train_C,
        device=device,
        epochs=cfg.epochs_c,
        lr=cfg.lr,
        title=f"Task C {cfg.task_c}",
        lambda_backbone=cfg.lambda_backbone,
        lambda_head=cfg.lambda_head,
        use_replay=False,
        freeze_head_rows=None,
    )
    accA_3 = eval_acc(model, test_A, device=device, title="Acc A (after C)")
    accB_3 = eval_acc(model, test_B, device=device, title="Acc B (after C)")
    accC_3 = eval_acc(model, test_C, device=device, title="Acc C (after C)")

    print("\n>>> PHASE 4: Recover A (head-only on tiny buffer)")
    recover_head_only(model, recA_loader, device=device, epochs=cfg.recovery_epochs, lr=cfg.lr, title=f"Recover A with {cfg.recovery_k_a} samples")
    accA_4 = eval_acc(model, test_A, device=device, title="Acc A (after recover A)")
    accB_4 = eval_acc(model, test_B, device=device, title="Acc B (after recover A)")
    accC_4 = eval_acc(model, test_C, device=device, title="Acc C (after recover A)")

    print("\n>>> PHASE 5: Recover B (head-only on tiny buffer)")
    recover_head_only(model, recB_loader, device=device, epochs=cfg.recovery_epochs, lr=cfg.lr, title=f"Recover B with {cfg.recovery_k_b} samples")
    accA_5 = eval_acc(model, test_A, device=device, title="Acc A (after recover B)")
    accB_5 = eval_acc(model, test_B, device=device, title="Acc B (after recover B)")
    accC_5 = eval_acc(model, test_C, device=device, title="Acc C (after recover B)")

    print("\n" + "="*72)
    print("RESULTS: Double Reversibility (A → B → C → recover A → recover B)")
    print("="*72)
    print(f"{'State':<22} | {'Acc A':>8} | {'Acc B':>8} | {'Acc C':>8}")
    print("-"*72)
    print(f"{'After A':<22} | {accA_1:>7.2f}% | {accB_1:>7.2f}% | {accC_1:>7.2f}%")
    print(f"{'After B':<22} | {accA_2:>7.2f}% | {accB_2:>7.2f}% | {accC_2:>7.2f}%")
    print(f"{'After C':<22} | {accA_3:>7.2f}% | {accB_3:>7.2f}% | {accC_3:>7.2f}%")
    print(f"{'After recover A':<22} | {accA_4:>7.2f}% | {accB_4:>7.2f}% | {accC_4:>7.2f}%")
    print(f"{'After recover B':<22} | {accA_5:>7.2f}% | {accB_5:>7.2f}% | {accC_5:>7.2f}%")
    print("-"*72)

if __name__ == "__main__":
    main()
