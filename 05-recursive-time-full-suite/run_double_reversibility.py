import copy
import numpy as np
import torch

from src.data import get_mnist, indices_for_classes, make_loader, build_replay_buffer, tensor_loader
from src.model import SimpleMLP
from src.train import eval_acc, train_plain, recover_head_only

def train_destruction(model, ref, loader, device, epochs, lr, lambda_backbone, lambda_head):
    import torch.nn as nn
    import torch.optim as optim
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    backbone_p = list(model.fc1.parameters()) + list(model.fc2.parameters())
    backbone_ref = list(ref.fc1.parameters()) + list(ref.fc2.parameters())
    head_p = list(model.fc3.parameters())
    head_ref = list(ref.fc3.parameters())

    def l2_delta(ps, ps0):
        s = 0.0
        for p, p0 in zip(ps, ps0):
            s = s + (p - p0).pow(2).sum()
        return s

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_task = loss_fn(model(xb), yb)
            reg_back = lambda_backbone * l2_delta(backbone_p, backbone_ref)
            reg_head = lambda_head * l2_delta(head_p, head_ref)
            loss = loss_task + reg_back + reg_head
            loss.backward()
            opt.step()

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    A = [0,1,2,3]
    B = [4,5,6]
    C = [7,8,9]

    train_ds, test_ds = get_mnist(root="./data")

    idxA_tr = indices_for_classes(train_ds, A)
    idxB_tr = indices_for_classes(train_ds, B)
    idxC_tr = indices_for_classes(train_ds, C)

    idxA_te = indices_for_classes(test_ds, A)
    idxB_te = indices_for_classes(test_ds, B)
    idxC_te = indices_for_classes(test_ds, C)

    trainA = make_loader(train_ds, idxA_tr, batch_size=64, shuffle=True)
    trainB = make_loader(train_ds, idxB_tr, batch_size=64, shuffle=True)
    trainC = make_loader(train_ds, idxC_tr, batch_size=64, shuffle=True)

    testA  = make_loader(test_ds, idxA_te, batch_size=1000, shuffle=False, num_workers=0, pin_memory=False)
    testB  = make_loader(test_ds, idxB_te, batch_size=1000, shuffle=False, num_workers=0, pin_memory=False)
    testC  = make_loader(test_ds, idxC_te, batch_size=1000, shuffle=False, num_workers=0, pin_memory=False)

    recA_X, recA_Y = build_replay_buffer(train_ds, idxA_tr, k=50, seed=seed)
    recB_X, recB_Y = build_replay_buffer(train_ds, idxB_tr, k=50, seed=seed+1)
    recA = tensor_loader(recA_X, recA_Y, batch_size=10, shuffle=True)
    recB = tensor_loader(recB_X, recB_Y, batch_size=10, shuffle=True)

    model = SimpleMLP().to(device)

    print("\n>>> PHASE 1: Train A")
    train_plain(model, trainA, device=device, epochs=3, lr=1e-3, title="Task A (0-3)")
    accA1 = eval_acc(model, testA, device=device, title="Acc A (after A)")
    accB1 = eval_acc(model, testB, device=device, title="Acc B (after A)")
    accC1 = eval_acc(model, testC, device=device, title="Acc C (after A)")

    ref = copy.deepcopy(model)
    for p in ref.parameters(): p.requires_grad = False

    print("\n>>> PHASE 2: Destruction on B (preserve backbone, drift head)")
    train_destruction(model, ref, trainB, device=device, epochs=3, lr=1e-3, lambda_backbone=10000.0, lambda_head=0.0)
    accA2 = eval_acc(model, testA, device=device, title="Acc A (after B)")
    accB2 = eval_acc(model, testB, device=device, title="Acc B (after B)")
    accC2 = eval_acc(model, testC, device=device, title="Acc C (after B)")

    print("\n>>> PHASE 3: Destruction on C (continue drift)")
    train_destruction(model, ref, trainC, device=device, epochs=3, lr=1e-3, lambda_backbone=10000.0, lambda_head=0.0)
    accA3 = eval_acc(model, testA, device=device, title="Acc A (after C)")
    accB3 = eval_acc(model, testB, device=device, title="Acc B (after C)")
    accC3 = eval_acc(model, testC, device=device, title="Acc C (after C)")

    print("\n>>> PHASE 4: Recover A (head-only)")
    recover_head_only(model, recA, device=device, epochs=20, lr=1e-3, title="Recover A from 50 samples")
    accA4 = eval_acc(model, testA, device=device, title="Acc A (after recover A)")
    accB4 = eval_acc(model, testB, device=device, title="Acc B (after recover A)")
    accC4 = eval_acc(model, testC, device=device, title="Acc C (after recover A)")

    print("\n>>> PHASE 5: Recover B (head-only)")
    recover_head_only(model, recB, device=device, epochs=20, lr=1e-3, title="Recover B from 50 samples")
    accA5 = eval_acc(model, testA, device=device, title="Acc A (after recover B)")
    accB5 = eval_acc(model, testB, device=device, title="Acc B (after recover B)")
    accC5 = eval_acc(model, testC, device=device, title="Acc C (after recover B)")

    print("\n" + "="*72)
    print("RESULTS: Double Reversibility (A → B → C → recover A → recover B)")
    print("="*72)
    print(f"{'State':<22} | {'Acc A':>8} | {'Acc B':>8} | {'Acc C':>8}")
    print("-"*72)
    print(f"{'After A':<22} | {accA1:>7.2f}% | {accB1:>7.2f}% | {accC1:>7.2f}%")
    print(f"{'After B':<22} | {accA2:>7.2f}% | {accB2:>7.2f}% | {accC2:>7.2f}%")
    print(f"{'After C':<22} | {accA3:>7.2f}% | {accB3:>7.2f}% | {accC3:>7.2f}%")
    print(f"{'After recover A':<22} | {accA4:>7.2f}% | {accB4:>7.2f}% | {accC4:>7.2f}%")
    print(f"{'After recover B':<22} | {accA5:>7.2f}% | {accB5:>7.2f}% | {accC5:>7.2f}%")
    print("-"*72)

if __name__ == "__main__":
    main()
