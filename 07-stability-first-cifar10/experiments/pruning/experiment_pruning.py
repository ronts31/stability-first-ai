import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy
import time
import os

# --- CONFIGURATION ---
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 5
PRUNING_AMOUNT = 0.7  # Remove 70% of weights (zero them) - "Deep Cut"
RECOVERY_ITERS = 1000 # Lazarus v3

# Recovery parameters
W_CONS = 1.0
W_STAB = 0.5
W_ENT = 0.05
LR_RECOVERY = 1e-4
H0 = 1.5
EPSILON = 0.05

torch.set_num_threads(os.cpu_count() or 8)
print(f"[RUNNING] Running Pruning Experiment on {DEVICE}")

# 1. DATA AND MODEL
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0, pin_memory=False)

testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0, pin_memory=False)

class StabilityCNN(nn.Module):
    def __init__(self):
        super(StabilityCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate(model, loader, name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"[METRIC] {name} Accuracy: {acc:.2f}%")
    return acc

# --- NEW FUNCTION: PRUNING (SURGERY) ---
def prune_weights(model, amount=0.3):
    """
    L1 Unstructured Pruning: zero out amount% of weights with smallest absolute values
    """
    print(f"[PRUNING] Pruning {amount*100:.0f}% of lowest weights...")
    
    # 1. Collect all weights into one list (only layer weights, not bias)
    all_weights = []
    for param in model.parameters():
        if param.dim() > 1:  # Don't touch bias, only layer weights
            all_weights.append(param.view(-1))
    
    if len(all_weights) == 0:
        print("[WARNING] No weights found for pruning!")
        return 0.0
    
    all_weights = torch.cat(all_weights)
    
    # 2. Find threshold
    # Sort by absolute value and take value at amount% boundary
    num_params = all_weights.numel()
    threshold_idx = int(num_params * amount)
    sorted_weights, _ = torch.sort(torch.abs(all_weights))
    threshold = sorted_weights[threshold_idx].item()
    
    print(f"   Threshold value: {threshold:.6f}")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Parameters to prune: {threshold_idx:,}")
    
    # 3. Zero weights below threshold
    pruned_count = 0
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                mask = torch.abs(param) >= threshold
                pruned_count += (~mask).sum().item()
                param.data.mul_(mask.float())  # Multiply by mask (0 or 1)
    
    print(f"   Actually pruned: {pruned_count:,} parameters ({100*pruned_count/num_params:.2f}%)")
    return threshold

def count_nonzero_weights(model):
    """Count nonzero weights"""
    nonzero = 0
    total = 0
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                nonzero += (param != 0).sum().item()
                total += param.numel()
    return nonzero, total

# --- LAZARUS PROTOCOL v3 (Unlabeled Anchor + Stability + Entropy Floor) ---
def lazarus_restore_v3(model, ref_model, unlabeled_loader,
                      iterations=1000, w_cons=1.0, w_stab=0.5, w_ent=0.05,
                      lr=1e-4, H0=1.5, epsilon=0.05):
    """
    Lazarus Protocol v3 for recovery after pruning
    Uses real images (without labels) as anchor
    """
    print(f"[LAZARUS] Initiating Protocol v3 (Healing Pruned Model)...")
    print(f"  Parameters: w_cons={w_cons}, w_stab={w_stab}, w_ent={w_ent}, lr={lr}, H0={H0}")
    
    model.train()
    ref_model.eval()
    opt = optim.Adam(model.parameters(), lr=lr)
    it = 0
    loader_iter = iter(unlabeled_loader)
    
    while it < iterations:
        try:
            images, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(unlabeled_loader)
            images, _ = next(loader_iter)
        images = images.to(DEVICE)
        opt.zero_grad()
        
        # Loss 1: Consistency (anchor on real images)
        with torch.no_grad():
            target = ref_model(images)
        out = model(images)
        loss_cons = F.mse_loss(out, target)
        
        # Loss 2: Stability (invariance to noise)
        noise = torch.randn_like(images) * epsilon
        out_pert = model(images + noise)
        loss_stab = F.mse_loss(out, out_pert)
        
        # Loss 3: Entropy Floor (collapse prevention)
        probs = F.softmax(out, dim=1)
        log_probs = F.log_softmax(out, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss_ent = F.relu(entropy - H0)
        
        loss = w_cons*loss_cons + w_stab*loss_stab + w_ent*loss_ent
        loss.backward()
        opt.step()
        
        if it % 200 == 0:
            print(f"   Iter {it}: cons={loss_cons.item():.4f} stab={loss_stab.item():.4f} "
                  f"H={entropy.item():.4f} total={loss.item():.4f}")
        it += 1

# --- RUN ---
if __name__ == "__main__":
    print("=" * 70)
    print("CIFAR-10 Pruning Experiment - Lazarus Effect")
    print("=" * 70)
    print()
    
    # A. Training
    print("--- Phase 1: Training (Time Emergence) ---")
    net = StabilityCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} complete. Loss: {running_loss / len(trainloader):.3f}")
    
    print(f"Training finished in {time.time() - start_time:.0f}s")
    acc_original = evaluate(net, testloader, "Original")
    
    # Count weights before pruning
    nonzero_before, total_before = count_nonzero_weights(net)
    print(f"[INFO] Weights before pruning: {nonzero_before:,}/{total_before:,} ({100*nonzero_before/total_before:.2f}% nonzero)")
    
    # Save reference model
    net_ref = copy.deepcopy(net)
    net_ref.eval()

    # B. Pruning (Catastrophe)
    print("\n--- Phase 2: Pruning (Lobotomy) ---")
    net_pruned = copy.deepcopy(net)
    threshold = prune_weights(net_pruned, amount=PRUNING_AMOUNT)
    acc_pruned = evaluate(net_pruned, testloader, "Pruned")
    
    # Count weights after pruning
    nonzero_after, total_after = count_nonzero_weights(net_pruned)
    print(f"[INFO] Weights after pruning: {nonzero_after:,}/{total_after:,} ({100*nonzero_after/total_after:.2f}% nonzero)")
    print(f"[INFO] Pruning removed: {nonzero_before - nonzero_after:,} weights")

    # C. Resurrection
    print("\n--- Phase 3: Resurrection (Regrowth) ---")
    net_resurrected = copy.deepcopy(net_pruned)
    lazarus_restore_v3(net_resurrected, net_ref, trainloader,
                      iterations=RECOVERY_ITERS,
                      w_cons=W_CONS, w_stab=W_STAB, w_ent=W_ENT,
                      lr=LR_RECOVERY, H0=H0, epsilon=EPSILON)
    
    acc_resurrected = evaluate(net_resurrected, testloader, "Resurrected")
    
    # Count weights after restoration
    nonzero_restored, total_restored = count_nonzero_weights(net_resurrected)
    print(f"[INFO] Weights after restoration: {nonzero_restored:,}/{total_restored:,} ({100*nonzero_restored/total_restored:.2f}% nonzero)")
    
    # Check if zeroed weights were restored
    regrown = nonzero_restored - nonzero_after
    print(f"[INFO] Regrown weights: {regrown:,} ({(100*regrown/(nonzero_before - nonzero_after)):.1f}% of pruned)")

    # D. Results
    print("\n" + "=" * 70)
    print("--- FINAL RESULTS ---")
    print("=" * 70)
    print(f"Original:    {acc_original:.2f}%")
    print(f"Pruned:      {acc_pruned:.2f}% (drop: {acc_original - acc_pruned:.2f}%)")
    print(f"Resurrected: {acc_resurrected:.2f}%")
    
    delta = acc_resurrected - acc_pruned
    recovery_pct = ((acc_resurrected - acc_pruned) / (acc_original - acc_pruned) * 100) if (acc_original - acc_pruned) > 0 else 0
    
    print(f"\nRecovery: {delta:+.2f}% accuracy")
    if recovery_pct > 0:
        print(f"Recovery rate: {recovery_pct:.1f}% of lost accuracy")
    
    print(f"\nWeight Analysis:")
    print(f"  Pruned: {nonzero_before - nonzero_after:,} weights removed")
    print(f"  Regrown: {regrown:,} weights restored")
    print(f"  Regrowth rate: {100*regrown/(nonzero_before - nonzero_after):.1f}%")
    
    if delta > 0:
        print(f"\n[SUCCESS] Lazarus Effect recovered +{delta:.2f}% accuracy without data!")
        if regrown > 0:
            print(f"[SUCCESS] Model regrew {regrown:,} weights!")
    else:
        print(f"\n[FAIL] Pruning was permanent - no recovery.")
    print("=" * 70)

