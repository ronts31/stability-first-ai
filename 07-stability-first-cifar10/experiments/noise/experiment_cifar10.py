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
DEVICE = torch.device("cpu") # i9-14900K (24 cores: 8P + 16E)
BATCH_SIZE = 64
EPOCHS = 5           # Enough for test to get ~60-70%
DAMAGE_ALPHA = 0.35  # Relative damage level (0.35 for strong stress test)
RECOVERY_ITERS = 1000 # Recovery iterations (Lazarus phase)

# Recovery parameters
W_CONS = 1.0    # Consistency weight (anchor)
W_STAB = 0.5    # Stability weight
W_ENT = 0.05    # Entropy floor weight
LR_RECOVERY = 1e-4  # Learning rate for recovery
H0 = 1.5        # Entropy threshold (entropy floor)
EPSILON = 0.05  # Noise level for stability

# Optimization for i9-14900K
# Use all available threads for PyTorch
torch.set_num_threads(os.cpu_count() or 8)
print(f"[RUNNING] Running on {DEVICE} with CIFAR-10")
print(f"[INFO] CPU threads: {torch.get_num_threads()}")
print(f"[INFO] PyTorch version: {torch.__version__}")

# 1. DATA (CIFAR-10)
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

# 2. ARCHITECTURE (Small CNN)
# Large enough for statistics, light enough for CPU
class StabilityCNN(nn.Module):
    def __init__(self):
        super(StabilityCNN, self).__init__()
        # Convolution 1: 3 channels (RGB) -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Convolution 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Fully Connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256) # 64 channels, 8x8 image
        self.fc2 = nn.Linear(256, 10)         # 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. HELPER FUNCTIONS
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

def damage_weights_relative(model, alpha=0.1, eps=1e-12):
    """
    Relative weight damage (more physical).
    alpha - fraction of weight norm (0.02, 0.05, 0.1, 0.2)
    """
    print(f"[DAMAGE] Applying relative damage (alpha={alpha})...")
    with torch.no_grad():
        for p in model.parameters():
            w = p.norm().clamp_min(eps)
            z = torch.randn_like(p)
            z = z / z.norm().clamp_min(eps)
            p.add_(z * (alpha * w))

# 4. STABILITY OPERATOR (LAZARUS EFFECT) - Version 3.0: Unlabeled Anchor + Stability + Entropy Floor
def lazarus_restore_v3(model, ref_model, unlabeled_loader,
                      iterations=1000, w_cons=1.0, w_stab=0.5, w_ent=0.05,
                      lr=1e-4, H0=1.5, epsilon=0.05, device=DEVICE):
    """
    Without labels:
    - Consistency: keep logits same as ref_model on real images
    - Stability: invariance to small input perturbations
    - Entropy floor: don't let entropy become too large, but don't force it to 0
    """
    print(f"[LAZARUS] Initiating Protocol v3 (Unlabeled Anchor + Stability + Entropy Floor)...")
    print(f"  Parameters: w_cons={w_cons}, w_stab={w_stab}, w_ent={w_ent}, lr={lr}, H0={H0}")
    
    model.train()
    ref_model.eval()
    opt = optim.Adam(model.parameters(), lr=lr)

    it = 0
    loader_iter = iter(unlabeled_loader)

    while it < iterations:
        try:
            images, _ = next(loader_iter)   # ignore labels
        except StopIteration:
            loader_iter = iter(unlabeled_loader)
            images, _ = next(loader_iter)

        images = images.to(device)

        opt.zero_grad()

        with torch.no_grad():
            target = ref_model(images)

        out = model(images)
        loss_cons = F.mse_loss(out, target)

        # stability via input noise / weak augmentation
        noise = torch.randn_like(images) * epsilon
        out_pert = model(images + noise)
        loss_stab = F.mse_loss(out, out_pert)

        # entropy floor (penalize only if entropy is too high)
        probs = F.softmax(out, dim=1)
        log_probs = F.log_softmax(out, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss_ent = F.relu(entropy - H0)

        loss = w_cons*loss_cons + w_stab*loss_stab + w_ent*loss_ent
        loss.backward()
        opt.step()

        if it % 100 == 0:
            print(f"   Iter {it}: cons={loss_cons.item():.4f} stab={loss_stab.item():.4f} "
                  f"H={entropy.item():.4f} ent_pen={loss_ent.item():.4f} total={loss.item():.4f}")
        it += 1

# 5. ADDITIONAL METRICS
def compute_entropy(model, loader, device=DEVICE):
    """Computes average entropy on dataset"""
    model.eval()
    entropies = []
    with torch.no_grad():
        for data in loader:
            images = data[0].to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            entropies.append(entropy)
    return torch.cat(entropies).mean().item()

def compute_agreement(model1, model2, loader, device=DEVICE):
    """Computes agreement between two models (MSE logits)"""
    model1.eval()
    model2.eval()
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for data in loader:
            images = data[0].to(device)
            out1 = model1(images)
            out2 = model2(images)
            mse_sum += F.mse_loss(out1, out2, reduction='sum').item()
            count += images.size(0)
    return mse_sum / count

# --- RUN EXPERIMENT ---
if __name__ == "__main__":
    print("=" * 70)
    print("CIFAR-10 Stability Experiment - Lazarus Effect")
    print("=" * 70)
    print()
    
    # A. Initialization
    net = StabilityCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # B. Training (Time Emergence)
    print("\n--- Phase 1: Training (Time Emergence) ---")
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
    
    # Save reference model BEFORE damage
    net_ref = copy.deepcopy(net)
    net_ref.eval()

    # C. Catastrophe (Collapse of Time)
    print("\n--- Phase 2: Catastrophe (Damage) ---")
    # Clone for experiment purity
    net_damaged = copy.deepcopy(net)
    damage_weights_relative(net_damaged, alpha=DAMAGE_ALPHA)
    acc_damaged = evaluate(net_damaged, testloader, "Damaged")
    
    # Damaged model metrics
    entropy_damaged = compute_entropy(net_damaged, testloader)
    agreement_damaged = compute_agreement(net_damaged, net_ref, testloader)
    print(f"[METRIC] Damaged - Entropy: {entropy_damaged:.4f}, Agreement (MSE): {agreement_damaged:.4f}")

    # D. Recovery (Lazarus Effect)
    print("\n--- Phase 3: Resurrection (Stability Force) ---")
    # Use trainloader WITHOUT labels (only images)
    net_resurrected = copy.deepcopy(net_damaged)
    lazarus_restore_v3(net_resurrected, net_ref, trainloader,
                      iterations=RECOVERY_ITERS,
                      w_cons=W_CONS, w_stab=W_STAB, w_ent=W_ENT,
                      lr=LR_RECOVERY, H0=H0, epsilon=EPSILON, device=DEVICE)
    
    acc_resurrected = evaluate(net_resurrected, testloader, "Resurrected")
    
    # Resurrected model metrics
    entropy_resurrected = compute_entropy(net_resurrected, testloader)
    agreement_resurrected = compute_agreement(net_resurrected, net_ref, testloader)
    print(f"[METRIC] Resurrected - Entropy: {entropy_resurrected:.4f}, Agreement (MSE): {agreement_resurrected:.4f}")

    # E. Summary
    print("\n" + "=" * 70)
    print("--- FINAL RESULTS ---")
    print("=" * 70)
    print(f"Original:    {acc_original:.2f}%")
    print(f"Damaged:     {acc_damaged:.2f}%")
    print(f"Resurrected: {acc_resurrected:.2f}%")
    
    delta = acc_resurrected - acc_damaged
    recovery_pct = ((acc_resurrected - acc_damaged) / (acc_original - acc_damaged) * 100) if (acc_original - acc_damaged) > 0 else 0
    
    print(f"\n--- Recovery Metrics ---")
    print(f"Accuracy recovery: {delta:+.2f}%")
    if recovery_pct > 0:
        print(f"Recovery rate: {recovery_pct:.1f}% of lost accuracy")
    
    print(f"\n--- Entropy Analysis ---")
    print(f"Damaged entropy:     {entropy_damaged:.4f}")
    print(f"Resurrected entropy:  {entropy_resurrected:.4f}")
    entropy_change = entropy_resurrected - entropy_damaged
    if entropy_resurrected < 0.5 and acc_resurrected < 20:
        print(f"⚠️  WARNING: Low entropy ({entropy_resurrected:.4f}) with low accuracy - possible collapse!")
    else:
        print(f"Entropy change: {entropy_change:+.4f}")
    
    print(f"\n--- Agreement Analysis ---")
    print(f"Damaged agreement (MSE):     {agreement_damaged:.4f}")
    print(f"Resurrected agreement (MSE): {agreement_resurrected:.4f}")
    agreement_improvement = agreement_damaged - agreement_resurrected
    print(f"Agreement improvement: {agreement_improvement:+.4f} (lower is better)")
    
    print(f"\n--- Conclusion ---")
    if delta > 0:
        print(f"[SUCCESS] Lazarus Effect recovered +{delta:.2f}% accuracy!")
        if agreement_improvement > 0:
            print(f"[SUCCESS] Agreement with reference improved by {agreement_improvement:.4f}")
    else:
        print(f"[FAIL] Stability operator didn't help.")
    print("=" * 70)

