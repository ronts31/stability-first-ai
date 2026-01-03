"""
Lazarus Effect Analysis: recovery curve and baseline comparison
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy
import time
import os
import json
from collections import defaultdict

# --- CONFIGURATION ---
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 5
RECOVERY_ITERS = 1000

# Recovery parameters
W_CONS = 1.0
W_STAB = 0.5
W_ENT = 0.05
LR_RECOVERY = 1e-4
H0 = 1.5
EPSILON = 0.05

# Tested damage levels
ALPHA_VALUES = [0.1, 0.2, 0.3]

torch.set_num_threads(os.cpu_count() or 8)

# 1. DATA
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

# 2. ARCHITECTURE
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

# 3. HELPER FUNCTIONS
def evaluate(model, loader):
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
    return 100 * correct / total

def damage_weights_relative(model, alpha=0.1, eps=1e-12):
    with torch.no_grad():
        for p in model.parameters():
            w = p.norm().clamp_min(eps)
            z = torch.randn_like(p)
            z = z / z.norm().clamp_min(eps)
            p.add_(z * (alpha * w))

def compute_entropy(model, loader):
    model.eval()
    entropies = []
    with torch.no_grad():
        for data in loader:
            images = data[0].to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            entropies.append(entropy)
    return torch.cat(entropies).mean().item()

def compute_agreement(model1, model2, loader):
    model1.eval()
    model2.eval()
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for data in loader:
            images = data[0].to(DEVICE)
            out1 = model1(images)
            out2 = model2(images)
            mse_sum += F.mse_loss(out1, out2, reduction='sum').item()
            count += images.size(0)
    return mse_sum / count

# 4. RECOVERY METHODS

def lazarus_restore_v3(model, ref_model, unlabeled_loader,
                      iterations=1000, w_cons=1.0, w_stab=0.5, w_ent=0.05,
                      lr=1e-4, H0=1.5, epsilon=0.05):
    """Full method: Consistency + Stability + Entropy Floor"""
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
        
        with torch.no_grad():
            target = ref_model(images)
        out = model(images)
        loss_cons = F.mse_loss(out, target)
        
        noise = torch.randn_like(images) * epsilon
        out_pert = model(images + noise)
        loss_stab = F.mse_loss(out, out_pert)
        
        probs = F.softmax(out, dim=1)
        log_probs = F.log_softmax(out, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss_ent = F.relu(entropy - H0)
        
        loss = w_cons*loss_cons + w_stab*loss_stab + w_ent*loss_ent
        loss.backward()
        opt.step()
        it += 1

def baseline_noise_only(model, unlabeled_loader, iterations=1000, lr=1e-4):
    """Baseline 1: Just noise without loss (no-op)"""
    # Do nothing - just return model as is
    pass

def baseline_entropy_only(model, ref_model, unlabeled_loader, iterations=1000, lr=1e-4, H0=1.5):
    """Baseline 2: Only Entropy Floor without Consistency and Stability"""
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
        
        out = model(images)
        probs = F.softmax(out, dim=1)
        log_probs = F.log_softmax(out, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss = F.relu(entropy - H0)
        
        loss.backward()
        opt.step()
        it += 1

def baseline_consistency_only(model, ref_model, unlabeled_loader, iterations=1000, lr=1e-4):
    """Baseline 3: Only Consistency without Stability and Entropy"""
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
        
        with torch.no_grad():
            target = ref_model(images)
        out = model(images)
        loss = F.mse_loss(out, target)
        
        loss.backward()
        opt.step()
        it += 1

# 5. MAIN ANALYSIS FUNCTION
def run_analysis():
    print("=" * 70)
    print("Lazarus Effect Analysis: Recovery Curve & Baseline Comparison")
    print("=" * 70)
    print()
    
    # Train base model
    print("--- Training Base Model ---")
    net = StabilityCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
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
    
    acc_original = evaluate(net, testloader)
    print(f"Original Accuracy: {acc_original:.2f}%\n")
    
    net_ref = copy.deepcopy(net)
    net_ref.eval()
    
    # Results
    results = defaultdict(dict)
    
    # Test different alpha
    print("--- Testing Different Damage Levels ---")
    for alpha in ALPHA_VALUES:
        print(f"\n[Alpha = {alpha}]")
        
        # Damage
        net_damaged = copy.deepcopy(net)
        damage_weights_relative(net_damaged, alpha=alpha)
        acc_damaged = evaluate(net_damaged, testloader)
        entropy_damaged = compute_entropy(net_damaged, testloader)
        agreement_damaged = compute_agreement(net_damaged, net_ref, testloader)
        
        print(f"  Damaged: {acc_damaged:.2f}% (drop: {acc_original - acc_damaged:.2f}%)")
        
        # Test different recovery methods
        methods = {
            'v3_full': (lazarus_restore_v3, 'Full (Consistency + Stability + Entropy)'),
            'baseline_noise': (baseline_noise_only, 'Baseline: Noise Only (no-op)'),
            'baseline_entropy': (baseline_entropy_only, 'Baseline: Entropy Only'),
            'baseline_consistency': (baseline_consistency_only, 'Baseline: Consistency Only'),
        }
        
        for method_name, (method_func, method_desc) in methods.items():
            net_restored = copy.deepcopy(net_damaged)
            
            if method_name == 'baseline_noise':
                method_func(net_restored, trainloader, RECOVERY_ITERS)
            elif method_name == 'baseline_entropy':
                method_func(net_restored, net_ref, trainloader, RECOVERY_ITERS, LR_RECOVERY, H0)
            elif method_name == 'baseline_consistency':
                method_func(net_restored, net_ref, trainloader, RECOVERY_ITERS, LR_RECOVERY)
            else:  # v3_full
                method_func(net_restored, net_ref, trainloader, RECOVERY_ITERS,
                           W_CONS, W_STAB, W_ENT, LR_RECOVERY, H0, EPSILON)
            
            acc_restored = evaluate(net_restored, testloader)
            entropy_restored = compute_entropy(net_restored, testloader)
            agreement_restored = compute_agreement(net_restored, net_ref, testloader)
            
            delta_acc = acc_restored - acc_damaged
            recovery_pct = ((acc_restored - acc_damaged) / (acc_original - acc_damaged) * 100) if (acc_original - acc_damaged) > 0 else 0
            
            results[alpha][method_name] = {
                'original': acc_original,
                'damaged': acc_damaged,
                'restored': acc_restored,
                'delta': delta_acc,
                'recovery_pct': recovery_pct,
                'entropy_damaged': entropy_damaged,
                'entropy_restored': entropy_restored,
                'agreement_damaged': agreement_damaged,
                'agreement_restored': agreement_restored,
                'agreement_improvement': agreement_damaged - agreement_restored,
            }
            
            print(f"  {method_desc}: {acc_restored:.2f}% (recovery: {delta_acc:+.2f}%, {recovery_pct:.1f}%)")
    
    # Output results
    print("\n" + "=" * 70)
    print("--- SUMMARY: Recovery Curve ---")
    print("=" * 70)
    
    print("\nFull Method (v3):")
    print("Alpha | Damaged | Restored | Delta  | Recovery % | Agreement Improvement")
    print("-" * 70)
    for alpha in ALPHA_VALUES:
        r = results[alpha]['v3_full']
        print(f"{alpha:5.2f} | {r['damaged']:7.2f}% | {r['restored']:8.2f}% | {r['delta']:+.2f}% | {r['recovery_pct']:10.1f}% | {r['agreement_improvement']:+.4f}")
    
    print("\nBaseline Comparison (alpha=0.2):")
    print("Method | Restored | Delta  | Recovery %")
    print("-" * 50)
    alpha_test = 0.2
    for method_name in ['baseline_noise', 'baseline_entropy', 'baseline_consistency', 'v3_full']:
        if method_name == 'v3_full':
            desc = 'Full (v3)'
        elif method_name == 'baseline_noise':
            desc = 'Noise Only'
        elif method_name == 'baseline_entropy':
            desc = 'Entropy Only'
        else:
            desc = 'Consistency Only'
        
        r = results[alpha_test][method_name]
        print(f"{desc:20} | {r['restored']:8.2f}% | {r['delta']:+.2f}% | {r['recovery_pct']:10.1f}%")
    
    # Save results
    with open('../../results/lazarus_analysis_results.json', 'w') as f:
        json.dump(dict(results), f, indent=2)
    print(f"\nResults saved to ../../results/lazarus_analysis_results.json")
    
    print("\n" + "=" * 70)
    print("--- Key Findings ---")
    print("=" * 70)
    
    # Analysis
    alpha_02 = results[0.2]['v3_full']
    print(f"\n1. Recovery Curve:")
    print(f"   - Alpha 0.1: Recovery {results[0.1]['v3_full']['recovery_pct']:.1f}%")
    print(f"   - Alpha 0.2: Recovery {results[0.2]['v3_full']['recovery_pct']:.1f}%")
    print(f"   - Alpha 0.3: Recovery {results[0.3]['v3_full']['recovery_pct']:.1f}%")
    
    print(f"\n2. Baseline Comparison (alpha=0.2):")
    print(f"   - Noise Only: {results[0.2]['baseline_noise']['delta']:+.2f}%")
    print(f"   - Entropy Only: {results[0.2]['baseline_entropy']['delta']:+.2f}%")
    print(f"   - Consistency Only: {results[0.2]['baseline_consistency']['delta']:+.2f}%")
    print(f"   - Full (v3): {results[0.2]['v3_full']['delta']:+.2f}%")
    
    print(f"\n3. Agreement Improvement (alpha=0.2):")
    print(f"   - Full method: {results[0.2]['v3_full']['agreement_improvement']:+.4f}")
    
    print("=" * 70)

if __name__ == "__main__":
    run_analysis()

