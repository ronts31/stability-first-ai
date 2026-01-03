"""
Statistical significance of Lazarus v3: 5-10 seeds for alpha=0.2 and 0.3
Computes mean ± std for all metrics
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy
import numpy as np
import json
import os
from collections import defaultdict

# --- CONFIGURATION ---
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 5
RECOVERY_ITERS = 1000
NUM_SEEDS = 5  # Number of seeds for statistics

# Recovery parameters
W_CONS = 1.0
W_STAB = 0.5
W_ENT = 0.05
LR_RECOVERY = 1e-4
H0 = 1.5
EPSILON = 0.05

# Tested damage levels
ALPHA_VALUES = [0.2, 0.3]

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

# 4. RECOVERY METHOD
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

# 5. MAIN FUNCTION
def run_statistical_analysis():
    print("=" * 70)
    print("Statistical Significance Analysis: Lazarus v3")
    print(f"Running {NUM_SEEDS} seeds for alpha={ALPHA_VALUES}")
    print("=" * 70)
    print()
    
    # Storage for results from all seeds
    all_results = defaultdict(lambda: defaultdict(list))
    
    for seed in range(NUM_SEEDS):
        print(f"\n--- Seed {seed + 1}/{NUM_SEEDS} ---")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Train base model
        print("Training base model...")
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
        
        acc_original = evaluate(net, testloader)
        print(f"Original Accuracy: {acc_original:.2f}%")
        
        net_ref = copy.deepcopy(net)
        net_ref.eval()
        
        # Test different alpha
        for alpha in ALPHA_VALUES:
            print(f"  Testing alpha={alpha}...")
            
            # Damage
            net_damaged = copy.deepcopy(net)
            damage_weights_relative(net_damaged, alpha=alpha)
            acc_damaged = evaluate(net_damaged, testloader)
            agreement_damaged = compute_agreement(net_damaged, net_ref, testloader)
            
            # Recovery
            net_restored = copy.deepcopy(net_damaged)
            lazarus_restore_v3(net_restored, net_ref, trainloader, RECOVERY_ITERS,
                             W_CONS, W_STAB, W_ENT, LR_RECOVERY, H0, EPSILON)
            
            acc_restored = evaluate(net_restored, testloader)
            agreement_restored = compute_agreement(net_restored, net_ref, testloader)
            
            # Compute metrics
            delta_acc = acc_restored - acc_damaged
            recovery_pct = ((acc_restored - acc_damaged) / (acc_original - acc_damaged) * 100) if (acc_original - acc_damaged) > 0 else 0
            agreement_improvement = agreement_damaged - agreement_restored
            
            # Save results
            all_results[alpha]['original'].append(acc_original)
            all_results[alpha]['damaged'].append(acc_damaged)
            all_results[alpha]['restored'].append(acc_restored)
            all_results[alpha]['delta'].append(delta_acc)
            all_results[alpha]['recovery_pct'].append(recovery_pct)
            all_results[alpha]['agreement_damaged'].append(agreement_damaged)
            all_results[alpha]['agreement_restored'].append(agreement_restored)
            all_results[alpha]['agreement_improvement'].append(agreement_improvement)
            
            print(f"    Damaged: {acc_damaged:.2f}%, Restored: {acc_restored:.2f}%, Delta: {delta_acc:+.2f}%")
    
    # Statistical analysis
    print("\n" + "=" * 70)
    print("--- STATISTICAL SUMMARY (Mean +/- Std) ---")
    print("=" * 70)
    
    summary = {}
    
    for alpha in ALPHA_VALUES:
        print(f"\nAlpha = {alpha}:")
        print("-" * 70)
        
        metrics = {
            'original': all_results[alpha]['original'],
            'damaged': all_results[alpha]['damaged'],
            'restored': all_results[alpha]['restored'],
            'delta': all_results[alpha]['delta'],
            'recovery_pct': all_results[alpha]['recovery_pct'],
            'agreement_improvement': all_results[alpha]['agreement_improvement'],
        }
        
        summary[alpha] = {}
        
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[alpha][metric_name] = {'mean': mean_val, 'std': std_val}
            
            if metric_name == 'delta':
                print(f"  Delta accuracy:      {mean_val:+.2f}% +/- {std_val:.2f}%")
            elif metric_name == 'recovery_pct':
                print(f"  Recovery rate:      {mean_val:.1f}% ± {std_val:.1f}%")
            elif metric_name == 'agreement_improvement':
                print(f"  Agreement improvement: {mean_val:+.4f} ± {std_val:.4f}")
            elif metric_name == 'original':
                print(f"  Original accuracy:  {mean_val:.2f}% ± {std_val:.2f}%")
            elif metric_name == 'damaged':
                print(f"  Damaged accuracy:   {mean_val:.2f}% ± {std_val:.2f}%")
            elif metric_name == 'restored':
                print(f"  Restored accuracy:  {mean_val:.2f}% ± {std_val:.2f}%")
    
    # Save results
    with open('../../results/lazarus_statistical_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("--- KEY METRICS FOR PUBLICATION ---")
    print("=" * 70)
    
    for alpha in ALPHA_VALUES:
        delta_mean = summary[alpha]['delta']['mean']
        delta_std = summary[alpha]['delta']['std']
        recovery_mean = summary[alpha]['recovery_pct']['mean']
        recovery_std = summary[alpha]['recovery_pct']['std']
        agreement_mean = summary[alpha]['agreement_improvement']['mean']
        agreement_std = summary[alpha]['agreement_improvement']['std']
        
        print(f"\nAlpha = {alpha}:")
        print(f"  Delta accuracy:      {delta_mean:+.2f}% +/- {delta_std:.2f}%")
        print(f"  Recovery rate:       {recovery_mean:.1f}% +/- {recovery_std:.1f}%")
        print(f"  Agreement improvement: {agreement_mean:+.4f} +/- {agreement_std:.4f}")
    
    print("\nResults saved to ../../results/lazarus_statistical_results.json")
    print("=" * 70)

if __name__ == "__main__":
    run_statistical_analysis()

