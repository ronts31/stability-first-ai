"""
Pruning curve: testing Lazarus v3 at different pruning levels
Mode comparison: Frozen Mask vs Regrow Allowed
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
PRUNING_AMOUNTS = [0.3, 0.5, 0.7, 0.8]  # Tested pruning levels
RECOVERY_ITERS = 1000

# Recovery parameters
W_CONS = 1.0
W_STAB = 0.5
W_ENT = 0.05
LR_RECOVERY = 1e-4
H0 = 1.5
EPSILON = 0.05

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

def prune_weights(model, amount=0.3):
    """L1 Unstructured Pruning with mask return"""
    all_weights = []
    for param in model.parameters():
        if param.dim() > 1:
            all_weights.append(param.view(-1))
    
    if len(all_weights) == 0:
        return None, 0.0
    
    all_weights = torch.cat(all_weights)
    num_params = all_weights.numel()
    threshold_idx = int(num_params * amount)
    sorted_weights, _ = torch.sort(torch.abs(all_weights))
    threshold = sorted_weights[threshold_idx].item()
    
    # Create mask for all parameters
    masks = {}
    pruned_count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() > 1:
                mask = torch.abs(param) >= threshold
                masks[name] = mask.clone()
                pruned_count += (~mask).sum().item()
                param.data.mul_(mask.float())
    
    return masks, threshold

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

# 3. RECOVERY METHODS

def lazarus_restore_v3_frozen_mask(model, ref_model, unlabeled_loader, masks,
                                   iterations=1000, w_cons=1.0, w_stab=0.5, w_ent=0.05,
                                   lr=1e-4, H0=1.5, epsilon=0.05):
    """
    Lazarus v3 with frozen mask (Frozen Mask mode)
    Zeroed weights cannot be restored
    """
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
        
        # Apply mask to gradients (freeze zeroed weights)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks and param.grad is not None:
                    param.grad.mul_(masks[name].float())
        
        opt.step()
        it += 1

def lazarus_restore_v3_regrow(model, ref_model, unlabeled_loader,
                              iterations=1000, w_cons=1.0, w_stab=0.5, w_ent=0.05,
                              lr=1e-4, H0=1.5, epsilon=0.05):
    """
    Lazarus v3 with weight restoration allowed (Regrow mode)
    Zeroed weights can be restored
    """
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

# 4. MAIN FUNCTION
def run_pruning_curve_analysis():
    print("=" * 70)
    print("Pruning Curve Analysis: Lazarus v3")
    print("Testing levels:", PRUNING_AMOUNTS)
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
    
    # Test different pruning levels
    print("--- Testing Different Pruning Levels ---")
    for pruning_amount in PRUNING_AMOUNTS:
        print(f"\n[Pruning = {pruning_amount*100:.0f}%]")
        
        # Damage
        net_pruned = copy.deepcopy(net)
        masks, threshold = prune_weights(net_pruned, amount=pruning_amount)
        acc_pruned = evaluate(net_pruned, testloader)
        agreement_pruned = compute_agreement(net_pruned, net_ref, testloader)
        nonzero_pruned, total = count_nonzero_weights(net_pruned)
        
        print(f"  Pruned: {acc_pruned:.2f}% (drop: {acc_original - acc_pruned:.2f}%)")
        print(f"  Nonzero weights: {nonzero_pruned:,}/{total:,} ({100*nonzero_pruned/total:.1f}%)")
        
        # Test two modes
        modes = {
            'frozen_mask': ('Frozen Mask', lazarus_restore_v3_frozen_mask),
            'regrow': ('Regrow Allowed', lazarus_restore_v3_regrow),
        }
        
        for mode_name, (mode_desc, method_func) in modes.items():
            print(f"  Testing {mode_desc}...")
            net_restored = copy.deepcopy(net_pruned)
            
            if mode_name == 'frozen_mask':
                method_func(net_restored, net_ref, trainloader, masks,
                           RECOVERY_ITERS, W_CONS, W_STAB, W_ENT,
                           LR_RECOVERY, H0, EPSILON)
            else:
                method_func(net_restored, net_ref, trainloader,
                           RECOVERY_ITERS, W_CONS, W_STAB, W_ENT,
                           LR_RECOVERY, H0, EPSILON)
            
            acc_restored = evaluate(net_restored, testloader)
            agreement_restored = compute_agreement(net_restored, net_ref, testloader)
            nonzero_restored, _ = count_nonzero_weights(net_restored)
            
            delta_acc = acc_restored - acc_pruned
            recovery_pct = ((acc_restored - acc_pruned) / (acc_original - acc_pruned) * 100) if (acc_original - acc_pruned) > 0 else 0
            agreement_improvement = agreement_pruned - agreement_restored
            regrowth_pct = ((nonzero_restored - nonzero_pruned) / (total - nonzero_pruned) * 100) if (total - nonzero_pruned) > 0 else 0
            
            results[pruning_amount][mode_name] = {
                'original': acc_original,
                'pruned': acc_pruned,
                'restored': acc_restored,
                'delta': delta_acc,
                'recovery_pct': recovery_pct,
                'agreement_pruned': agreement_pruned,
                'agreement_restored': agreement_restored,
                'agreement_improvement': agreement_improvement,
                'nonzero_pruned': nonzero_pruned,
                'nonzero_restored': nonzero_restored,
                'regrowth_pct': regrowth_pct,
            }
            
            print(f"    {mode_desc}: {acc_restored:.2f}% (recovery: {delta_acc:+.2f}%, {recovery_pct:.1f}%)")
            print(f"      Agreement improvement: {agreement_improvement:+.4f}")
            print(f"      Nonzero: {nonzero_restored:,} ({100*nonzero_restored/total:.1f}%, regrowth: {regrowth_pct:.1f}%)")
    
    # Output results
    print("\n" + "=" * 70)
    print("--- PRUNING CURVE SUMMARY ---")
    print("=" * 70)
    
    print("\nFrozen Mask Mode:")
    print("Pruning | Pruned | Restored | Delta  | Recovery % | Agreement | Nonzero %")
    print("-" * 80)
    for pruning_amount in PRUNING_AMOUNTS:
        r = results[pruning_amount]['frozen_mask']
        print(f"{pruning_amount*100:6.0f}% | {r['pruned']:6.2f}% | {r['restored']:8.2f}% | {r['delta']:+.2f}% | {r['recovery_pct']:9.1f}% | {r['agreement_improvement']:+.4f} | {100*r['nonzero_restored']/(r['nonzero_restored']+r['nonzero_pruned']):.1f}%")
    
    print("\nRegrow Allowed Mode:")
    print("Pruning | Pruned | Restored | Delta  | Recovery % | Agreement | Regrowth %")
    print("-" * 80)
    for pruning_amount in PRUNING_AMOUNTS:
        r = results[pruning_amount]['regrow']
        print(f"{pruning_amount*100:6.0f}% | {r['pruned']:6.2f}% | {r['restored']:8.2f}% | {r['delta']:+.2f}% | {r['recovery_pct']:9.1f}% | {r['agreement_improvement']:+.4f} | {r['regrowth_pct']:9.1f}%")
    
    print("\n" + "=" * 70)
    print("--- COMPARISON: Frozen Mask vs Regrow (70% pruning) ---")
    print("=" * 70)
    
    pruning_test = 0.7
    if pruning_test in results:
        frozen = results[pruning_test]['frozen_mask']
        regrow = results[pruning_test]['regrow']
        
        print(f"\nFrozen Mask Mode:")
        print(f"  Accuracy recovery: {frozen['delta']:+.2f}% ({frozen['recovery_pct']:.1f}%)")
        print(f"  Agreement improvement: {frozen['agreement_improvement']:+.4f}")
        print(f"  Nonzero weights: {frozen['nonzero_restored']:,} ({100*frozen['nonzero_restored']/(frozen['nonzero_restored']+frozen['nonzero_pruned']):.1f}%)")
        
        print(f"\nRegrow Allowed Mode:")
        print(f"  Accuracy recovery: {regrow['delta']:+.2f}% ({regrow['recovery_pct']:.1f}%)")
        print(f"  Agreement improvement: {regrow['agreement_improvement']:+.4f}")
        print(f"  Nonzero weights: {regrow['nonzero_restored']:,} ({100*regrow['nonzero_restored']/(regrow['nonzero_restored']+regrow['nonzero_pruned']):.1f}%)")
        print(f"  Regrowth: {regrow['regrowth_pct']:.1f}% of pruned weights")
        
        print(f"\nComparison:")
        print(f"  Accuracy: Regrow {'better' if regrow['delta'] > frozen['delta'] else 'worse'} by {abs(regrow['delta'] - frozen['delta']):.2f}%")
        print(f"  Agreement: Regrow {'better' if regrow['agreement_improvement'] > frozen['agreement_improvement'] else 'worse'} by {abs(regrow['agreement_improvement'] - frozen['agreement_improvement']):.4f}")
    
    # Save results
    with open('../../results/pruning_curve_results.json', 'w') as f:
        json.dump(dict(results), f, indent=2)
    print(f"\nResults saved to ../../results/pruning_curve_results.json")
    print("=" * 70)

if __name__ == "__main__":
    run_pruning_curve_analysis()

