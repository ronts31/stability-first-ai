"""
Experiment 6: Subjective Time (The Critic & Surprise)
Hypothesis: The system should autonomously regulate its plasticity (Lambda) based on 'Surprise'.
Mechanism: A 'Critic' network predicts the loss. Large prediction error = High Surprise = Low Lambda (High Plasticity).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --- The Main System ---
class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Backbone (Memory)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        # Head (Interface)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def get_features(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.relu(self.fc2(x))

    def forward(self, x):
        h = self.get_features(x)
        return self.fc3(h)

# --- The Critic (Metacognition) ---
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Features from MainModel (128 dim)
        # Output: Predicted Loss (Scalar)
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predicts scalar Loss
        )
    
    def forward(self, features):
        return self.net(features)

# --- Data ---
def get_loaders():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.13,), (0.31,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    
    # Task A: 0-4 (Routine past)
    idx_a = [i for i, t in enumerate(train_ds.targets) if t < 5]
    # Task B: 5-9 (Shocking new future)
    idx_b = [i for i, t in enumerate(train_ds.targets) if t >= 5]
    
    # Windows compatibility: num_workers=0, pin_memory=False
    train_a = DataLoader(Subset(train_ds, idx_a), batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
    train_b = DataLoader(Subset(train_ds, idx_b), batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
    test_all = DataLoader(test_ds, batch_size=1000, num_workers=0, pin_memory=False)
    
    return train_a, train_b, test_all

# --- Training with Subjective Time ---
def train_subjective(model, critic, loader, ref_model, optimizer, critic_opt, task_name, base_lambda=5000.0):
    loss_fn = nn.CrossEntropyLoss(reduction='none')  # Need per-sample loss for analysis
    
    avg_loss = 0
    avg_surprise = 0
    avg_lambda = 0
    
    model.train()
    critic.train()
    
    history_lambda = []
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # 1. Main Model Forward
        features = model.get_features(x)
        logits = model.fc3(features)
        
        # Calculate ACTUAL Loss (per sample)
        real_loss = loss_fn(logits, y)  # [batch]
        mean_task_loss = real_loss.mean()
        
        # 2. Critic Forward (Predict Loss)
        # Critic tries to guess 'real_loss' based on 'features'
        predicted_loss = critic(features.detach()).squeeze()  # [batch]
        
        # 3. Calculate SURPRISE
        # Surprise = |Real - Predicted|
        # We detach real_loss because we don't want to optimize the Main Model to satisfy the Critic
        surprise = torch.abs(real_loss.detach() - predicted_loss).mean()
        
        # 4. Determine SUBJECTIVE TIME (Lambda)
        # High Surprise -> Low Lambda (Time Slows Down / High Plasticity)
        # Low Surprise -> High Lambda (Time Speeds Up / High Stability)
        sensitivity = 10.0
        current_lambda = base_lambda / (1.0 + surprise.item() * sensitivity)
        
        # 5. Stability Loss (Backbone Anchor)
        reg_loss = 0.0
        if ref_model is not None:
            # L2 distance on backbone only
            for p, p_ref in zip(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                                list(ref_model.fc1.parameters()) + list(ref_model.fc2.parameters())):
                reg_loss += (p - p_ref).pow(2).sum()
        
        total_loss = mean_task_loss + (current_lambda * reg_loss)
        
        # 6. Optimization Step (Main Model)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 7. Optimization Step (Critic)
        # Critic learns to predict the loss better
        critic_loss = nn.MSELoss()(predicted_loss, real_loss.detach())
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        
        # Stats
        avg_loss += mean_task_loss.item()
        avg_surprise += surprise.item()
        avg_lambda += current_lambda
        history_lambda.append(current_lambda)
        
    n = len(loader)
    print(f"[{task_name}] Loss: {avg_loss/n:.4f} | Surprise: {avg_surprise/n:.4f} | Avg Lambda: {avg_lambda/n:.1f}")
    return history_lambda

def eval_acc(model, loader, title):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print(f"[{title}] acc={acc:.2f}%")
    return acc

def main():
    print("="*80)
    print("EXPERIMENT 6: SUBJECTIVE TIME (THE CRITIC)")
    print("="*80)
    set_seed(SEED)
    train_a, train_b, test_all = get_loaders()
    
    model = MainModel().to(DEVICE)
    critic = Critic().to(DEVICE)
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    
    # 1. Childhood (Train A)
    # Critic learns that A is "Normal"
    print("\n" + "="*80)
    print("Phase 1: Childhood (Task A 0-4)")
    print("="*80)
    for ep in range(3):
        train_subjective(model, critic, train_a, None, opt, critic_opt, f"Epoch {ep+1}", base_lambda=0.0)
    
    eval_acc(model, test_all, "After Task A")
    
    # Freeze Time (Snapshot)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # 2. Adulthood (Train B with Subjective Time)
    # The Critic expects A-like data (Low Loss). 
    # It sees B-like data (High Loss) -> High Surprise -> Low Lambda -> Fast Learning.
    # As it learns B, Surprise drops -> Lambda increases -> Stability returns.
    print("\n" + "="*80)
    print("Phase 2: Adulthood (Task B 5-9) - Adaptive Plasticity")
    print("="*80)
    
    all_lambdas = []
    for ep in range(5):
        # We start with High Base Lambda (Conservative), but Surprise will lower it dynamically
        lambdas = train_subjective(model, critic, train_b, ref_model, opt, critic_opt, f"Epoch {ep+1}", base_lambda=10000.0)
        all_lambdas.extend(lambdas)
    
    # Evaluate
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    eval_acc(model, test_all, "Final Accuracy (A+B)")
    
    # Plotting Subjective Time
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(all_lambdas, linewidth=1.5, alpha=0.7)
        plt.title("Subjective Time: Elasticity of Weights (Lambda)", fontsize=14, fontweight="bold")
        plt.xlabel("Training Steps (Experience)", fontsize=12)
        plt.ylabel("Lambda (Stiffness / Stability)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("subjective_time.png", dpi=150)
        print("\n[OK] Graph saved: subjective_time.png")
        print("Interpretation: Drops in Lambda indicate 'Moments of Surprise' where the model opened itself to change.")
    except Exception as e:
        print(f"\n[WARN] Could not create plot: {e}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()






