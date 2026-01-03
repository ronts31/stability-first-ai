# =========================
# Colab FULL: Active Sleep (Generative Replay) with Teacher labeling of dreams
# =========================
import copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# -------------------------
# Settings
# -------------------------
SEED = 42
FAST = False  # True = faster, weaker signal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detected device:", DEVICE)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed(SEED)

# -------------------------
# Data
# -------------------------
def get_mnist(root="./data"):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root, train=False, download=True, transform=tfm)
    return train_ds, test_ds

def indices_for_classes(dataset, classes):
    s = set(int(c) for c in classes)
    return [i for i,t in enumerate(dataset.targets) if int(t) in s]

def make_loader(dataset, indices, batch_size=128, shuffle=True, num_workers=0, pin_memory=False):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)

@torch.no_grad()
def eval_acc(model, loader, title):
    model.eval()
    correct=0; total=0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total += y.numel()
    acc = 100.0*correct/total
    print(f"[{title}] acc={acc:.2f}%")
    return acc

# -------------------------
# Model
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def backbone(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

    def forward(self, x):
        return self.fc3(self.backbone(x))

# -------------------------
# VAE (dream generator for A)
# -------------------------
class VAE(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = self.dec(z)
        return x.view(-1,1,28,28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

def vae_loss(x, xhat, mu, logvar):
    bce = nn.functional.binary_cross_entropy(xhat, x, reduction="sum")
    kl  = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl

def train_vae_on_A(vae, loaderA, epochs=10, lr=1e-3):
    vae.train()
    opt = optim.Adam(vae.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        s=0.0
        for x,_y in loaderA:
            x = x.to(DEVICE)
            # unnormalize to [0,1]
            x01 = x*0.3081 + 0.1307
            x01 = torch.clamp(x01, 0.0, 1.0)

            opt.zero_grad()
            xhat, mu, logvar = vae(x01)
            loss = vae_loss(x01, xhat, mu, logvar)
            loss.backward()
            opt.step()
            s += float(loss.detach().cpu())
        print(f"VAE epoch {ep}/{epochs} loss={s/len(loaderA):.1f}")

@torch.no_grad()
def sample_dreams_A(vae, n):
    vae.eval()
    z = torch.randn(n, vae.z_dim, device=DEVICE)
    x01 = vae.decode(z)
    # normalize back for classifier
    x_norm = (x01 - 0.1307) / 0.3081
    return x_norm

# -------------------------
# Core helpers
# -------------------------
def l2_delta(params, ref_params):
    s=0.0
    for p,p0 in zip(params, ref_params):
        s = s + (p-p0).pow(2).sum()
    return s

def train_plain(model, loader, epochs=3, lr=1e-3, title=""):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    print(f"\n>>> Train (plain): {title}")
    for ep in range(1, epochs+1):
        s=0.0
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            s += float(loss.detach().cpu())
        print(f"epoch {ep}/{epochs} loss={s/len(loader):.4f}")

def train_B_head_only(model, loaderB, epochs=5, lr=1e-3):
    # Freeze backbone
    for p in model.fc1.parameters(): p.requires_grad = False
    for p in model.fc2.parameters(): p.requires_grad = False
    for p in model.fc3.parameters(): p.requires_grad = True

    model.train()
    opt = optim.Adam(model.fc3.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    print("\n>>> PHASE 2: Train B in HEAD ONLY (backbone frozen)")
    for ep in range(1, epochs+1):
        s=0.0
        for x,y in loaderB:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            s += float(loss.detach().cpu())
        print(f"epoch {ep}/{epochs} loss={s/len(loaderB):.4f}")

@torch.no_grad()
def collect_features(model, dataset, indices, max_n=4000):
    model.eval()
    if len(indices) > max_n:
        indices = random.sample(indices, max_n)
    feats=[]; labels=[]
    for i in indices:
        x,y = dataset[i]
        x = x.unsqueeze(0).to(DEVICE)
        h = model.backbone(x).squeeze(0).detach().cpu()
        feats.append(h); labels.append(int(y))
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)

def train_linear_probe(Xtr, Ytr, Xte, Yte, epochs=30, lr=1e-2):
    probe = nn.Linear(Xtr.size(1), 10).to(DEVICE)
    opt = optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr.to(DEVICE), Ytr.to(DEVICE)), batch_size=256, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb,yb in loader:
            opt.zero_grad()
            loss = ce(probe(xb), yb)
            loss.backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(Xte.to(DEVICE)).argmax(1).cpu()
    return 100.0*(pred==Yte).float().mean().item()

# -------------------------
# ACTIVE SLEEP: full model updates + (real B) + (dream A labeled by teacherA) + optional B-distill
# -------------------------
def active_sleep(
    model,
    teacherA,
    teacherB,
    vaeA,
    loaderB_real,
    anchor_ref_A=None,
    sleep_epochs=12,
    lr=1e-3,
    dream_fraction=0.75,
    lambda_anchor=50.0,
    use_B_distill=True,
    T=2.0,
    batch_cap=128,
):
    print("\n>>> PHASE 3: ACTIVE SLEEP (unfrozen head + generative replay)")
    print(f"    sleep_epochs={sleep_epochs} lr={lr} dream_fraction={dream_fraction} lambda_anchor={lambda_anchor} "
          f"use_B_distill={use_B_distill} T={T}")

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    model.train()
    teacherA.eval()
    for p in teacherA.parameters():
        p.requires_grad = False
    if teacherB is not None:
        teacherB.eval()
        for p in teacherB.parameters():
            p.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    # Backbone anchor to snapshot after A
    if anchor_ref_A is not None:
        anchor_ref_A.eval()
        for p in anchor_ref_A.parameters():
            p.requires_grad = False
        back_p   = list(model.fc1.parameters()) + list(model.fc2.parameters())
        back_ref = list(anchor_ref_A.fc1.parameters()) + list(anchor_ref_A.fc2.parameters())
    else:
        back_p, back_ref = None, None

    for ep in range(1, sleep_epochs+1):
        s_tot=s_Bhard=s_Adist=s_Bdist=s_reg=0.0
        n_batches=0

        for xB, yB_true in loaderB_real:
            xB, yB_true = xB.to(DEVICE), yB_true.to(DEVICE)
            if xB.size(0) > batch_cap:
                xB = xB[:batch_cap]
                yB_true = yB_true[:batch_cap]

            # dreams from A
            nA = max(1, int(xB.size(0)*dream_fraction))
            xA = sample_dreams_A(vaeA, nA)

            with torch.no_grad():
                logitsA_t = teacherA(xA)
                if teacherB is not None:
                    logitsB_t = teacherB(xB)

            opt.zero_grad()

            # 1) Real B: hard labels (keeps plasticity)
            logitsB = model(xB)
            loss_B_hard = ce(logitsB, yB_true)

            # 2) Dream A: distill from teacherA (restores A access)
            logitsA = model(xA)
            log_pA = nn.functional.log_softmax(logitsA / T, dim=1)
            qA     = nn.functional.softmax(logitsA_t / T, dim=1)
            loss_A_dist = (T*T) * kl(log_pA, qA)

            # 3) Optional: distill B from teacherB (stabilize B mapping)
            loss_B_dist = 0.0
            if use_B_distill and teacherB is not None:
                log_pB = nn.functional.log_softmax(logitsB / T, dim=1)
                qB     = nn.functional.softmax(logitsB_t / T, dim=1)
                loss_B_dist = (T*T) * kl(log_pB, qB)

            # 4) Soft backbone anchor to A-identity
            reg = 0.0
            if back_p is not None and lambda_anchor > 0:
                reg = lambda_anchor * l2_delta(back_p, back_ref)

            loss = loss_B_hard + loss_A_dist + loss_B_dist + reg
            loss.backward()
            opt.step()

            s_tot     += float(loss.detach().cpu())
            s_Bhard   += float(loss_B_hard.detach().cpu())
            s_Adist   += float(loss_A_dist.detach().cpu())
            s_Bdist   += float(loss_B_dist.detach().cpu()) if isinstance(loss_B_dist, torch.Tensor) else float(loss_B_dist)
            s_reg     += float(reg.detach().cpu()) if isinstance(reg, torch.Tensor) else float(reg)
            n_batches += 1

        print(f"epoch {ep}/{sleep_epochs} total={s_tot/n_batches:.4f} "
              f"B_hard={s_Bhard/n_batches:.4f} A_dist={s_Adist/n_batches:.4f} "
              f"B_dist={s_Bdist/n_batches:.4f} reg={s_reg/n_batches:.4f}")

# =========================
# Main execution
# =========================
if __name__ == '__main__':
    # =========================
    # Build loaders
    # =========================
    A_classes = [0,1,2,3,4]
    B_classes = [5,6,7,8,9]

    train_ds, test_ds = get_mnist("./data")
idxA_tr = indices_for_classes(train_ds, A_classes)
idxB_tr = indices_for_classes(train_ds, B_classes)
idxA_te = indices_for_classes(test_ds, A_classes)
idxB_te = indices_for_classes(test_ds, B_classes)

bsA = 128
bsB = 128
if FAST:
    bsA = 256
    bsB = 256

trainA = make_loader(train_ds, idxA_tr, batch_size=bsA, shuffle=True, num_workers=0, pin_memory=False)
trainB = make_loader(train_ds, idxB_tr, batch_size=bsB, shuffle=True, num_workers=0, pin_memory=False)
testA  = make_loader(test_ds,  idxA_te, batch_size=1000, shuffle=False, num_workers=0, pin_memory=False)
testB  = make_loader(test_ds,  idxB_te, batch_size=1000, shuffle=False, num_workers=0, pin_memory=False)

# =========================
# PHASE 1: Train A
# =========================
model = SimpleMLP().to(DEVICE)

epochsA = 3 if not FAST else 2
epochsB = 5 if not FAST else 3

train_plain(model, trainA, epochs=epochsA, lr=1e-3, title="Task A (0-4)")
eval_acc(model, testA, "A after A")
eval_acc(model, testB, "B after A (~0)")

teacherA = copy.deepcopy(model).to(DEVICE)
for p in teacherA.parameters(): p.requires_grad = False

anchor_ref_A = copy.deepcopy(model).to(DEVICE)
for p in anchor_ref_A.parameters(): p.requires_grad = False

# =========================
# PHASE 2: Train B head-only
# =========================
train_B_head_only(model, trainB, epochs=epochsB, lr=1e-3)
eval_acc(model, testA, "A after B (head-only)")
eval_acc(model, testB, "B after B (head-only)")

teacherB = copy.deepcopy(model).to(DEVICE)
for p in teacherB.parameters(): p.requires_grad = False

# =========================
# Backbone-only probe BEFORE sleep (on A∪B test)
# =========================
# Build AB feature sets for probe
idxAB_tr = idxA_tr[:6000] + idxB_tr[:6000]
idxAB_te = idxA_te + idxB_te

Xtr, Ytr = collect_features(model, train_ds, idxAB_tr, max_n=8000)
Xte, Yte = collect_features(model, test_ds, idxAB_te,  max_n=4000)

probe_epochs = 30 if not FAST else 15
probe_acc_before = train_linear_probe(Xtr, Ytr, Xte, Yte, epochs=probe_epochs, lr=1e-2)
print(f"\n[BACKBONE-ONLY] Fresh linear probe BEFORE sleep: {probe_acc_before:.2f}%")

# =========================
# Train VAE on A
# =========================
vae_epochs = 10 if not FAST else 4
vae = VAE(z_dim=32 if not FAST else 16).to(DEVICE)
print("\n>>> Train VAE_A on Task A distribution (dream generator)")
train_vae_on_A(vae, trainA, epochs=vae_epochs, lr=1e-3)

# =========================
# PHASE 3: ACTIVE SLEEP
# =========================
sleep_epochs = 12 if not FAST else 6
active_sleep(
    model=model,
    teacherA=teacherA,
    teacherB=teacherB,
    vaeA=vae,
    loaderB_real=trainB,
    anchor_ref_A=anchor_ref_A,
    sleep_epochs=sleep_epochs,
    lr=1e-3,
    dream_fraction=0.75,     # try 0.5..1.0
    lambda_anchor=50.0,      # try 20..150
    use_B_distill=True,
    T=2.0,
    batch_cap=128,
)

# =========================
# Evaluate after sleep
# =========================
eval_acc(model, testA, "A after ACTIVE SLEEP")
eval_acc(model, testB, "B after ACTIVE SLEEP")

# Backbone-only probe AFTER sleep
Xtr2, Ytr2 = collect_features(model, train_ds, idxAB_tr, max_n=8000)
Xte2, Yte2 = collect_features(model, test_ds,  idxAB_te, max_n=4000)
probe_acc_after = train_linear_probe(Xtr2, Ytr2, Xte2, Yte2, epochs=probe_epochs, lr=1e-2)
print(f"[BACKBONE-ONLY] Fresh linear probe AFTER sleep:  {probe_acc_after:.2f}%")

print("\n" + "="*86)
print("SUMMARY: Active Sleep (Generative Replay + Teacher-labeled dreams)")
print("="*86)
print(f"Backbone-only probe: before={probe_acc_before:.2f}% | after={probe_acc_after:.2f}%")
print("If A doesn't recover: increase sleep_epochs and/or dream_fraction, reduce lambda_anchor slightly.")
print("If B collapses: decrease dream_fraction and/or increase lambda_anchor, keep B_distill=True.")
print("="*86)

