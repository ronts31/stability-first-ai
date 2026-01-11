# test_recursive_cifar10_b200.py
# Ready-to-run version tuned for NVIDIA B200 (BF16 + TF32), with fixes:
# - Correct per-sample loss for SubjectiveTimeCritic
# - Consistent freeze/optimizer behavior (train head + late backbone with small LR)
# - Normalized stability loss
# - CLIP KL gated by CLIP confidence + lower max KL weight
# - Fixed Unknown counting in final test
# - Cleaner replay/pain gradient logic (single replay sample per step)

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[WARNING] CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


# ----------------------------
# B200 runtime knobs
# ----------------------------
def setup_b200():
    assert torch.cuda.is_available(), "CUDA is required for B200"
    torch.backends.cudnn.benchmark = True
    # TF32 helps speed for conv/matmul on recent GPUs while keeping good accuracy
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Better matmul kernels selection
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ----------------------------
# Complexity sensor
# ----------------------------
class ComplexitySensor:
    def __init__(self, sensitivity=2.5):
        self.history = []
        self.mean = 0
        self.std = 1
        self.sensitivity = sensitivity
        self.calibrated = False

    def update(self, loss):
        self.history.append(loss)
        if len(self.history) > 500:
            self.history.pop(0)

    def calibrate(self):
        if len(self.history) > 10:
            self.mean = float(np.mean(self.history))
            self.std = float(np.std(self.history) + 1e-6)
            self.calibrated = True
            print(f"[SENSOR] Baseline set. Mean={self.mean:.3f}, Std={self.std:.3f}")

    def is_shock(self, loss):
        if not self.calibrated:
            return False
        z_score = (loss - self.mean) / self.std
        return z_score > self.sensitivity


# ----------------------------
# Subjective Time Critic
# ----------------------------
class SubjectiveTimeCritic(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)  # [B]

    @staticmethod
    def surprise(predicted_loss_ps, real_loss_ps):
        # both are [B]
        return (real_loss_ps.detach() - predicted_loss_ps).abs().mean()

    @staticmethod
    def compute_lambda(surprise_value, base_lambda=10000.0, sensitivity=10.0):
        # surprise_value: scalar tensor
        s = float(surprise_value.item())
        return base_lambda / (1.0 + s * sensitivity)


# ----------------------------
# Dream VAE
# ----------------------------
class DreamVAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16->8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8->4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
        )
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 16->32
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl


# ----------------------------
# Shared backbone + expandable head
# ----------------------------
class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.hidden_size = 512

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)  # 32->16
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2)  # 16->8
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pool2d(h, 2)  # 8->4
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.gap(h)  # 4->1
        return h.view(h.size(0), -1)  # [B,512]


class ExpandableHead(nn.Module):
    def __init__(self, hidden_size, output_size, prev_dims=None):
        super().__init__()
        prev_dims = prev_dims or []
        self.adapters = nn.ModuleList([nn.Linear(p, hidden_size) for p in prev_dims])
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, backbone_features, prev_hiddens):
        h = backbone_features
        for i, adapter in enumerate(self.adapters):
            if i < len(prev_hiddens):
                h = h + adapter(prev_hiddens[i])
        return self.fc(h), h


# Compatibility column for sleep compression
class TemporalColumn(nn.Module):
    def __init__(self, hidden_size, output_size, prev_dims=None):
        super().__init__()
        self.backbone = SharedBackbone()
        self.head = ExpandableHead(hidden_size, output_size, prev_dims or [])
        self.hidden_size = hidden_size

    def forward(self, x, prev_hiddens):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        feats = self.backbone(x)
        return self.head(feats, prev_hiddens)


# ----------------------------
# Curiosity module (CLIP)
# ----------------------------
class CuriosityModule:
    def __init__(self):
        if not CLIP_AVAILABLE:
            self.available = False
            return

        self.available = True
        self.device = "cuda"
        print("[CURIOSITY] Loading World Knowledge (CLIP)...")
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        self.cifar10_concepts = [
            "a photo of an airplane", "a photo of a car", "a photo of a bird",
            "a photo of a cat", "a photo of a deer", "a photo of a dog",
            "a photo of a frog", "a photo of a horse", "a photo of a ship",
            "a photo of a truck",
        ]
        self.text_inputs = clip.tokenize(self.cifar10_concepts).to(self.device)

        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

        print("[CURIOSITY] CLIP loaded successfully!")

    def _prep(self, x):
        x = x * 0.5 + 0.5
        x = torch.clamp(x, 0, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        return x

    def what_is_this(self, image_tensor, return_probs=False):
        if not self.available:
            return (None, None, 0.0) if not return_probs else (None, None, 0.0, None)

        try:
            image_in = self._prep(image_tensor.to(self.device))
            with torch.no_grad():
                logits_per_image, _ = self.model(image_in, self.text_inputs)
                probs = logits_per_image.softmax(dim=-1).squeeze(0)  # [10]
            best_idx = int(torch.argmax(probs).item())
            best_label = self.cifar10_concepts[best_idx]
            conf = float(probs[best_idx].item())
            if return_probs:
                return best_idx, best_label, conf, probs.detach()
            return best_idx, best_label, conf
        except Exception as e:
            print(f"[CURIOSITY] Error: {e}")
            return (None, None, 0.0) if not return_probs else (None, None, 0.0, None)


# ----------------------------
# Recursive agent
# ----------------------------
class RecursiveAgent(nn.Module):
    def __init__(self, use_curiosity=False, use_subjective_time=False, use_vae_dreams=False):
        super().__init__()
        self.hidden_size = 512
        self.output_size = 11
        self.unknown_class_idx = 10

        self.shared_backbone = SharedBackbone()
        self.heads = nn.ModuleList([ExpandableHead(self.hidden_size, self.output_size)])
        self.columns = nn.ModuleList([TemporalColumn(self.hidden_size, self.output_size)])

        self.sensor = ComplexitySensor()
        self.active_classes_per_column = {}

        self.use_curiosity = bool(use_curiosity and CLIP_AVAILABLE)
        if self.use_curiosity:
            self.curiosity = CuriosityModule()

        self.use_subjective_time = bool(use_subjective_time)
        if self.use_subjective_time:
            self.critic = SubjectiveTimeCritic(feature_dim=self.hidden_size)
            self.ref_backbone = None

        self.use_vae_dreams = bool(use_vae_dreams)
        if self.use_vae_dreams:
            self.dream_vae = DreamVAE(z_dim=128)
            self.vae_trained = False

        self.conflict_buffer = []
        self.max_conflicts = 100

        self.replay_buffer = {"X": [], "Y": []}
        self.max_replay_size = 1000

    def set_initial_responsibility(self, classes):
        self.active_classes_per_column[0] = classes

    def freeze_past(self, use_fractal_time=False, train_late_backbone=True):
        print("[FREEZING] Memory (Crystallization)...")

        # Option: "Fractal time" => freeze early layers; keep late trainable
        if use_fractal_time and train_late_backbone:
            print("[FRACTAL TIME] Freeze early backbone; keep late backbone trainable")
            for name, p in self.shared_backbone.named_parameters():
                if ("conv1" in name) or ("conv2" in name) or ("bn1" in name) or ("bn2" in name):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            # Full freeze backbone
            for p in self.shared_backbone.parameters():
                p.requires_grad = False

        # Freeze old heads (all except the last after expansion)
        for i in range(len(self.heads) - 1):
            for p in self.heads[i].parameters():
                p.requires_grad = False

        if self.use_subjective_time:
            self.ref_backbone = copy.deepcopy(self.shared_backbone)
            self.ref_backbone.eval()
            for p in self.ref_backbone.parameters():
                p.requires_grad = False

    def _set_bn_train(self, train: bool):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train(train)

    def recalibrate_bn(self, loader, device, num_batches=20):
        was_training = self.training
        self.eval()
        self._set_bn_train(True)
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= num_batches:
                    break
                _ = self.shared_backbone(x.to(device))
        self._set_bn_train(False)
        if was_training:
            self.train()
        else:
            self.eval()

    def expand(self, new_classes_indices, use_fractal_time=False, train_late_backbone=True):
        self.freeze_past(use_fractal_time=use_fractal_time, train_late_backbone=train_late_backbone)

        prev_dims = [h.hidden_size for h in self.heads]
        device = next(self.parameters()).device

        new_head = ExpandableHead(self.hidden_size, self.output_size, prev_dims).to(device)
        self.heads.append(new_head)

        new_col = TemporalColumn(self.hidden_size, self.output_size, prev_dims).to(device)
        self.columns.append(new_col)

        self.active_classes_per_column[len(self.heads) - 1] = new_classes_indices
        self.sensor = ComplexitySensor()
        print(f"[EMERGENCE] Head {len(self.heads)} created (shared backbone). Scope: {new_classes_indices}")
        return new_head

    def add_to_replay_buffer(self, X, Y):
        for x, y in zip(X, Y):
            if len(self.replay_buffer["X"]) < self.max_replay_size:
                self.replay_buffer["X"].append(x.detach().cpu().clone())
                self.replay_buffer["Y"].append(int(y.item()) if isinstance(y, torch.Tensor) else int(y))
            else:
                j = np.random.randint(0, self.max_replay_size)
                self.replay_buffer["X"][j] = x.detach().cpu().clone()
                self.replay_buffer["Y"][j] = int(y.item()) if isinstance(y, torch.Tensor) else int(y)

    def sample_replay_batch(self, batch_size, device):
        if len(self.replay_buffer["X"]) == 0:
            return None, None
        n = min(batch_size, len(self.replay_buffer["X"]))
        idx = np.random.choice(len(self.replay_buffer["X"]), n, replace=False)
        X = torch.stack([self.replay_buffer["X"][i] for i in idx]).to(device)
        Y = torch.tensor([self.replay_buffer["Y"][i] for i in idx], dtype=torch.long).to(device)
        return X, Y

    def record_conflict(self, confidence_model, entropy_model, clip_class, clip_label, clip_conf, image, true_label=None):
        self.conflict_buffer.append({
            "confidence_model": float(confidence_model),
            "entropy_model": float(entropy_model),
            "clip_class": int(clip_class) if clip_class is not None else None,
            "clip_label": str(clip_label),
            "clip_conf": float(clip_conf),
            "image": image.detach().clone(),
            "true_label": int(true_label) if true_label is not None else None
        })
        if len(self.conflict_buffer) > self.max_conflicts:
            self.conflict_buffer.pop(0)

    def get_conflict_statistics(self):
        if not self.conflict_buffer:
            return None
        total = len(self.conflict_buffer)
        correct_clip = sum(
            1 for c in self.conflict_buffer
            if c["true_label"] is not None and c["clip_class"] == c["true_label"]
        )
        avg_entropy = float(np.mean([c["entropy_model"] for c in self.conflict_buffer]))
        avg_clip_conf = float(np.mean([c["clip_conf"] for c in self.conflict_buffer]))
        return {
            "total_conflicts": total,
            "clip_correct": correct_clip,
            "clip_accuracy": correct_clip / total if total else 0.0,
            "avg_entropy": avg_entropy,
            "avg_clip_confidence": avg_clip_conf
        }

    def get_clip_soft_targets(self, images):
        if not self.use_curiosity:
            return None

        try:
            x = images.to("cuda", non_blocking=True)
            x = x * 0.5 + 0.5
            x = torch.clamp(x, 0, 1)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - self.curiosity.clip_mean) / self.curiosity.clip_std
            with torch.no_grad():
                logits_per_image, _ = self.curiosity.model(x, self.curiosity.text_inputs)
                probs = logits_per_image.softmax(dim=-1)  # [B,10]
            return probs
        except Exception as e:
            print(f"[WARNING] CLIP batch failed: {e}")
            return None

    def train_vae_on_data(self, loader, device, epochs=5, lr=1e-3):
        if not self.use_vae_dreams:
            return
        print(f"[VAE] Training dream generator on {len(loader)} batches...")
        self.dream_vae.to(device)
        opt = optim.Adam(self.dream_vae.parameters(), lr=lr)
        self.dream_vae.train()
        for ep in range(epochs):
            total = 0.0
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                x_recon, mu, logvar = self.dream_vae(x)
                l = vae_loss(x, x_recon, mu, logvar, beta=1.0)
                l.backward()
                opt.step()
                total += float(l.item())
            if (ep + 1) % 1 == 0:
                print(f"   VAE epoch {ep+1}/{epochs}: Loss {total/len(loader):.4f}")
        self.dream_vae.eval()
        self.vae_trained = True
        print("[VAE] Dream generator ready!")

    def sample_dreams(self, n, device):
        if self.use_vae_dreams and self.vae_trained:
            with torch.no_grad():
                z = torch.randn(n, self.dream_vae.z_dim, device=device)
                dreams = self.dream_vae.decode(z)
                dreams = torch.clamp(dreams, -1, 1)
            return dreams
        noise = torch.randn(n, 3, 32, 32, device=device)
        return torch.tanh(noise * 0.5)

    def forward(self, x, return_features=False):
        feats = self.shared_backbone(x)
        hiddens = []
        logits_sum = torch.zeros(x.size(0), self.output_size, device=x.device)

        for i, head in enumerate(self.heads):
            out, h = head(feats, hiddens)
            hiddens.append(h)
            if i in self.active_classes_per_column:
                idx = self.active_classes_per_column[i]
                mask = torch.zeros_like(out)
                mask[:, idx] = 1.0
                logits_sum = logits_sum + out * mask
            else:
                logits_sum = logits_sum + out

        # Unknown heuristic (not trained)
        probs_known = torch.softmax(logits_sum[:, :10], dim=1)
        entropy = -torch.sum(probs_known * torch.log(probs_known + 1e-9), dim=1)
        max_prob_known, _ = probs_known.max(dim=1)

        unknown_mask = (max_prob_known < 0.18) | (entropy > 1.95)
        if unknown_mask.any():
            max_logit_known, _ = logits_sum[:, :10].max(dim=1)
            logits_sum[unknown_mask, self.unknown_class_idx] = max_logit_known[unknown_mask] + 0.8

        if return_features:
            return logits_sum, feats
        return logits_sum


# ----------------------------
# CIFAR split
# ----------------------------
def get_cifar_split():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data")

    train_full = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)
    test_full = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transform)

    vehicles = [0, 1, 8, 9]
    animals = [2, 3, 4, 5, 6, 7]

    def idx_of(ds, classes):
        out = []
        for i in range(len(ds)):
            if ds.targets[i] in classes:
                out.append(i)
        return out

    print("Sorting Data into 'Machines' vs 'Nature'...")
    return (
        Subset(train_full, idx_of(train_full, vehicles)),
        Subset(train_full, idx_of(train_full, animals)),
        Subset(test_full, idx_of(test_full, vehicles)),
        Subset(test_full, idx_of(test_full, animals)),
        vehicles,
        animals,
    )


def pair_margin_loss(logits10, targets, pairs=((0, 8), (1, 9), (3, 5)), margin=0.2):
    """
    Enforce margin separation for confusing pairs (Plane↔Ship, Car↔Truck, Cat↔Dog).
    logits10: [B, 10]
    targets: [B]
    pairs: list of (class_a, class_b) tuples
    margin: minimum logit difference
    """
    loss = 0.0
    for a, b in pairs:
        mask_a = targets == a
        mask_b = targets == b
        if mask_a.any():
            loss = loss + F.relu(margin - (logits10[mask_a, a] - logits10[mask_a, b])).mean()
        if mask_b.any():
            loss = loss + F.relu(margin - (logits10[mask_b, b] - logits10[mask_b, a])).mean()
    return loss


@torch.no_grad()
def eval_masked(agent, loader, allowed_classes, device, block_unknown=True):
    was_training = agent.training
    agent.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = agent(x)
        out_masked = out.clone()
        out_masked[:, [i for i in range(10) if i not in allowed_classes]] = -float("inf")
        if block_unknown:
            out_masked[:, agent.unknown_class_idx] = -float("inf")
        pred = out_masked.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    if was_training:
        agent.train()
    return 100.0 * correct / max(1, total)


# ----------------------------
# Main run
# ----------------------------
def run_drone_simulation():
    setup_b200()
    device = torch.device("cuda")
    print(f"Running on: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1024 ** 3:.2f} GB")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print()

    train_A, train_B, test_A, test_B, classes_A, classes_B = get_cifar_split()

    loader_A = DataLoader(train_A, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    loader_B = DataLoader(train_B, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader_A = DataLoader(test_A, batch_size=500, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_B = DataLoader(test_B, batch_size=500, shuffle=False, num_workers=4, pin_memory=True)

    use_curiosity = CLIP_AVAILABLE
    use_subjective_time = True
    use_vae_dreams = True
    use_fractal_time = True
    train_late_backbone = True      # IMPORTANT: if True, include late backbone params into optimizer_phase2
    use_adaptive_pain = True

    agent = RecursiveAgent(
        use_curiosity=use_curiosity,
        use_subjective_time=use_subjective_time,
        use_vae_dreams=use_vae_dreams,
    ).to(device)

    # memory format optimization
    agent = agent.to(memory_format=torch.channels_last)

    agent.set_initial_responsibility(classes_A)

    if use_curiosity:
        print("[INFO] Curiosity Module (CLIP) enabled")
    if use_subjective_time:
        print("[INFO] Subjective Time Critic enabled")
    if use_vae_dreams:
        print("[INFO] VAE Dream Generator enabled")
    if use_fractal_time:
        print("[INFO] Fractal Time enabled")
    if use_adaptive_pain:
        print("[INFO] Adaptive Time/Pain enabled")

    # Separate criteria: training scalar + per-sample for critic
    criterion_train = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_none = nn.CrossEntropyLoss(reduction="none")

    # AMP BF16 on B200
    amp_dtype = torch.bfloat16
    scaler = None  # BF16 doesn't need GradScaler (it is for FP16)

    # Phase1 optimizer (all params)
    optimizer = optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    critic_optimizer = None
    if use_subjective_time:
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-3)

    acc_A_hist, acc_B_hist = [], []
    step = 0
    phase_transition_step = []

    print(f"\n--- PHASE 1: URBAN ENVIRONMENT (Learning Machines: {classes_A}) ---")

    if use_vae_dreams:
        print("[VAE] Pre-training dream generator on Phase 1 data...")
        agent.train_vae_on_data(loader_A, device, epochs=5, lr=1e-3)

    steps_per_epoch_A = len(loader_A)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch_A * 15, eta_min=1e-5)

    replay_samples_collected = 0

    for epoch in range(15):
        agent.train()
        for data, target in loader_A:
            data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, features = agent(data, return_features=True)
                loss = criterion_train(logits[:, :10], target)
                # Add pair margin loss to reduce Plane↔Ship, Car↔Truck, Cat↔Dog confusion
                pm = pair_margin_loss(logits[:, :10], target, margin=0.15)
                loss = loss + 0.05 * pm

                surprise = None
                if use_subjective_time:
                    pred_ps = agent.critic(features.detach())                 # [B]
                    real_ps = criterion_none(logits[:, :10], target)          # [B]
                    surprise = SubjectiveTimeCritic.surprise(pred_ps, real_ps)

                    # critic learns per-sample
                    critic_loss = F.mse_loss(pred_ps, real_ps.detach())
                else:
                    critic_loss = None

            # update critic first (separate backward)
            if use_subjective_time:
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward(retain_graph=True)
                critic_optimizer.step()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if replay_samples_collected < agent.max_replay_size:
                n = min(32, data.size(0))
                agent.add_to_replay_buffer(data[:n], target[:n])
                replay_samples_collected += n

            agent.sensor.update(float(loss.item()))
            if step == 50:
                agent.sensor.calibrate()

            if step % 50 == 0:
                acc = eval_masked(agent, test_loader_A, classes_A, device, block_unknown=True)
                acc_A_hist.append(acc)
                acc_B_hist.append(0)
                s = f" | Surprise: {float(surprise.item()):.4f}" if surprise is not None else ""
                print(f"Step {step}: Loss {float(loss.item()):.2f} | Acc Machines: {acc:.1f}%{s}")

            step += 1

    print(f"\n--- PHASE 2: WILDERNESS (Reality Shift to Animals: {classes_B}) ---")
    phase_transition_step.append(len(acc_A_hist))

    expansion_count = 0
    last_expansion_step = -1000
    COOLDOWN_STEPS = 200
    CLIP_TRUST_THRESHOLD = 0.6
    MAX_LAYERS = 5

    SLEEP_TRIGGER_STEPS = 500
    SLEEP_TRIGGER_ERRORS = 100
    error_count_phase2 = 0
    last_sleep_step = -1000

    optimizer_phase2 = None
    scheduler_phase2 = None
    steps_per_epoch_B = len(loader_B)
    total_steps_phase2 = steps_per_epoch_B * 8

    # Helper to build phase2 optimizer (head + optionally late backbone)
    def build_phase2_optimizer(new_head: nn.Module):
        head_params = list(new_head.parameters())
        late_backbone_params = [p for p in agent.shared_backbone.parameters() if p.requires_grad]

        if train_late_backbone and len(late_backbone_params) > 0:
            # two param groups: head fast, late backbone slow
            opt = optim.AdamW(
                [
                    {"params": head_params, "lr": 2e-3},
                    {"params": late_backbone_params, "lr": 5e-4},
                ],
                weight_decay=1e-4,
            )
        else:
            # head only
            opt = optim.AdamW(head_params, lr=2e-3, weight_decay=1e-4)
        return opt

    for epoch in range(8):
        agent.train()
        for data, target in loader_B:
            data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)

            # 1) shock check (no grad)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                test_out = agent(data)
                test_loss = criterion_train(test_out[:, :10], target)

            is_shock = agent.sensor.is_shock(float(test_loss.item()))
            can_expand = (step - last_expansion_step) > COOLDOWN_STEPS
            has_budget = len(agent.heads) < MAX_LAYERS

            if is_shock and can_expand and has_budget:
                print(f"\n[VISUAL CORTEX SHOCK] Loss {float(test_loss.item()):.2f} detected (High Surprise).")
                print(f"[SAFETY] Cooldown OK, Budget OK ({len(agent.heads)}/{MAX_LAYERS} heads)")

                if agent.use_curiosity:
                    print("[CURIOSITY] Querying Oracle (CLIP)...")
                    best_idx, best_label, conf = agent.curiosity.what_is_this(data[0:1])
                    if best_idx is not None:
                        if conf > CLIP_TRUST_THRESHOLD:
                            print(f"[EUREKA] CLIP confident ({conf*100:.1f}%): '{best_label}'")
                            print(f"[ADAPTATION] Triggering Phase Transition for concept: {best_label}...")

                            with torch.no_grad():
                                agent.eval()
                                mo = agent(data[0:1])
                                agent.train()
                                mp = torch.softmax(mo[:, :10], dim=1)
                                model_conf, model_pred = mp.max(dim=1)
                                model_entropy = float((-torch.sum(mp * torch.log(mp + 1e-9), dim=1)).item())
                                print(f"[LOG] Model confidence: {float(model_conf.item()):.3f}, Entropy: {model_entropy:.3f}")

                                agent.record_conflict(
                                    confidence_model=float(model_conf.item()),
                                    entropy_model=model_entropy,
                                    clip_class=best_idx,
                                    clip_label=best_label,
                                    clip_conf=conf,
                                    image=data[0:1],
                                    true_label=int(target[0].item()) if target.numel() > 0 else None,
                                )

                            new_head = agent.expand(
                                new_classes_indices=classes_B,
                                use_fractal_time=use_fractal_time,
                                train_late_backbone=train_late_backbone,
                            )
                            agent.recalibrate_bn(loader_B, device, num_batches=20)

                            optimizer_phase2 = build_phase2_optimizer(new_head)

                            steps_already_done = 0
                            remaining_steps = max(total_steps_phase2 - steps_already_done, steps_per_epoch_B)
                            scheduler_phase2 = CosineAnnealingLR(optimizer_phase2, T_max=remaining_steps, eta_min=1e-5)

                            expansion_count += 1
                            last_expansion_step = step

                            if use_subjective_time:
                                agent.ref_backbone = copy.deepcopy(agent.shared_backbone)
                                agent.ref_backbone.eval()
                                for p in agent.ref_backbone.parameters():
                                    p.requires_grad = False
                        else:
                            print(f"[IGNORE] CLIP unsure ({conf*100:.1f}% < {CLIP_TRUST_THRESHOLD*100:.0f}%). Skip expansion.")
            elif is_shock and not can_expand and (step % 50 == 0):
                remaining = COOLDOWN_STEPS - (step - last_expansion_step)
                print(f"[COOLDOWN] Shock detected but refractory period ({remaining} steps)")

            elif is_shock and not has_budget:
                print(f"\n[CRITICAL] Head Limit ({MAX_LAYERS}) reached. Consider SLEEP here (disabled in this file).")

            # Intelligent sleep trigger placeholder (kept, but no compression code here)
            steps_since_sleep = step - last_sleep_step
            should_sleep = (
                len(agent.heads) >= 2
                and expansion_count > 0
                and steps_since_sleep > SLEEP_TRIGGER_STEPS
                and (error_count_phase2 > SLEEP_TRIGGER_ERRORS or steps_since_sleep > SLEEP_TRIGGER_STEPS * 2)
            )
            if should_sleep:
                print(f"\n[INTELLIGENT SLEEP] Would trigger after {steps_since_sleep} steps and {error_count_phase2} errors (sleep disabled).")
                # sleep disabled => do NOT reset counters, only mark last_sleep_step for throttle
                last_sleep_step = step
                # error_count_phase2 remains accumulated

            # 2) training step
            current_opt = optimizer_phase2 if optimizer_phase2 is not None else optimizer
            current_opt.zero_grad(set_to_none=True)

            # sample replay once per step (used for replay loss + pain)
            x_replay, y_replay = agent.sample_replay_batch(batch_size=64, device=device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs, features = agent(data, return_features=True)
                loss_new = criterion_train(outputs[:, :10], target)
                # Add pair margin loss to reduce Plane↔Ship, Car↔Truck, Cat↔Dog confusion
                pm = pair_margin_loss(outputs[:, :10], target, margin=0.15)
                loss_new = loss_new + 0.05 * pm

                replay_loss = 0.0
                if x_replay is not None:
                    out_rep = agent(x_replay)
                    replay_loss = criterion_train(out_rep[:, :10], y_replay)

                # subjective time critic + stability anchor (normalized)
                surprise = None
                stability_loss = 0.0
                current_lambda = 10000.0

                if use_subjective_time and agent.ref_backbone is not None:
                    pred_ps = agent.critic(features.detach())
                    real_ps = criterion_none(outputs[:, :10], target)
                    surprise = SubjectiveTimeCritic.surprise(pred_ps, real_ps)
                    current_lambda = SubjectiveTimeCritic.compute_lambda(surprise, base_lambda=10000.0, sensitivity=10.0)

                    # normalized stability over trainable backbone params only
                    cnt = 0
                    for p, p_ref in zip(agent.shared_backbone.parameters(), agent.ref_backbone.parameters()):
                        if p.requires_grad:
                            stability_loss = stability_loss + F.mse_loss(p, p_ref, reduction="mean")
                            cnt += 1
                    if cnt > 0:
                        stability_loss = stability_loss / cnt

                    critic_loss = F.mse_loss(pred_ps, real_ps.detach())
                else:
                    critic_loss = None

                # adaptive pain: gradient conflict -> adaptive lambda
                adaptive_lambda = current_lambda
                if use_adaptive_pain and (x_replay is not None):
                    # only params that are actually updated by optimizer
                    backbone_params = [p for p in agent.shared_backbone.parameters() if p.requires_grad]
                    if len(backbone_params) > 0:
                        # Need grads; do NOT autocast for grad vectors reliability
                        # compute outside autocast block in fp32
                        pass

                # CLIP KL (gated by CLIP confidence)
                kl_loss = 0.0
                if agent.use_curiosity:
                    probs_model = torch.softmax(outputs[:, :10], dim=1)
                    ent = -torch.sum(probs_model * torch.log(probs_model + 1e-9), dim=1)
                    hi = ent > 1.5
                    if hi.any():
                        idx = torch.where(hi)[0]
                        MAX_UNCERTAIN = 16
                        if idx.numel() > MAX_UNCERTAIN:
                            idx = idx[:MAX_UNCERTAIN]
                        clip_targets = agent.get_clip_soft_targets(data[idx])
                        if clip_targets is not None:
                            clip_conf, _ = clip_targets.max(dim=1)
                            mask_ok = clip_conf > 0.8
                            if mask_ok.any():
                                idx2 = idx[mask_ok]
                                clip_probs_ok = clip_targets[mask_ok]
                                kl = F.kl_div(
                                    torch.log(probs_model[idx2] + 1e-9),
                                    clip_probs_ok,
                                    reduction="batchmean",
                                )
                                # lower max weight than before (0.1)
                                if expansion_count > 0:
                                    steps_since_expand = step - last_expansion_step
                                    kl_weight = min(0.1, 0.1 * (steps_since_expand / 500.0))
                                else:
                                    kl_weight = 0.1
                                kl_loss = kl_weight * kl
                                if step % 50 == 0:
                                    print(f"[LOG] High entropy: {int(idx2.numel())}, KL: {float(kl_loss.item()):.4f}")

                # total loss assembly
                total_loss = loss_new
                if x_replay is not None:
                    total_loss = total_loss + 0.25 * replay_loss
                if (use_subjective_time and agent.ref_backbone is not None) and (stability_loss != 0.0):
                    total_loss = total_loss + current_lambda * stability_loss
                if kl_loss != 0.0:
                    total_loss = total_loss + kl_loss

            # critic update (separate)
            if use_subjective_time and critic_loss is not None:
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward(retain_graph=True)
                critic_optimizer.step()

            # adaptive pain grad conflict in fp32 (optional)
            if use_adaptive_pain and (x_replay is not None):
                backbone_params = [p for p in agent.shared_backbone.parameters() if p.requires_grad]
                if len(backbone_params) > 0:
                    # compute gradients for new and old tasks
                    # Use fp32 for grad vectors
                    with torch.amp.autocast("cuda", enabled=False):
                        out_new = agent(data.float())
                        ln = criterion_train(out_new[:, :10], target)
                        out_old = agent(x_replay.float())
                        lo = criterion_train(out_old[:, :10], y_replay)

                    g_new = torch.autograd.grad(ln, backbone_params, retain_graph=True, allow_unused=True)
                    g_old = torch.autograd.grad(lo, backbone_params, retain_graph=True, allow_unused=True)
                    g_new = [g for g in g_new if g is not None]
                    g_old = [g for g in g_old if g is not None]

                    if len(g_new) and len(g_old):
                        gn = torch.cat([g.detach().flatten() for g in g_new])
                        go = torch.cat([g.detach().flatten() for g in g_old])
                        dot = float(torch.dot(gn, go).item())
                        n1 = float(gn.pow(2).sum().sqrt().item()) + 1e-8
                        n2 = float(go.pow(2).sum().sqrt().item()) + 1e-8
                        cos = dot / (n1 * n2)
                        pain = max(0.0, min(1.0, (1.0 - cos) * 0.5))
                        adaptive_lambda = 100.0 + (20000.0 - 100.0) * pain

                        # if stability loss exists, replace lambda (re-weight stability)
                        if use_subjective_time and agent.ref_backbone is not None and stability_loss != 0.0:
                            # small correction term (avoid double-counting)
                            # We approximate by adding delta_lambda * stability_loss
                            delta = adaptive_lambda - current_lambda
                            if abs(delta) > 1e-6:
                                total_loss = total_loss + float(delta) * stability_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            current_opt.step()

            if scheduler_phase2 is not None:
                scheduler_phase2.step()
            elif optimizer_phase2 is None:
                scheduler.step()

            agent.sensor.update(float(total_loss.item()))

            # error counter
            if expansion_count > 0:
                with torch.no_grad():
                    agent.eval()
                    out = agent(data)
                    pred = out[:, :10].argmax(dim=1)
                    error_count_phase2 += int((pred != target).sum().item())
                    agent.train()

            if step % 50 == 0:
                acc_A = eval_masked(agent, test_loader_A, classes_A, device, block_unknown=True)
                acc_B = eval_masked(agent, test_loader_B, classes_B, device, block_unknown=True)
                acc_A_hist.append(acc_A)
                acc_B_hist.append(acc_B)

                with torch.no_grad():
                    out = agent(data)
                    p = torch.softmax(out[:, :10], dim=1)
                    ent = -torch.sum(p * torch.log(p + 1e-9), dim=1)
                    mp, _ = p.max(dim=1)
                    unk_rate = ((mp < 0.2) | (ent > 1.8)).float().mean().item()

                s = f"{float(surprise.item()):.4f}" if surprise is not None else "n/a"
                print(
                    f"Step {step}: Loss {float(total_loss.item()):.2f} | Mem(M): {acc_A:.1f}% | "
                    f"New(A): {acc_B:.1f}% | Heads: {len(agent.heads)} | UnknownRate: {unk_rate*100:.1f}% | "
                    f"Errors: {error_count_phase2} | Surprise: {s}"
                )

            step += 1

    # ----------------------------
    # Final test on all classes (fixed unknown counting)
    # ----------------------------
    print("\n--- TESTING ON ALL CLASSES (Including Unseen) ---")

    class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck", "Unknown"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data")
    test_full = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    test_loader_all = DataLoader(test_full, batch_size=500, shuffle=False, num_workers=4, pin_memory=True)

    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    class_predictions = {i: {j: 0 for j in range(11)} for i in range(10)}
    unknown_count = 0

    agent.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        for data, target in test_loader_all:
            data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            outputs = agent(data)

            # unblocked pred (for unknown statistics)
            pred_unblocked = outputs.argmax(dim=1)
            unknown_count += int((pred_unblocked == agent.unknown_class_idx).sum().item())

            # blocked pred for accuracy (mask by group and forbid unknown)
            for i in range(data.size(0)):
                out = outputs[i:i+1]
                true_class = int(target[i].item())

                if true_class in classes_A:
                    outm = out.clone()
                    outm[:, [j for j in range(10) if j not in classes_A]] = -float("inf")
                    outm[:, agent.unknown_class_idx] = -float("inf")
                else:
                    outm = out.clone()
                    outm[:, [j for j in range(10) if j not in classes_B]] = -float("inf")
                    outm[:, agent.unknown_class_idx] = -float("inf")

                pred = int(outm.argmax(dim=1).item())

                class_total[true_class] += 1
                class_predictions[true_class][pred] += 1
                if pred == true_class:
                    class_correct[true_class] += 1

    print("\n=== CLASSIFICATION RESULTS ===")
    print(f"{'Class':<10} {'Name':<10} {'Trained':<10} {'Accuracy':<10} {'Total':<10}")
    print("-" * 50)

    for i, name in enumerate(class_names):
        if i < 10:
            trained = "YES" if (i in classes_A or i in classes_B) else "NO"
            acc = 100.0 * class_correct[i] / max(1, class_total[i])
            print(f"{i:<10} {name:<10} {trained:<10} {acc:>6.1f}%    {class_total[i]:<10}")
        else:
            print(f"{i:<10} {name:<10} {'N/A':<10} {'N/A':<10} {unknown_count:<10} (times predicted)")

    print("\n=== ERROR ANALYSIS ===")
    for true_class in range(10):
        if class_total[true_class] > 0:
            errors = [(pred_c, cnt) for pred_c, cnt in class_predictions[true_class].items() if pred_c != true_class and cnt > 0]
            if errors:
                errors.sort(key=lambda x: x[1], reverse=True)
                pred_c, cnt = errors[0]
                print(f"{class_names[true_class]} -> {class_names[pred_c]}: {cnt} times")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(acc_A_hist, label="Urban (Machines)", linewidth=3)
    axes[0].plot(acc_B_hist, label="Nature (Animals)", linewidth=3)
    if phase_transition_step:
        axes[0].axvline(x=phase_transition_step[0], color="r", linestyle="--", label="Environment Shift")
    axes[0].set_title("Recursive Emergence (CIFAR-10)")
    axes[0].set_ylabel("Accuracy %")
    axes[0].set_xlabel("Training Steps (x50)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    class_accs = [100.0 * class_correct[i] / max(1, class_total[i]) for i in range(10)]
    colors = ["green" if i in classes_A else "blue" for i in range(10)]
    bars = axes[1].bar(range(10), class_accs, color=colors, alpha=0.7)
    axes[1].set_title("Accuracy by Class (Phase1=Green, Phase2=Blue)")
    axes[1].set_ylabel("Accuracy %")
    axes[1].set_xlabel("Class")
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels([f"{i}\n{class_names[i]}" for i in range(10)], rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 100)

    for bar, acc in zip(bars, class_accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(project_root, "test_outputs", "cifar10_drone_result_b200.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[SAVED] Graph saved as {out_path}")
    plt.show()


if __name__ == "__main__":
    run_drone_simulation()
