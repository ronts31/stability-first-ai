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


# КРИТИЧНО: Soft Routing Gate для устойчивости к масштабированию
class SoftRoutingGate(nn.Module):
    """
    Обучаемый gate для soft routing вместо жёсткого маскирования.
    Позволяет системе "сшивать" навыки, а не только хранить их.
    Управляется через temperature для контроля сложности.
    """
    def __init__(self, feature_dim, num_heads, temperature=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.base_temperature = temperature  # базовая temperature
        self.current_temperature = temperature  # текущая (управляется сложностью)
        # Маленькая сеть для предсказания ответственности каждого head
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_heads),
        )
    
    def set_temperature(self, temperature):
        """Устанавливает temperature для управления routing (высокая = равномерный, низкая = уверенный)"""
        self.current_temperature = max(0.1, min(5.0, float(temperature)))
    
    def forward(self, features):
        """
        Args:
            features: [B, feature_dim] - features из backbone
        Returns:
            gates: [B, num_heads] - soft weights для каждого head (sum=1)
        """
        logits = self.gate_net(features) / self.current_temperature
        gates = torch.softmax(logits, dim=-1)  # [B, num_heads]
        return gates


# КРИТИЧНО: Complexity Controller - единый метаконтроллер для управления временем и рекурсией
class ComplexityController:
    """
    Управляет временем и рекурсией через управление сложностью.
    Превращает разрозненные сигналы в единую политику вычисления/памяти.
    """
    def __init__(self):
        # Веса для вычисления состояния сложности
        self.w_surprise = 0.30
        self.w_pain = 0.25
        self.w_entropy = 0.25
        self.w_unknown = 0.20
        
        # Complexity Budget (закон сохранения сложности)
        self.complexity_budget = 1.0  # [0..1]
        self.budget_recovery_rate = 0.015  # восстановление за шаг (увеличено для лучшего баланса)
        self.budget_decay_rate = 0.998  # медленное затухание
        
        # Стоимости действий (уменьшены для баланса с recovery_rate)
        self.cost_recursion = 0.015  # за один рекурсивный проход (ещё уменьшено)
        self.cost_replay = 0.003  # за единицу replay_ratio (ещё уменьшено)
        self.cost_kl = 0.008  # за KL distillation (ещё уменьшено)
        self.cost_expansion = 0.30  # за expansion (оставляем высоким, т.к. это редкое событие)
        
        # История для стабилизации
        self.complexity_history = []
        self.max_history = 100
    
    def compute_complexity(self, surprise, pain, entropy, unknown_rate):
        """
        Вычисляет единое состояние сложности из разрозненных сигналов.
        
        Args:
            surprise: float [0..2+] - неожиданность от SubjectiveTimeCritic
            pain: float [0..1] - конфликт градиентов
            entropy: float [0..log(num_classes)] - энтропия предсказаний
            unknown_rate: float [0..1] - доля unknown предсказаний
        
        Returns:
            complexity: float [0..1] - единое состояние сложности
        """
        # Нормализуем входы
        s_n = min(1.0, surprise / 1.2)  # surprise ~ 0..2
        p_n = min(1.0, pain)  # pain уже [0..1]
        e_n = min(1.0, entropy / 2.3)  # max entropy для 10 классов ≈ 2.3
        u_n = unknown_rate  # уже [0..1]
        
        # Взвешенная сумма
        C = (
            self.w_surprise * s_n +
            self.w_pain * p_n +
            self.w_entropy * e_n +
            self.w_unknown * u_n
        )
        C = max(0.0, min(1.0, C))  # clamp [0..1]
        
        # Обновляем историю
        self.complexity_history.append(C)
        if len(self.complexity_history) > self.max_history:
            self.complexity_history.pop(0)
        
        return C
    
    def get_actions(self, complexity, has_expansion_budget, cooldown_ok):
        """
        Выдаёт действия на основе состояния сложности.
        
        Returns:
            dict с ключами:
                - n_recursions: int [1..3] - сколько рекурсивных проходов
                - replay_ratio: float [0.1..0.4] - доля replay в батче
                - gate_temperature: float [0.7..2.0] - temperature для routing
                - crystal_target: float [0..1] - целевой crystal_level
                - expand_allowed: bool - разрешение на expansion
        """
        # КРИТИЧНО: n_recursions зависит от сложности
        n_recursions = 1 + int(round(2.0 * complexity))  # 1..3
        
        # replay_ratio: высокая сложность → больше памяти
        replay_ratio = 0.10 + 0.30 * complexity  # 10%..40%
        
        # gate_temperature: высокая сложность → более равномерный routing (поиск)
        gate_temperature = 0.7 + 1.3 * complexity  # 0.7..2.0
        
        # crystal_target: высокая сложность → меньше кристаллизации (больше пластичности)
        crystal_target = 1.0 - complexity
        
        # expand_allowed: только при высокой сложности и наличии ресурсов
        expand_allowed = (complexity > 0.7 and has_expansion_budget and cooldown_ok)
        
        return {
            "n_recursions": n_recursions,
            "replay_ratio": replay_ratio,
            "gate_temperature": gate_temperature,
            "crystal_target": crystal_target,
            "expand_allowed": expand_allowed,
        }
    
    def update_budget(self, actions, used_expansion=False, used_kl=False):
        """
        Обновляет complexity budget на основе использованных действий.
        
        Args:
            actions: dict от get_actions()
            used_expansion: bool - был ли использован expansion
            used_kl: bool - был ли использован KL distillation
        """
        # Расходуем budget на действия
        cost = 0.0
        cost += actions["n_recursions"] * self.cost_recursion
        cost += actions["replay_ratio"] * self.cost_replay
        if used_kl:
            cost += self.cost_kl
        if used_expansion:
            cost += self.cost_expansion
        
        # Тратим budget
        self.complexity_budget = max(0.0, self.complexity_budget - cost)
        
        # Восстанавливаем budget (медленно)
        self.complexity_budget = min(1.0, self.complexity_budget + self.budget_recovery_rate)
    
    def get_budget_status(self):
        """Возвращает статус budget для логирования"""
        return {
            "budget": self.complexity_budget,
            "avg_complexity": float(np.mean(self.complexity_history)) if self.complexity_history else 0.0,
        }


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
        
        # КРИТИЧНО: Soft routing вместо жёсткого маскирования
        self.use_soft_routing = True  # можно отключить для обратной совместимости
        # КРИТИЧНО: создаём gate на MAX_LAYERS один раз, чтобы не терять обучение при expansion
        # MAX_LAYERS будет определён позже в run_drone_simulation, пока создаём на 1 head
        # Gate будет пересоздан в run_drone_simulation с правильным размером
        self.routing_gate = SoftRoutingGate(feature_dim=self.hidden_size, num_heads=1)
        
        # Growth budget для устойчивого expansion
        self.growth_budget = 1.0  # начальный budget
        self.growth_cost_per_expansion = 0.3  # стоимость одного expansion
        self.unknown_trained = False  # флаг для отключения эвристики Unknown
        
        # КРИТИЧНО: Complexity Controller для управления временем и рекурсией
        self.complexity_controller = ComplexityController()

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

        # --- TIME CRYSTALLIZATION STATE ---
        self.crystal_level = 0.0       # [0..1]
        self.crystal_momentum = 0.98   # EMA для сглаживания
        self.crystal_lock_threshold = 0.92
        self.crystal_unlock_threshold = 0.70
        self.crystal_lock_steps = 0
        self.crystal_lock_patience = 150  # сколько шагов держать высокий уровень, чтобы hard-freeze
        self.hard_frozen_layers = set()

        # фрактальные веса защиты по группам слоёв (ранние сильнее)
        self.fractal_alpha = {
            "early": 1.00,   # conv1/conv2/bn1/bn2
            "mid":   0.35,   # conv3/bn3
            "late":  0.15,   # conv4/bn4
        }

    def _layer_group(self, name: str) -> str:
        """Определяет группу слоя для фрактальной защиты."""
        if any(k in name for k in ["conv1", "conv2", "bn1", "bn2"]):
            return "early"
        if any(k in name for k in ["conv3", "bn3"]):
            return "mid"
        if any(k in name for k in ["conv4", "bn4"]):
            return "late"
        return "late"

    @torch.no_grad()
    def update_time_crystallization(self, surprise: float, pain: float, entropy: float):
        """
        Обновляет уровень кристаллизации времени на основе неожиданности, конфликта и энтропии.
        
        Args:
            surprise: чем выше, тем больше пластичность (меньше кристаллизация)
            pain: конфликт градиентов [0..1], чем выше тем больше пластичность
            entropy: неопределенность, чем выше тем больше пластичность
        """
        # Нормируем входы (важно: без резких скачков)
        s = float(surprise)
        p = float(pain)
        e = float(entropy)

        # Стабильность мира = низкая неожиданность/конфликт/неопределенность
        # Значения под CIFAR-10: surprise ~ 0..2, entropy ~ 0..2.3
        s_n = min(1.0, s / 1.2)          # 0..1
        e_n = min(1.0, e / 2.0)          # 0..1
        p_n = min(1.0, p)                # 0..1

        # Усилен вес нестабильности (surprise/entropy важнее) чтобы crystal_level не рос слишком быстро
        instability = 0.65 * s_n + 0.20 * p_n + 0.15 * e_n
        target_crystal = 1.0 - instability  # чем стабильнее, тем больше кристалл

        # EMA с более быстрой реакцией вниз (меньше momentum при нестабильности)
        # Если нестабильность высокая - быстрее снижаем crystal_level
        effective_momentum = self.crystal_momentum if target_crystal > self.crystal_level else 0.95
        self.crystal_level = effective_momentum * self.crystal_level + (1 - effective_momentum) * target_crystal
        self.crystal_level = float(max(0.0, min(1.0, self.crystal_level)))

    def _alpha_by_group(self, group: str):
        """
        Динамические веса защиты в зависимости от crystal_level и группы слоя.
        Late кристаллизуется быстрее, early остается жидким дольше.
        """
        c = float(self.crystal_level)  # 0..1

        # КРИТИЧНО: уменьшены alpha примерно в 10 раз для предотвращения взрыва loss
        # late кристаллизуется быстрее: уже при c~0.3-0.5 заметно растёт
        if group == "late":
            return 0.01 + 0.09 * (c ** 1.0)

        # mid умеренно
        if group == "mid":
            return 0.005 + 0.045 * (c ** 1.5)

        # early — остаётся жидким дольше: растёт медленно и поздно
        # при c<0.5 почти не мешает учиться новым текстурам
        return 0.002 + 0.020 * (c ** 3.0)

    def crystallization_regularizer(self):
        """
        Пер-слойная защита (EWC-lite): ||W - W_ref||^2 с динамическим весом в зависимости от crystal_level.
        Требует наличия self.ref_backbone (снимок).
        Нормализовано через mean() для сопоставимости между слоями.
        """
        if (not self.use_subjective_time) or (self.ref_backbone is None):
            return 0.0

        reg = 0.0
        cnt = 0
        # КРИТИЧНО: матчим параметры по имени, а не через zip (более надежно)
        ref_params = dict(self.ref_backbone.named_parameters())
        for name, p in self.shared_backbone.named_parameters():
            if not p.requires_grad:
                continue
            if name not in ref_params:
                continue  # пропускаем если параметр отсутствует в ref
            p_ref = ref_params[name]
            group = self._layer_group(name)
            # Динамический alpha в зависимости от crystal_level и группы
            alpha = self._alpha_by_group(group)
            # Нормализация через mean() для сопоставимости между слоями
            reg = reg + (alpha * (p - p_ref).pow(2).mean())
            cnt += 1
        if cnt > 0:
            reg = reg / cnt  # среднее по всем слоям
        return reg

    def auto_hard_freeze_if_needed(self):
        """
        Опционально: если crystal_level долго высокий — замораживаем ранние слои автоматически.
        Если упал — размораживаем.
        """
        # lock/unlock ранних слоёв
        if self.crystal_level >= self.crystal_lock_threshold:
            self.crystal_lock_steps += 1
        else:
            self.crystal_lock_steps = max(0, self.crystal_lock_steps - 2)

        # LOCK
        if self.crystal_lock_steps >= self.crystal_lock_patience:
            for name, p in self.shared_backbone.named_parameters():
                if self._layer_group(name) == "early":
                    if p.requires_grad:  # только если еще не заморожен
                        p.requires_grad = False
                        self.hard_frozen_layers.add(name)

        # UNLOCK (если мир снова нестабилен)
        if self.crystal_level <= self.crystal_unlock_threshold and len(self.hard_frozen_layers) > 0:
            for name, p in self.shared_backbone.named_parameters():
                if name in self.hard_frozen_layers:
                    p.requires_grad = True
            self.hard_frozen_layers.clear()

    def time_lr_scale(self):
        """
        Возвращает коэффициенты скорости обучения для групп слоёв в зависимости от crystal_level.
        crystal_level=0 => всё пластично (scale=1.0)
        crystal_level=1 => всё почти заморожено (особенно early)
        """
        c = float(self.crystal_level)
        # чем выше кристалл, тем сильнее заморозка (early сильнее late)
        # КРИТИЧНО: head ослаблен (было min 0.50, теперь min 0.80) для лучшей адаптации в Phase2
        return {
            "early": max(0.02, 1.0 - 0.98 * c),   # при c=1 => 0.02
            "mid":   max(0.05, 1.0 - 0.90 * c),   # при c=1 => 0.10
            "late":  max(0.15, 1.0 - 0.70 * c),   # при c=1 => 0.30
            "head":  max(0.80, 1.0 - 0.10 * c),   # голова почти всегда учится (ослаблено с 0.50)
        }

    def set_initial_responsibility(self, classes):
        self.active_classes_per_column[0] = classes

    def freeze_past(self, use_fractal_time=False, train_late_backbone=True):
        print("[FREEZING] Memory (Crystallization)...")

        # Option: "Fractal time" => freeze early layers; keep late trainable
        if use_fractal_time and train_late_backbone:
            print("[FRACTAL TIME] Freeze early backbone; keep late backbone trainable")
            for name, p in self.shared_backbone.named_parameters():
                # КРИТИЧНО: не трогаем уже замороженные hard-freeze слои
                if name in self.hard_frozen_layers:
                    continue  # сохраняем hard-freeze состояние
                group = self._layer_group(name)
                if group == "early":
                    p.requires_grad = False
                elif group in ["mid", "late"]:
                    # Размораживаем mid/late только если они не были hard-frozen
                    p.requires_grad = True
                # else (head layers) не трогаем
        else:
            # Full freeze backbone (но сохраняем hard-frozen состояние)
            for name, p in self.shared_backbone.named_parameters():
                if name not in self.hard_frozen_layers:
                    p.requires_grad = False
                # hard-frozen остаются замороженными

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
                # КРИТИЧНО: используем memory_format для консистентности с тренировкой
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                _ = self.shared_backbone(x)
        self._set_bn_train(False)
        if was_training:
            self.train()
        else:
            self.eval()

    def expand(self, new_classes_indices, use_fractal_time=False, train_late_backbone=True):
        """
        Расширяет агента новой головой. Кристаллизация теперь автоматическая через update_time_crystallization.
        """
        # Freeze old heads (все кроме последней после расширения)
        for i in range(len(self.heads) - 1):
            for p in self.heads[i].parameters():
                p.requires_grad = False

        # Обновляем ref_backbone для кристаллизации (если используется SubjectiveTimeCritic)
        if self.use_subjective_time:
            self.ref_backbone = copy.deepcopy(self.shared_backbone)
            self.ref_backbone.eval()
            for p in self.ref_backbone.parameters():
                p.requires_grad = False
            print("[CRYSTALLIZATION] Reference backbone snapshot created for automatic time crystallization")

        prev_dims = [h.hidden_size for h in self.heads]
        device = next(self.parameters()).device

        new_head = ExpandableHead(self.hidden_size, self.output_size, prev_dims).to(device)
        self.heads.append(new_head)

        new_col = TemporalColumn(self.hidden_size, self.output_size, prev_dims).to(device)
        self.columns.append(new_col)

        self.active_classes_per_column[len(self.heads) - 1] = new_classes_indices
        self.sensor = ComplexitySensor()
        
        # Проверка: убеждаемся что mid/late backbone trainable (для warmup)
        if train_late_backbone:
            mid_late_params = []
            for name, p in self.shared_backbone.named_parameters():
                g = self._layer_group(name)
                if g in ["mid", "late"]:
                    mid_late_params.append((name, p.requires_grad))
            trainable_count = sum(1 for _, req in mid_late_params if req)
            print(f"[EMERGENCE] Head {len(self.heads)} created (shared backbone). Scope: {new_classes_indices}")
            print(f"[EMERGENCE] Mid/Late backbone trainable: {trainable_count}/{len(mid_late_params)} params")
        else:
            print(f"[EMERGENCE] Head {len(self.heads)} created (shared backbone). Scope: {new_classes_indices}")
        
        print(f"[CRYSTALLIZATION] Automatic time crystallization enabled (crystal_level will adapt dynamically)")
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
        # КРИТИЧНО: сохраняем image на CPU для экономии памяти буфера
        self.conflict_buffer.append({
            "confidence_model": float(confidence_model),
            "entropy_model": float(entropy_model),
            "clip_class": int(clip_class) if clip_class is not None else None,
            "clip_label": str(clip_label),
            "clip_conf": float(clip_conf),
            "image": image.detach().cpu().clone(),
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
        
        # КРИТИЧНО: Soft routing вместо жёсткого маскирования
        if self.use_soft_routing and len(self.heads) > 1:
            # КРИТИЧНО: используем срез из предварительно созданного gate (не пересоздаём!)
            gates_full = self.routing_gate(feats)  # [B, MAX_LAYERS]
            gates = gates_full[:, :len(self.heads)]  # [B, H] - берём только активные heads
            # Нормализуем (на случай если сумма != 1 из-за среза)
            gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-9)
            
            # КРИТИЧНО: смешиваем вероятности, а не логиты (снижает "искусственную" энтропию)
            # Собираем вероятности от всех heads с soft weights
            probs_sum = torch.zeros(x.size(0), self.output_size, device=x.device)
            for i, head in enumerate(self.heads):
                out, h = head(feats, hiddens)
                hiddens.append(h)
                # Soft routing: взвешенная сумма вероятностей (не логитов!)
                p_i = torch.softmax(out, dim=1)
                probs_sum = probs_sum + gates[:, i:i+1] * p_i
            
            # Конвертируем обратно в логиты для совместимости
            logits_sum = torch.log(probs_sum + 1e-9)
        else:
            # Fallback: жёсткое маскирование (для обратной совместимости)
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

        # КРИТИЧНО: Unknown эвристика отключена если unknown обучается (expansion_count > 0)
        # Эвристика применяется только в eval и только до обучения unknown
        if not self.training and not self.unknown_trained:
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


def compute_class_weights_from_dataset(dataset, num_classes=10, beta=0.9999):
    """
    Вычисляет веса классов на основе глобальных counts по датасету (не по батчу).
    Возвращает фиксированные веса для использования в loss.
    Оптимизировано для torch.utils.data.Subset.
    """
    import numpy as np
    from torch.utils.data import Subset
    
    # КРИТИЧНО: оптимизация для Subset - используем targets напрямую
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = dataset.indices
        # Используем targets исходного датасета (быстрее чем dataset[i][1])
        if hasattr(base, 'targets'):
            targets = np.array(base.targets)[indices]
        else:
            # Fallback на медленный способ если targets нет
            targets = np.array([dataset[i][1] for i in range(len(dataset))])
        counts = torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=num_classes).float()
    else:
        # Медленный способ для обычных датасетов
        class_counts = torch.zeros(num_classes, dtype=torch.float32)
        for i in range(len(dataset)):
            target = dataset[i][1]  # (image, target)
            if isinstance(target, torch.Tensor):
                target = target.item()
            class_counts[target] += 1.0
        counts = class_counts
    
    # Правильная формула Class-Balanced Loss (Cui et al.)
    # effective_num = 1 - beta^n
    # weight = (1 - beta) / (1 - beta^n)
    beta_tensor = torch.tensor(beta, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(beta_tensor, counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    
    # Нормализация: средний вес = 1.0
    weights = weights / weights.mean()
    
    return weights


def compute_class_weights_for_active_classes(dataset_subset, active_classes, beta=0.9999):
    """
    Вычисляет веса классов только для активных классов в subset.
    Это исправляет проблему когда subset содержит только часть классов (например, только animals).
    
    Args:
        dataset_subset: torch.utils.data.Subset с данными только для active_classes
        active_classes: список глобальных индексов классов (например [2,3,4,5,6,7])
        beta: параметр для Class-Balanced Loss
    
    Returns:
        torch.Tensor[10] - веса для всех 10 классов, но только active_classes имеют ненулевые веса
    """
    import numpy as np
    from torch.utils.data import Subset
    
    # Создаем маппинг глобальный индекс -> локальный индекс
    class_to_local = {c: i for i, c in enumerate(active_classes)}
    
    # Подсчитываем counts только для активных классов
    counts = torch.zeros(len(active_classes), dtype=torch.float32)
    
    # Оптимизация для Subset
    if isinstance(dataset_subset, Subset):
        base = dataset_subset.dataset
        indices = dataset_subset.indices
        if hasattr(base, 'targets'):
            targets = np.array(base.targets)[indices]
        else:
            targets = np.array([dataset_subset[i][1] for i in range(len(dataset_subset))])
        
        for target in targets:
            if target in class_to_local:
                counts[class_to_local[target]] += 1.0
    else:
        # Медленный способ для обычных датасетов
        for i in range(len(dataset_subset)):
            y = dataset_subset[i][1]
            if isinstance(y, torch.Tensor):
                y = y.item()
            y = int(y)
            if y in class_to_local:
                counts[class_to_local[y]] += 1.0
    
    # Правильная формула Class-Balanced Loss (Cui et al.)
    beta_tensor = torch.tensor(beta, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(beta_tensor, counts)
    w = (1.0 - beta) / (effective_num + 1e-8)
    
    # Нормализация: средний вес = 1.0 (только по активным классам)
    w = w / w.mean()
    
    # Создаем выходной тензор для всех 10 классов
    out = torch.zeros(10, dtype=torch.float32)
    for c, wi in zip(active_classes, w):
        out[c] = wi
    
    return out


def class_balanced_loss(logits, targets, class_weights, num_classes=10):
    """
    Class-balanced loss с фиксированными весами (вычисленными по датасету).
    
    Args:
        logits: [B, num_classes]
        targets: [B]
        class_weights: [num_classes] - фиксированные веса классов
        num_classes: количество классов
    
    Returns:
        weighted_loss: scalar
    """
    # Веса для каждого примера в батче
    sample_weights = class_weights[targets]  # [B]
    
    # Стандартный cross-entropy с весами
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
    weighted_loss = (ce_loss * sample_weights).mean()
    
    return weighted_loss


def check_clip_diversity(agent, data_batch, target_classes, min_diversity=3, confidence_threshold=0.6):
    """
    Проверяет разнообразие CLIP ответов перед expansion.
    Требует минимум min_diversity разных концептов с confidence > threshold.
    
    КРИТИЧНО: для рекурсивной эмергенции также обнаруживает НОВЫЕ концепты (не в target_classes),
    которые могут триггерить expansion для создания новых heads.
    
    Returns:
        (is_diverse, detected_classes, diversity_info, new_concepts)
    """
    if not agent.use_curiosity:
        return True, set(), "CLIP not available", set()
    
    detected_classes = set()
    new_concepts = set()  # концепты, не входящие в target_classes (потенциальные новые heads)
    sample_size = min(32, data_batch.size(0))  # проверяем первые N образцов
    
    for i in range(sample_size):
        best_idx, best_label, conf = agent.curiosity.what_is_this(data_batch[i:i+1])
        if best_idx is not None and conf > confidence_threshold:
            if best_idx in target_classes:  # целевые классы
                detected_classes.add(best_idx)
            elif best_idx < 10:  # новый концепт из CIFAR-10, но не в текущих target_classes
                new_concepts.add(best_idx)
    
    is_diverse = len(detected_classes) >= min_diversity
    diversity_info = f"Detected {len(detected_classes)}/{min_diversity} diverse concepts: {sorted(detected_classes)}"
    if new_concepts:
        diversity_info += f" | New concepts (potential expansion): {sorted(new_concepts)}"
    
    return is_diverse, detected_classes, diversity_info, new_concepts


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
    
    # КРИТИЧНО: рекалибруем сенсор для Phase2 (baseline другой, loss масштаб другой)
    print("[SENSOR] Recalibrating sensor for Phase2 (new baseline)...")
    agent.sensor = ComplexitySensor(sensitivity=2.5)  # новый сенсор для Phase2
    sensor_recalibrated = False
    phase2_steps = 0  # КРИТИЧНО: локальный счётчик для Phase2 (не глобальный step!)
    
    # Создаём teacher-снимок для distillation на dreams (сон с удержанием структуры)
    teacher = None
    if use_vae_dreams and agent.vae_trained:
        print("[SLEEP] Creating teacher snapshot for dream distillation...")
        teacher = copy.deepcopy(agent)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print("[SLEEP] Teacher ready for dream supervision")
        
        # КРИТИЧНО: переобучаем VAE на Phase2 данных для более релевантных dreams
        print("[VAE] Fine-tuning dream generator on Phase2 data (animals)...")
        agent.train_vae_on_data(loader_B, device, epochs=3, lr=5e-4)  # меньше эпох, меньший LR
        print("[VAE] Dream generator fine-tuned for Phase2")
    
    # Вычисляем фиксированные веса классов для class-balanced loss (один раз по датасету)
    # КРИТИЧНО: train_B содержит только классы 2..7 (animals), поэтому нужно считать веса
    # только для активных классов, иначе получим неправильные веса (0 для отсутствующих классов)
    print("[PHASE2] Computing class weights for balanced loss (active classes only)...")
    class_weights_phase2 = compute_class_weights_for_active_classes(train_B, classes_B, beta=0.9999)
    class_weights_phase2 = class_weights_phase2.to(device)
    print(f"[PHASE2] Class weights: {class_weights_phase2.cpu().numpy()}")
    print(f"[PHASE2] Active classes {classes_B} weights: {class_weights_phase2[torch.tensor(classes_B, device=device)].cpu().numpy()}")

    expansion_count = 0
    last_expansion_step = -1000
    clip_diversity_at_expansion = 0  # сохраняем разнообразие CLIP при expansion
    COOLDOWN_STEPS = 200
    CLIP_TRUST_THRESHOLD = 0.6
    CLIP_MIN_DIVERSITY = 3  # минимум разных концептов для expansion
    MAX_LAYERS = 5
    
    # КРИТИЧНО: пересоздаём routing gate с правильным размером MAX_LAYERS
    if agent.use_soft_routing:
        agent.routing_gate = SoftRoutingGate(feature_dim=agent.hidden_size, num_heads=MAX_LAYERS).to(device)
    
    # КРИТИЧНО: fallback механизм для expansion
    FORCE_EXPANSION_STEPS = 500  # принудительный expansion после N шагов без expansion
    FALLBACK_EXPANSION_THRESHOLD = 0.40  # если accuracy < 40% и loss > 2.0, расширяемся без CLIP
    fallback_expansion_attempted = False

    SLEEP_TRIGGER_STEPS = 500
    SLEEP_TRIGGER_ERRORS = 100
    error_count_phase2 = 0
    last_sleep_step = -1000

    # Warmup после expansion (эмоциональный разогрев)
    WARMUP_STEPS = 200  # первые N шагов с повышенной пластичностью
    WARMUP_DECAY_STEPS = 300  # плавный возврат к нормальному режиму

    optimizer_phase2 = None
    scheduler_phase2 = None
    steps_per_epoch_B = len(loader_B)
    total_steps_phase2 = steps_per_epoch_B * 8

    # Helper to build phase2 optimizer (head + optionally late backbone with groups)
    def build_phase2_optimizer(new_head: nn.Module):
        head_params = list(new_head.parameters())

        early, mid, late = [], [], []
        for name, p in agent.shared_backbone.named_parameters():
            if not p.requires_grad:
                continue
            g = agent._layer_group(name)
            if g == "early":
                early.append(p)
            elif g == "mid":
                mid.append(p)
            else:
                late.append(p)

        # базовые LR (до scaling кристаллом)
        base_lr_head = 2e-3
        base_lr_early = 1e-4
        base_lr_mid = 2e-4
        base_lr_late = 5e-4

        param_groups = [
            {"params": head_params, "lr": base_lr_head, "tag": "head"},
        ]
        if train_late_backbone:
            if len(late):
                param_groups.append({"params": late, "lr": base_lr_late, "tag": "late"})
            if len(mid):
                param_groups.append({"params": mid, "lr": base_lr_mid, "tag": "mid"})
            if len(early):
                param_groups.append({"params": early, "lr": base_lr_early, "tag": "early"})
        
        # КРИТИЧНО: добавляем routing gate в optimizer если используется soft routing
        # Gate создаётся один раз на MAX_LAYERS, поэтому всегда есть
        if agent.use_soft_routing:
            gate_params = list(agent.routing_gate.parameters())
            param_groups.append({"params": gate_params, "lr": 1e-3, "tag": "routing_gate"})

        opt = optim.AdamW(param_groups, weight_decay=1e-4)
        
        # Логирование для диагностики
        print(f"[OPTIMIZER] Phase2 optimizer created with {len(param_groups)} param groups:")
        for pg in param_groups:
            tag = pg.get("tag", "unknown")
            n_params = sum(p.numel() for p in pg["params"])
            print(f"  - {tag}: {n_params} params, base_lr={pg['lr']:.6f}")
        
        return opt

    for epoch in range(8):
        agent.train()
        for data, target in loader_B:
            # Подмешивание dreams (20% батча) для "сна" с distillation
            W_dream = 0.2
            B = data.size(0)
            dream_B = int(B * W_dream)
            real_B = B - dream_B

            data_real = data[:real_B]
            target_real = target[:real_B]
            
            # КРИТИЧНО: перемещаем data_real и target_real на device сразу после создания
            # чтобы избежать проблем с device mismatch в pain-блоке
            data_real = data_real.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target_real = target_real.to(device, non_blocking=True)

            dreams = None
            if dream_B > 0 and use_vae_dreams and agent.vae_trained:
                dreams = agent.sample_dreams(dream_B, device=device)
                # Убеждаемся что dreams на правильном устройстве и формате
                dreams = dreams.to(device=device, non_blocking=True)
                dreams = dreams.to(memory_format=torch.channels_last)
                
                data_mix = torch.cat([data_real, dreams], dim=0)
            else:
                data_mix = data_real

            # КРИТИЧНО: инициализируем pain_value до использования в Complexity Controller
            pain_value = 0.0  # значение по умолчанию, будет обновлено позже если нужно
            
            # 1) shock check (no grad) - используем только real данные
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                test_out = agent(data_real)
                test_loss = criterion_train(test_out[:, :10], target_real)
                
                # КРИТИЧНО: обновляем сенсор test_loss сразу после вычисления (для соответствия shock check)
                agent.sensor.update(float(test_loss.item()))
                
                # Вычисляем метрики для формализованного контроллера expansion
                test_pred = test_out[:, :10].argmax(dim=1)
                test_acc = float((test_pred == target_real).float().mean().item())
                
                # КРИТИЧНО: вычисляем энтропию только по активным классам (маскируем неактивные)
                logits_test = test_out[:, :10].clone()
                # Маскируем неактивные классы (если expansion_count > 0, активны только classes_B)
                if expansion_count > 0:
                    mask_inactive = torch.ones(10, device=logits_test.device, dtype=torch.bool)
                    mask_inactive[torch.tensor(classes_B, device=logits_test.device)] = False
                    logits_test[:, mask_inactive] = -float("inf")
                
                probs_test = torch.softmax(logits_test, dim=1)
                # КРИТИЧНО: per-sample энтропия (не средняя!) для unknown_rate
                entropy_vec = -torch.sum(probs_test * torch.log(probs_test + 1e-9), dim=1)  # [B]
                entropy_test = entropy_vec.mean().item()  # средняя для контроллера
                max_prob_test, _ = probs_test.max(dim=1)
                # КРИТИЧНО: сравниваем per-sample энтропию, а не скаляр
                unknown_rate_test = float(((max_prob_test < 0.18) | (entropy_vec > 1.95)).float().mean().item())
                
                # Conflict rate (из буфера конфликтов)
                conflict_stats = agent.get_conflict_statistics()
                conflict_rate = conflict_stats["total_conflicts"] / max(1, agent.max_conflicts) if conflict_stats else 0.0

            # КРИТИЧНО: рекалибруем сенсор после накопления истории (локальный счётчик Phase2)
            if not sensor_recalibrated and phase2_steps >= 50:
                agent.sensor.calibrate()
                sensor_recalibrated = True
                print(f"[SENSOR] Phase2 baseline set. Mean={agent.sensor.mean:.3f}, Std={agent.sensor.std:.3f}")

            # КРИТИЧНО: формализованный контроллер expansion вместо эвристик
            is_shock = agent.sensor.is_shock(float(test_loss.item()))
            can_expand = (step - last_expansion_step) > COOLDOWN_STEPS
            has_budget = len(agent.heads) < MAX_LAYERS
            
            # КРИТИЧНО: growth_budget растёт без decay (простое накопление)
            agent.growth_budget = min(1.0, agent.growth_budget + 0.001)
            has_growth_budget = agent.growth_budget >= agent.growth_cost_per_expansion
            
            # КРИТИЧНО: для рекурсивной эмергенции - новые концепты будут храниться здесь
            expansion_new_classes = None  # будет установлено если CLIP обнаружит новые концепты
            
            # Формализованный контроллер: need_expand_score = w1*shock + w2*high_entropy + w3*unknown_rate + w4*conflict_rate
            # КРИТИЧНО: unknown_rate теперь имеет больший вес, т.к. неизвестное должно триггерить expansion
            w1, w2, w3, w4 = 0.3, 0.2, 0.3, 0.2  # веса сигналов (unknown_rate увеличен с 0.2 до 0.3)
            shock_signal = 1.0 if is_shock else 0.0
            high_entropy_signal = min(1.0, max(0.0, (entropy_test - 1.0) / 1.0))  # нормализуем [0..1]
            unknown_rate_signal = unknown_rate_test  # уже [0..1] - высокий unknown_rate = нужен новый head
            conflict_rate_signal = conflict_rate  # уже [0..1]
            
            need_expand_score = (
                w1 * shock_signal +
                w2 * high_entropy_signal +
                w3 * unknown_rate_signal +
                w4 * conflict_rate_signal
            )
            EXPANSION_THRESHOLD = 0.5  # порог для expansion
            
            # Fallback условия (для обратной совместимости)
            force_expansion = (expansion_count == 0 and step >= FORCE_EXPANSION_STEPS and has_budget)
            fallback_expansion = (
                not fallback_expansion_attempted and 
                can_expand and 
                has_budget and 
                test_acc < FALLBACK_EXPANSION_THRESHOLD and 
                float(test_loss.item()) > 2.0
            )

            # КРИТИЧНО: формализованный контроллер expansion
            # Используем Complexity Controller для разрешения expansion
            should_expand = False
            expansion_reason = ""
            detected_classes = set()
            
            if force_expansion:
                should_expand = True
                expansion_reason = f"FORCED (no expansion after {FORCE_EXPANSION_STEPS} steps)"
                print(f"\n[FORCE EXPANSION] {expansion_reason}")
            elif fallback_expansion:
                should_expand = True
                expansion_reason = f"FALLBACK (acc={test_acc*100:.1f}% < {FALLBACK_EXPANSION_THRESHOLD*100:.0f}%, loss={float(test_loss.item()):.2f} > 2.0)"
                fallback_expansion_attempted = True
                print(f"\n[FALLBACK EXPANSION] {expansion_reason}")
                print(f"[SAFETY] Budget OK ({len(agent.heads)}/{MAX_LAYERS} heads)")
            elif len(agent.heads) > 0 and actions is not None and actions["expand_allowed"]:
                # КРИТИЧНО: Complexity Controller разрешает expansion (работает даже после sleep)
                should_expand = True
                expansion_reason = f"COMPLEXITY CONTROLLER (C={complexity:.3f} > 0.7, budget={agent.complexity_controller.complexity_budget:.3f})"
                print(f"\n[COMPLEXITY EXPANSION] {expansion_reason}")
            elif need_expand_score > EXPANSION_THRESHOLD and can_expand and has_budget and has_growth_budget:
                # Формализованный контроллер сработал
                should_expand = True
                expansion_reason = f"CONTROLLER (score={need_expand_score:.3f} > {EXPANSION_THRESHOLD:.2f}, signals: shock={shock_signal:.2f}, entropy={high_entropy_signal:.2f}, unknown={unknown_rate_signal:.2f}, conflict={conflict_rate_signal:.2f})"
                print(f"\n[EXPANSION CONTROLLER] {expansion_reason}")
                print(f"[GROWTH BUDGET] {agent.growth_budget:.3f} >= {agent.growth_cost_per_expansion:.3f}")
            elif is_shock and can_expand and has_budget:
                print(f"\n[VISUAL CORTEX SHOCK] Loss {float(test_loss.item()):.2f} detected (High Surprise).")
                print(f"[SAFETY] Cooldown OK, Budget OK ({len(agent.heads)}/{MAX_LAYERS} heads)")

                if agent.use_curiosity:
                    print("[CURIOSITY] Querying Oracle (CLIP)...")
                    # CLIP Diversity Check: требуем минимум 3 разных концепта
                    # КРИТИЧНО: используем data_real (уже на GPU и channels_last) вместо data (CPU)
                    is_diverse, detected_classes, diversity_info, new_concepts = check_clip_diversity(
                        agent, data_real, classes_B, 
                        min_diversity=CLIP_MIN_DIVERSITY, 
                        confidence_threshold=CLIP_TRUST_THRESHOLD
                    )
                    
                    # КРИТИЧНО: если CLIP обнаружил новые концепты (не в текущих классах), это триггер для expansion
                    # КРИТИЧНО: для рекурсивной эмергенции неизвестное выносится в новые heads
                    if new_concepts and len(new_concepts) >= 2:  # минимум 2 новых концепта для expansion
                        print(f"[NEW CONCEPTS] CLIP detected {len(new_concepts)} new concepts: {sorted(new_concepts)}")
                        print(f"[RECURSIVE EMERGENCE] New concepts trigger expansion for structural growth")
                        should_expand = True
                        expansion_reason = f"NEW CONCEPTS DETECTED ({len(new_concepts)} concepts: {sorted(new_concepts)})"
                        # КРИТИЧНО: сохраняем новые концепты для создания нового head
                        expansion_new_classes = list(new_concepts)  # новые классы для expansion
                        print(f"[EXPANSION] Will create new head for classes: {expansion_new_classes}")
                    else:
                        expansion_new_classes = None
                    
                    if not is_diverse and not should_expand:
                        print(f"[DIVERSITY CHECK] FAILED: {diversity_info}")
                        print(f"[DIVERSITY CHECK] Need at least {CLIP_MIN_DIVERSITY} diverse concepts.")
                        # КРИТИЧНО: если CLIP не сработал, но условия fallback выполнены - расширяемся
                        if test_acc < FALLBACK_EXPANSION_THRESHOLD and float(test_loss.item()) > 2.0:
                            should_expand = True
                            expansion_reason = f"FALLBACK after CLIP failure (acc={test_acc*100:.1f}%, loss={float(test_loss.item()):.2f})"
                            fallback_expansion_attempted = True
                            print(f"[FALLBACK EXPANSION] {expansion_reason}")
                        else:
                            print(f"[DIVERSITY CHECK] Skipping expansion (but continuing training).")
                    elif is_diverse and not should_expand:
                        print(f"[DIVERSITY CHECK] PASSED: {diversity_info}")
                        should_expand = True
                        expansion_reason = f"SHOCK + CLIP diversity ({len(detected_classes)} concepts)"
                else:
                    # Если CLIP недоступен, используем fallback
                    if test_acc < FALLBACK_EXPANSION_THRESHOLD and float(test_loss.item()) > 2.0:
                        should_expand = True
                        expansion_reason = f"FALLBACK (CLIP unavailable, acc={test_acc*100:.1f}%, loss={float(test_loss.item()):.2f})"
                        fallback_expansion_attempted = True
                        print(f"[FALLBACK EXPANSION] {expansion_reason}")
            
            if should_expand and has_budget:
                # Берем первый уверенный ответ для логирования (если CLIP доступен)
                best_idx, best_label, conf = None, "unknown", 0.0
                if agent.use_curiosity and len(detected_classes) > 0:
                    best_idx, best_label, conf = agent.curiosity.what_is_this(data_real[0:1])
                    if best_idx is not None:
                        print(f"[EUREKA] CLIP confident ({conf*100:.1f}%): '{best_label}' (one of {len(detected_classes)} detected)")
                
                print(f"[ADAPTATION] Triggering Phase Transition: {expansion_reason}...")

                with torch.no_grad():
                    agent.eval()
                    mo = agent(data_real[0:1])
                    agent.train()
                    mp = torch.softmax(mo[:, :10], dim=1)
                    model_conf, model_pred = mp.max(dim=1)
                    model_entropy = float((-torch.sum(mp * torch.log(mp + 1e-9), dim=1)).item())
                    print(f"[LOG] Model confidence: {float(model_conf.item()):.3f}, Entropy: {model_entropy:.3f}")

                    if best_idx is not None:
                        agent.record_conflict(
                            confidence_model=float(model_conf.item()),
                            entropy_model=model_entropy,
                            clip_class=best_idx,
                            clip_label=best_label,
                            clip_conf=conf,
                            image=data_real[0:1],
                            true_label=int(target_real[0].item()) if target_real.numel() > 0 else None,
                        )

                # КРИТИЧНО: для рекурсивной эмергенции используем новые концепты если они обнаружены
                # Иначе используем оригинальные classes_B
                expansion_classes = expansion_new_classes if expansion_new_classes is not None else classes_B
                
                new_head = agent.expand(
                    new_classes_indices=expansion_classes,
                    use_fractal_time=use_fractal_time,
                    train_late_backbone=train_late_backbone,
                )
                # КРИТИЧНО: применяем fractal-freeze после expand()
                agent.freeze_past(use_fractal_time=use_fractal_time, train_late_backbone=train_late_backbone)
                agent.recalibrate_bn(loader_B, device, num_batches=20)

                optimizer_phase2 = build_phase2_optimizer(new_head)
                
                # Сохраняем lr_base для каждого param group (для LR scaling кристаллом)
                for pg in optimizer_phase2.param_groups:
                    pg["lr_base"] = pg["lr"]
                    pg["scheduler_factor"] = 1.0

                steps_already_done = 0
                remaining_steps = max(total_steps_phase2 - steps_already_done, steps_per_epoch_B)
                scheduler_phase2 = CosineAnnealingLR(optimizer_phase2, T_max=remaining_steps, eta_min=1e-5)

                expansion_count += 1
                last_expansion_step = step
                # Тратим growth_budget
                agent.growth_budget = max(0.0, agent.growth_budget - agent.growth_cost_per_expansion)
                # КРИТИЧНО: НЕ пересоздаём routing gate (он уже создан на MAX_LAYERS)
                # Сохраняем фактическое разнообразие (если было обнаружено CLIP)
                if len(detected_classes) > 0:
                    clip_diversity_at_expansion = len(detected_classes)
                else:
                    clip_diversity_at_expansion = len(classes_B)  # fallback: используем все целевые классы
                # ref_backbone уже создан в agent.expand()
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

            # КРИТИЧНО: Complexity Controller - предварительное вычисление для первого прохода
            # Используем приближение surprise из предыдущего шага (или entropy_test)
            # КРИТИЧНО: работаем даже после sleep (когда expansion_count может быть 0, но есть heads)
            complexity = 0.0
            actions = None
            # Complexity Controller работает если есть хотя бы один head (включая после sleep)
            if len(agent.heads) > 0:
                # Для первого прохода используем приближение: surprise ≈ 0.5 * entropy_test
                surp_approx = 0.5 * entropy_test if entropy_test > 0 else 0.0
                complexity = agent.complexity_controller.compute_complexity(
                    surprise=surp_approx,
                    pain=pain_value,  # будет обновлён позже в pain-блоке
                    entropy=entropy_test,
                    unknown_rate=unknown_rate_test
                )
                
                # Получаем действия от контроллера
                has_expansion_budget = agent.growth_budget >= agent.growth_cost_per_expansion
                actions = agent.complexity_controller.get_actions(
                    complexity=complexity,
                    has_expansion_budget=has_expansion_budget,
                    cooldown_ok=can_expand
                )
                
                # КРИТИЧНО: применяем gate_temperature к routing gate
                if agent.use_soft_routing and agent.routing_gate is not None:
                    agent.routing_gate.set_temperature(actions["gate_temperature"])
            
            # КРИТИЧНО: Memory Scheduler - управление replay через сложность
            if actions is not None:
                replay_batch_size = int(64 * actions["replay_ratio"] / 0.25)  # масштабируем от базового 64
            else:
                replay_batch_size = 64  # fallback
            
            # sample replay с динамическим размером
            x_replay, y_replay = agent.sample_replay_batch(batch_size=replay_batch_size, device=device)

            # КРИТИЧНО: Внутренняя рекурсия - динамический compute loop
            # Рекурсивные проходы для "думать ещё раз" при высокой сложности
            n_recursions = actions["n_recursions"] if (actions is not None and expansion_count > 0) else 1
            used_recursions = 0
            
            all_outputs = []
            all_features = []
            all_surprises = []
            
            # ---- forward (BF16) с рекурсией ----
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                # Pass0: начальный прогноз
                outputs_pass0, features_pass0 = agent(data_mix, return_features=True)
                all_outputs.append(outputs_pass0)
                all_features.append(features_pass0)
                used_recursions += 1
                
                # Вычисляем surprise для Pass0 (для остановки рекурсии)
                surprise_pass0 = None
                if use_subjective_time and agent.ref_backbone is not None:
                    pred_ps_pass0 = agent.critic(features_pass0[:real_B].detach())
                    real_ps_pass0 = criterion_none(outputs_pass0[:real_B, :10], target_real)
                    surprise_pass0 = SubjectiveTimeCritic.surprise(pred_ps_pass0, real_ps_pass0)
                    all_surprises.append(float(surprise_pass0.item()))
                else:
                    all_surprises.append(0.0)
                
                # Рекурсивные проходы (если сложность высокая и budget позволяет)
                for pass_idx in range(1, n_recursions):
                    # Проверяем: нужно ли продолжать (surprise падает?)
                    if len(all_surprises) >= 2:
                        delta_surprise = all_surprises[-2] - all_surprises[-1]
                        if delta_surprise < 0.01:  # surprise не падает - останавливаемся
                            break
                    
                    # Проверяем budget
                    if agent.complexity_controller.complexity_budget < agent.complexity_controller.cost_recursion:
                        break
                    
                    # PassN: повторный прогноз с дополнительным replay
                    # КРИТИЧНО: используем тот же data_mix для консистентности размеров
                    # (replay уже добавлен в data_mix через Memory Scheduler)
                    data_mix_recursive = data_mix  # используем тот же батч для всех проходов
                    
                    outputs_pass, features_pass = agent(data_mix_recursive, return_features=True)
                    all_outputs.append(outputs_pass)
                    all_features.append(features_pass)
                    
                    # Вычисляем surprise для этого прохода
                    # КРИТИЧНО: используем только real_B для surprise (первые real_B элементов)
                    surprise_pass = None
                    if use_subjective_time and agent.ref_backbone is not None:
                        # Берём только real данные для surprise (первые real_B элементов)
                        if outputs_pass.size(0) >= real_B:
                            pred_ps_pass = agent.critic(features_pass[:real_B].detach())
                            real_ps_pass = criterion_none(outputs_pass[:real_B, :10], target_real)
                            surprise_pass = SubjectiveTimeCritic.surprise(pred_ps_pass, real_ps_pass)
                            all_surprises.append(float(surprise_pass.item()))
                        else:
                            # Fallback: если батч меньше real_B, используем весь батч
                            n_available = outputs_pass.size(0)
                            pred_ps_pass = agent.critic(features_pass[:n_available].detach())
                            real_ps_pass = criterion_none(outputs_pass[:n_available, :10], target_real[:n_available])
                            surprise_pass = SubjectiveTimeCritic.surprise(pred_ps_pass, real_ps_pass)
                            all_surprises.append(float(surprise_pass.item()))
                    else:
                        all_surprises.append(0.0)
                    
                    used_recursions += 1
                
                # КРИТИЧНО: усредняем предсказания от всех проходов (или берём последний)
                if len(all_outputs) > 1:
                    # Взвешенное усреднение (последний проход важнее)
                    weights = torch.linspace(0.5, 1.0, len(all_outputs), device=device)
                    weights = weights / weights.sum()
                    outputs = sum(w * out for w, out in zip(weights, all_outputs))
                    features = all_features[-1]  # берём features от последнего прохода
                else:
                    outputs = all_outputs[0]
                    features = all_features[0]
                
                # Используем surprise от последнего прохода для дальнейших вычислений
                if len(all_surprises) > 0 and surprise_pass0 is not None:
                    surprise = surprise_pass0
                else:
                    # Fallback: вычисляем surprise из финальных outputs
                    if use_subjective_time and agent.ref_backbone is not None:
                        pred_ps = agent.critic(features[:real_B].detach())
                        real_ps = criterion_none(outputs[:real_B, :10], target_real)
                        surprise = SubjectiveTimeCritic.surprise(pred_ps, real_ps)
                    else:
                        surprise = None
                
                # КРИТИЧНО: обновляем complexity после вычисления surprise (для следующего шага)
                # Работаем даже после sleep (когда expansion_count может быть 0, но есть heads)
                if len(agent.heads) > 0 and surprise is not None:
                    surp_val = float(surprise.item())
                    complexity = agent.complexity_controller.compute_complexity(
                        surprise=surp_val,
                        pain=pain_value,  # будет обновлён позже в pain-блоке
                        entropy=entropy_test,
                        unknown_rate=unknown_rate_test
                    )
                    # Обновляем actions с правильной complexity (для следующего шага)
                    has_expansion_budget = agent.growth_budget >= agent.growth_cost_per_expansion
                    actions = agent.complexity_controller.get_actions(
                        complexity=complexity,
                        has_expansion_budget=has_expansion_budget,
                        cooldown_ok=can_expand
                    )
                
                # КРИТИЧНО: Entropy penalty для стабилизации routing gates (если используется soft routing)
                routing_entropy_loss = torch.zeros((), device=device, dtype=torch.float32)
                if agent.use_soft_routing and len(agent.heads) > 1:
                    # Вычисляем gates_full и берём срез (как в forward)
                    gates_full = agent.routing_gate(features[:real_B])  # [real_B, MAX_LAYERS]
                    gates = gates_full[:, :len(agent.heads)]  # [B, H]
                    gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-9)  # нормализуем
                    # Entropy penalty: поощряем равномерное распределение ответственности
                    gate_entropy = -torch.sum(gates * torch.log(gates + 1e-9), dim=1).mean()
                    # Целевая entropy (максимальная для равномерного распределения)
                    target_entropy = torch.log(torch.tensor(len(agent.heads), dtype=torch.float32, device=gates.device))
                    # Penalty если entropy слишком низкая (коллапс к одному head)
                    routing_entropy_loss = F.relu(target_entropy * 0.7 - gate_entropy)  # penalty если entropy < 70% от максимума
                
                # Проверка outputs на inf/nan
                if not torch.isfinite(outputs).all():
                    print(f"[ERROR] outputs contains inf/nan at step {step}")
                    # Попытка исправить: заменить inf/nan на 0
                    outputs = torch.where(torch.isfinite(outputs), outputs, torch.zeros_like(outputs))
                    if not torch.isfinite(outputs).all():
                        print(f"[ERROR] Cannot fix outputs, skipping step {step}")
                        continue

                # Supervised loss на real данных
                # Используем class-balanced loss если веса вычислены (только в Phase2 после expansion)
                if expansion_count > 0 and class_weights_phase2 is not None:
                    # КРИТИЧНО: проверяем только веса активных классов (неактивные = 0)
                    active_w = class_weights_phase2[torch.tensor(classes_B, device=device)]
                    if torch.isfinite(active_w).all() and (active_w > 0).all():
                        loss_new = class_balanced_loss(outputs[:real_B, :10], target_real, class_weights_phase2, num_classes=10)
                    else:
                        # Fallback на обычный loss если веса проблемные
                        loss_new = criterion_train(outputs[:real_B, :10], target_real)
                else:
                    # Phase1 или если веса не вычислены - используем обычный loss
                    loss_new = criterion_train(outputs[:real_B, :10], target_real)
                
                # КРИТИЧНО: Outlier Exposure ОТКЛЮЧЕН для рекурсивной эмергенции
                # Неизвестное должно триггерить expansion новых heads, а не обучаться как отдельный класс.
                # Система будет расширяться структурно при встрече с новыми концептами через:
                # 1. unknown_rate в Complexity Controller (триггерит expansion)
                # 2. CLIP обнаружение новых концептов (триггерит expansion)
                # 3. High entropy + shock (триггерит expansion)
                loss_unknown = torch.zeros((), device=device, dtype=torch.float32)
                agent.unknown_trained = True  # помечаем что unknown не обучается как класс, а триггерит expansion
                # Outlier Exposure код удалён - неизвестное выносится в новые heads через expansion
                
                # Distillation loss на dreams (сон с удержанием структуры)
                loss_dream = 0.0
                if dreams is not None and dream_B > 0 and teacher is not None:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                        t_out = teacher(dreams)[:, :10]
                        t_p = torch.softmax(t_out / 2.0, dim=1)  # T=2.0 для мягких вероятностей

                    s_out = outputs[real_B:, :10]
                    s_logp = torch.log_softmax(s_out / 2.0, dim=1)  # T=2.0

                    # KL(student || teacher) - студент должен следовать учителю
                    loss_dream = F.kl_div(s_logp, t_p, reduction="batchmean") * (2.0 * 2.0)  # T^2 для масштабирования
                
                # Усиленная pair margin loss для cat/dog в Phase2 (животные) - только на real данных
                if expansion_count > 0:
                    pm_catdog = pair_margin_loss(outputs[:real_B, :10], target_real, pairs=((3, 5),), margin=0.20)
                    loss_new = loss_new + 0.12 * pm_catdog
                    # Остальные пары (plane/ship, car/truck) слабее
                    pm_other = pair_margin_loss(outputs[:real_B, :10], target_real, pairs=((0, 8), (1, 9)), margin=0.15)
                    loss_new = loss_new + 0.05 * pm_other

                # replay loss (только на real данных для consistency)
                replay_loss = 0.0
                if x_replay is not None:
                    out_rep = agent(x_replay)
                    replay_loss = criterion_train(out_rep[:, :10], y_replay)

                # ---- subjective time critic (per-sample, inside autocast) ----
                # КРИТИЧНО: surprise уже вычислен в рекурсивном цикле, но обновляем critic_loss
                critic_loss = None
                if use_subjective_time and agent.ref_backbone is not None and surprise is not None:
                    # Используем surprise от последнего прохода, но обновляем critic
                    pred_ps = agent.critic(features[:real_B].detach())          # [real_B]
                    real_ps = criterion_none(outputs[:real_B, :10], target_real)  # [real_B]
                    critic_loss = F.mse_loss(pred_ps, real_ps.detach())

            # entropy (no grad) - только на real данных
            # КРИТИЧНО: вычисляем энтропию только по активным классам (маскируем неактивные)
            with torch.no_grad():
                logits_m = outputs[:real_B, :10].clone()
                if expansion_count > 0:
                    # Маскируем неактивные классы для Phase2
                    mask_inactive = torch.ones(10, device=logits_m.device, dtype=torch.bool)
                    mask_inactive[torch.tensor(classes_B, device=logits_m.device)] = False
                    logits_m[:, mask_inactive] = -float("inf")
                
                probs_m = torch.softmax(logits_m, dim=1)
                ent_batch = (-(probs_m * torch.log(probs_m + 1e-9)).sum(dim=1)).mean().item()

            # ---- adaptive pain (MUST be computed BEFORE crystallization update) ----
            # КРИТИЧНО: pain_value уже инициализирован выше, обновляем только если нужно
            adaptive_lambda = None
            if use_adaptive_pain and (x_replay is not None) and len(agent.heads) > 0:
                backbone_params = [p for p in agent.shared_backbone.parameters() if p.requires_grad]
                if len(backbone_params) > 0:
                    # КРИТИЧНО: замораживаем BN обновление во время pain-градиентов
                    # чтобы не портить статистики BN дополнительными forward проходами
                    was_training = agent.training
                    agent.train()  # градиенты нужны
                    agent._set_bn_train(False)  # BN в eval режиме (не обновляет running stats)
                    
                    # fp32 grads (используем полный forward для надежности)
                    try:
                        with torch.amp.autocast("cuda", enabled=False):
                            # КРИТИЧНО: используем data_real и target_real (уже на device)
                            # вместо data и target (которые могут быть на CPU)
                            out_new_fp32 = agent(data_real.float())
                            ln = criterion_train(out_new_fp32[:, :10], target_real)

                            out_old_fp32 = agent(x_replay.float())
                            lo = criterion_train(out_old_fp32[:, :10], y_replay)
                            
                            # Проверка на NaN/inf
                            if not torch.isfinite(ln) or not torch.isfinite(lo):
                                raise ValueError("Loss contains NaN/inf")
                    except Exception as e:
                        if step % 50 == 0:
                            print(f"[WARNING] Pain computation failed: {e}, skipping pain")
                        ln = None
                        lo = None
                    finally:
                        # Восстанавливаем BN режим
                        agent._set_bn_train(True)
                        if not was_training:
                            agent.eval()

                    if ln is not None and lo is not None and torch.isfinite(ln) and torch.isfinite(lo):
                        try:
                            g_new = torch.autograd.grad(ln, backbone_params, retain_graph=True, allow_unused=True, create_graph=False)
                            g_old = torch.autograd.grad(lo, backbone_params, retain_graph=True, allow_unused=True, create_graph=False)
                            g_new = [g for g in g_new if g is not None and torch.isfinite(g).all()]
                            g_old = [g for g in g_old if g is not None and torch.isfinite(g).all()]

                            if len(g_new) and len(g_old):
                                gn = torch.cat([g.detach().flatten() for g in g_new])
                                go = torch.cat([g.detach().flatten() for g in g_old])
                                # Проверка на inf/nan
                                if torch.isfinite(gn).all() and torch.isfinite(go).all():
                                    dot = float(torch.dot(gn, go).item())
                                    n1 = float(gn.pow(2).sum().sqrt().item()) + 1e-8
                                    n2 = float(go.pow(2).sum().sqrt().item()) + 1e-8
                                    if n1 > 0 and n2 > 0 and torch.isfinite(torch.tensor([dot, n1, n2])).all():
                                        cos = dot / (n1 * n2)
                                        pain_value = max(0.0, min(1.0, (1.0 - cos) * 0.5))
                                        adaptive_lambda = 100.0 + (20000.0 - 100.0) * pain_value
                        except Exception as e:
                            if step % 50 == 0:
                                print(f"[WARNING] Gradient computation failed: {e}")

            # ---- CLIP KL (unchanged, but computed AFTER forward) ----
            # Используем только real данные для CLIP KL
            kl_loss = 0.0
            if agent.use_curiosity:
                # КРИТИЧНО: вычисляем энтропию только по активным классам (маскируем неактивные)
                logits_model = outputs[:real_B, :10].clone()
                if expansion_count > 0:
                    # Маскируем неактивные классы для Phase2
                    mask_inactive = torch.ones(10, device=logits_model.device, dtype=torch.bool)
                    mask_inactive[torch.tensor(classes_B, device=logits_model.device)] = False
                    logits_model[:, mask_inactive] = -float("inf")
                
                probs_model = torch.softmax(logits_model, dim=1)
                ent = -torch.sum(probs_model * torch.log(probs_model + 1e-9), dim=1)  # [real_B]
                hi = ent > 1.5  # порог для "высокой энтропии" (неуверенность модели)
                if hi.any():
                    idx = torch.where(hi)[0]
                    MAX_UNCERTAIN = 16
                    if idx.numel() > MAX_UNCERTAIN:
                        idx = idx[:MAX_UNCERTAIN]
                    clip_targets = agent.get_clip_soft_targets(data_real[idx])
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
                            if expansion_count > 0:
                                steps_since_expand = step - last_expansion_step
                                kl_weight = min(0.1, 0.1 * (steps_since_expand / 500.0))
                            else:
                                kl_weight = 0.1
                            kl_loss = kl_weight * kl
                            if step % 50 == 0:
                                # КРИТИЧНО: это количество сэмплов с высокой энтропией, а не сама энтропия
                                print(f"[LOG] High entropy samples: {int(idx2.numel())}, KL: {float(kl_loss.item()):.4f}")

            # ---- UPDATE TIME CRYSTALLIZATION (NOW sees pain_value correctly) ----
            # КРИТИЧНО: вычисляем warmup множители ОДИН РАЗ перед использованием
            lr_warmup_mult_backbone = 1.0
            cryst_strength_warmup_mult = 1.0
            if expansion_count > 0:
                steps_since_expand = step - last_expansion_step
                if steps_since_expand < WARMUP_STEPS:
                    lr_warmup_mult_backbone = 1.5
                    cryst_strength_warmup_mult = 0.0
                elif steps_since_expand < (WARMUP_STEPS + WARMUP_DECAY_STEPS):
                    decay_progress = (steps_since_expand - WARMUP_STEPS) / WARMUP_DECAY_STEPS
                    lr_warmup_mult_backbone = 1.5 - decay_progress * 0.5  # 1.5 -> 1.0
                    cryst_strength_warmup_mult = 0.0 + decay_progress * 1.0  # 0.0 -> 1.0
                else:
                    lr_warmup_mult_backbone = 1.0
                    cryst_strength_warmup_mult = 1.0
            
            # КРИТИЧНО: Time Crystallization работает даже после sleep (если есть ref_backbone)
            if len(agent.heads) > 0 and use_subjective_time and agent.ref_backbone is not None:
                surp_val = float(surprise.item()) if surprise is not None else 0.0
                agent.update_time_crystallization(surp_val, pain_value, ent_batch)
                
                # КРИТИЧНО: Complexity Controller управляет crystal_target
                if actions is not None:
                    # Плавно подводим crystal_level к целевому значению от контроллера
                    crystal_target = actions["crystal_target"]
                    agent.crystal_level = 0.95 * agent.crystal_level + 0.05 * crystal_target
                    agent.crystal_level = float(max(0.0, min(1.0, agent.crystal_level)))
                
                # Корректировка crystal_level на основе разнообразия CLIP
                # Используем фактическое разнообразие относительно максимального возможного
                if clip_diversity_at_expansion > 0:
                    # diversity_ratio: 0..1 (сколько классов из целевых обнаружено)
                    max_possible = len(classes_B)  # максимальное возможное разнообразие
                    diversity_ratio = clip_diversity_at_expansion / max(1, max_possible)
                    # При низком разнообразии снижаем crystal_level (больше пластичности)
                    # При высоком - оставляем как есть или немного повышаем
                    agent.crystal_level = agent.crystal_level * (0.7 + 0.3 * diversity_ratio)
                    agent.crystal_level = float(max(0.0, min(1.0, agent.crystal_level)))
                
                agent.auto_hard_freeze_if_needed()

            # ---- assemble total loss (single protection mechanism) ----
            total_loss = loss_new
            
            # Добавляем loss для Unknown класса (Outlier Exposure)
            if isinstance(loss_unknown, torch.Tensor) and loss_unknown.item() != 0.0 and torch.isfinite(loss_unknown):
                total_loss = total_loss + 0.1 * loss_unknown  # небольшой вес для стабильности
            
            # Добавляем entropy penalty для routing gates
            if isinstance(routing_entropy_loss, torch.Tensor) and routing_entropy_loss.item() != 0.0 and torch.isfinite(routing_entropy_loss):
                total_loss = total_loss + 0.05 * routing_entropy_loss  # небольшой вес для стабилизации
            
            # Проверка на NaN/inf в loss_new с детальной диагностикой
            if not torch.isfinite(loss_new):
                print(f"[ERROR] loss_new is NaN/inf at step {step}")
                # Детальная диагностика
                with torch.no_grad():
                    print(f"  - outputs min/max: {outputs.min().item():.3f}/{outputs.max().item():.3f}")
                    print(f"  - outputs contains inf: {torch.isinf(outputs).any().item()}")
                    print(f"  - outputs contains nan: {torch.isnan(outputs).any().item()}")
                    print(f"  - target range: {target_real.min().item()} to {target_real.max().item()}")
                    if expansion_count > 0 and class_weights_phase2 is not None:
                        print(f"  - class_weights min/max: {class_weights_phase2.min().item():.3f}/{class_weights_phase2.max().item():.3f}")
                # Пропускаем шаг
                current_opt.zero_grad(set_to_none=True)
                continue

            # КРИТИЧНО: сохраняем компоненты loss для диагностики
            loss_new_val = float(loss_new.item())
            unknown_val = float(loss_unknown.item()) if isinstance(loss_unknown, torch.Tensor) and torch.isfinite(loss_unknown) else 0.0
            
            # Добавляем dream distillation loss (сон с удержанием структуры)
            dream_val = 0.0
            if loss_dream != 0.0 and torch.isfinite(loss_dream):
                dream_val = float(loss_dream.item())
                total_loss = total_loss + 0.2 * loss_dream  # вес для dreams

            replay_val = 0.0
            if x_replay is not None:
                replay_term = 0.25 * replay_loss
                if torch.isfinite(replay_term):
                    replay_val = float(replay_term.item())
                    total_loss = total_loss + replay_term
                else:
                    if step % 50 == 0:
                        print(f"[WARNING] replay_loss is NaN/inf, skipping")

            kl_val = 0.0
            if kl_loss != 0.0 and torch.isfinite(kl_loss):
                if isinstance(kl_loss, torch.Tensor):
                    kl_val = float(kl_loss.item())
                else:
                    kl_val = float(kl_loss)
                total_loss = total_loss + kl_loss
            
            # reg_val будет вычислен ниже в блоке crystallization regularizer

            # crystallization regularizer replaces old current_lambda * stability_loss
            reg_val = 0.0  # инициализация для диагностики
            # КРИТИЧНО: работаем даже после sleep (когда expansion_count может быть 0, но есть heads)
            if len(agent.heads) > 0 and use_subjective_time and agent.ref_backbone is not None:
                cryst_reg = agent.crystallization_regularizer()
                if cryst_reg != 0.0 and torch.isfinite(cryst_reg):
                    # КРИТИЧНО: base_strength уменьшен с 300 до 30 для предотвращения взрыва loss
                    # crystal strength controlled by crystal_level and optionally pain (adaptive_lambda)
                    base_strength = 30.0 * agent.crystal_level

                    # КРИТИЧНО: pain должен УМЕНЬШАТЬ защиту (больше пластичности при конфликте)
                    # pain_value in [0..1], чем выше тем больше конфликт
                    if adaptive_lambda is not None and pain_value > 0:
                        # pain_value растёт с конфликтом, multiplier должен падать
                        # mult = 1.0 -> 0.3 при pain_value = 0 -> 1
                        mult = 1.0 - 0.7 * pain_value
                        base_strength = base_strength * mult
                    
                    # Warmup multiplier (уже вычислен выше)
                    base_strength = base_strength * cryst_strength_warmup_mult
                    reg_term = base_strength * cryst_reg
                    if torch.isfinite(reg_term):
                        reg_val = float(reg_term.item())
                        total_loss = total_loss + reg_term
                    else:
                        if step % 50 == 0:
                            print(f"[WARNING] cryst_reg term is NaN/inf, skipping")
            
            # КРИТИЧНО: обновляем Complexity Budget на основе использованных действий
            # Работаем даже после sleep (когда expansion_count может быть 0, но есть heads)
            if len(agent.heads) > 0 and actions is not None:
                used_expansion = should_expand and has_budget
                used_kl = (kl_loss != 0.0 and torch.isfinite(kl_loss))
                agent.complexity_controller.update_budget(
                    actions=actions,
                    used_expansion=used_expansion,
                    used_kl=used_kl
                )

            # ---- critic update ----
            if use_subjective_time and critic_loss is not None:
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward(retain_graph=True)
                critic_optimizer.step()

            # ---- backward main ----
            # Проверка на NaN/inf перед backward
            if not torch.isfinite(total_loss):
                print(f"[ERROR] Loss is NaN/inf at step {step}, skipping backward")
                continue
            
            total_loss.backward()
            
            # Проверка градиентов на inf/nan перед clipping
            grad_norm_before = 0.0
            for p in agent.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        print(f"[ERROR] Non-finite gradients detected at step {step}, zeroing")
                        p.grad.zero_()
                    else:
                        grad_norm_before += p.grad.data.norm(2).item() ** 2
            grad_norm_before = grad_norm_before ** 0.5
            
            # Более агрессивный gradient clipping
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            
            # Проверка после clipping
            grad_norm_after = 0.0
            for p in agent.parameters():
                if p.grad is not None:
                    grad_norm_after += p.grad.data.norm(2).item() ** 2
            grad_norm_after = grad_norm_after ** 0.5
            
            if not torch.isfinite(torch.tensor([grad_norm_after])):
                print(f"[ERROR] Gradients still non-finite after clipping at step {step}, skipping step")
                current_opt.zero_grad(set_to_none=True)
                continue
            
            current_opt.step()

            # Scheduler step (ВАЖНО: после optimizer.step())
            if scheduler_phase2 is not None:
                scheduler_phase2.step()
                
                # КРИТИЧНО: LR scaling применяется ПОСЛЕ scheduler
                # Используем get_last_lr() чтобы получить "чистое" scheduler значение
                # и применяем наши множители к нему (не итеративно умножаем!)
                if expansion_count > 0:
                    # warmup_mult уже вычислен выше
                    sched_lrs = scheduler_phase2.get_last_lr()
                    scales = agent.time_lr_scale()
                    
                    for pg, sched_lr in zip(optimizer_phase2.param_groups, sched_lrs):
                        tag = pg.get("tag", None)
                        if tag in scales:
                            base_scale = scales[tag]
                            # Warmup multiplier применяется только к backbone (mid/late)
                            if tag in ["mid", "late"]:
                                warmup_mult = lr_warmup_mult_backbone
                            else:
                                warmup_mult = 1.0
                            
                            # Применяем множители к "чистому" scheduler LR
                            new_lr = sched_lr * base_scale * warmup_mult
                            
                            # Проверка на разумность LR перед применением
                            if torch.isfinite(torch.tensor([new_lr])) and 1e-8 < new_lr < 1.0:
                                pg["lr"] = new_lr
                            # Иначе оставляем LR от scheduler
            elif optimizer_phase2 is None:
                scheduler.step()

            # КРИТИЧНО: НЕ обновляем sensor здесь (уже обновлён test_loss выше)
            # Это предотвращает "двойное" обновление и несоответствие между shock check и baseline

            # error counter (только на real данных)
            if expansion_count > 0:
                with torch.no_grad():
                    agent.eval()
                    out = agent(data_real)
                    pred = out[:, :10].argmax(dim=1)
                    error_count_phase2 += int((pred != target_real).sum().item())
                    agent.train()

            if step % 50 == 0:
                acc_A = eval_masked(agent, test_loader_A, classes_A, device, block_unknown=True)
                acc_B = eval_masked(agent, test_loader_B, classes_B, device, block_unknown=True)
                acc_A_hist.append(acc_A)
                acc_B_hist.append(acc_B)

                with torch.no_grad():
                    out = agent(data_real)
                    p = torch.softmax(out[:, :10], dim=1)
                    ent = -torch.sum(p * torch.log(p + 1e-9), dim=1)
                    mp, _ = p.max(dim=1)
                    unk_rate = ((mp < 0.2) | (ent > 1.8)).float().mean().item()
                    
                    # Диагностика "frog collapse": распределение предсказаний по животным
                    preds = out[:, :10].argmax(dim=1)
                    # Фильтруем только целевые классы животных
                    animal_mask = torch.isin(preds, torch.tensor(classes_B, device=preds.device))
                    if animal_mask.any() and animal_mask.sum() > 0:
                        animal_preds = preds[animal_mask]
                        uniq, cnt = animal_preds.unique(return_counts=True)
                        if len(uniq) > 0:
                            top3 = sorted(zip(uniq.cpu().tolist(), cnt.cpu().tolist()), key=lambda x: -x[1])[:3]
                            class_names_short = ["Pl", "Car", "Bd", "Ct", "Dr", "Dg", "Fg", "Hs", "Sh", "Tk"]
                            top3_str = ", ".join([f"{class_names_short[c]}:{cnt}" for c, cnt in top3])
                        else:
                            top3_str = "none"
                    else:
                        top3_str = "none"

                s = f"{float(surprise.item()):.4f}" if surprise is not None else "n/a"
                cryst_info = f" | Crystal: {agent.crystal_level:.3f}" if expansion_count > 0 else ""
                
                # Warmup статус
                warmup_info = ""
                if expansion_count > 0:
                    steps_since_expand = step - last_expansion_step
                    if steps_since_expand < WARMUP_STEPS:
                        warmup_info = f" | WARMUP({steps_since_expand}/{WARMUP_STEPS})"
                    elif steps_since_expand < (WARMUP_STEPS + WARMUP_DECAY_STEPS):
                        warmup_info = f" | WARMUP-DECAY({steps_since_expand - WARMUP_STEPS}/{WARMUP_DECAY_STEPS})"
                
                pred_info = f" | Pred: {top3_str}" if expansion_count > 0 else ""
                
                # КРИТИЧНО: диагностика компонентов loss для выявления источника взрыва
                loss_components = f"Lnew:{loss_new_val:.2f} R:{replay_val:.2f} KL:{kl_val:.3f} D:{dream_val:.2f} U:{unknown_val:.3f} Reg:{reg_val:.2f}"
                
                # КРИТИЧНО: Complexity Controller статус
                # Показываем даже после sleep (когда expansion_count может быть 0, но есть heads)
                complexity_info = ""
                if len(agent.heads) > 0 and actions is not None:
                    budget_status = agent.complexity_controller.get_budget_status()
                    complexity_info = f" | C:{complexity:.3f} R:{used_recursions} B:{budget_status['budget']:.2f} T:{actions['gate_temperature']:.2f}"
                
                print(
                    f"Step {step}: Loss {float(total_loss.item()):.2f} ({loss_components}) | Mem(M): {acc_A:.1f}% | "
                    f"New(A): {acc_B:.1f}% | Heads: {len(agent.heads)} | UnknownRate: {unk_rate*100:.1f}% | "
                    f"Errors: {error_count_phase2} | Surprise: {s}{cryst_info}{warmup_info}{pred_info}{complexity_info}"
                )

            step += 1
            phase2_steps += 1  # КРИТИЧНО: увеличиваем локальный счётчик Phase2

    # ----------------------------
    # Latent space visualization (diagnostic)
    # ----------------------------
    def visualize_latent_space(agent, loader, device, title_suffix=""):
        """
        Визуализация латентного пространства features для диагностики.
        Показывает, насколько хорошо backbone разделяет классы.
        """
        try:
            from sklearn.manifold import TSNE  # type: ignore
            use_tsne = True
        except ImportError:
            try:
                from sklearn.decomposition import PCA  # type: ignore
                use_tsne = False
                print("[VIZ] t-SNE not available, using PCA")
            except ImportError:
                print("[VIZ] sklearn not available, skipping latent visualization")
                print("[VIZ] Install with: pip install scikit-learn")
                return

        agent.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            for data, target in loader:
                data = data.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                _, features = agent(data, return_features=True)
                # КРИТИЧНО: конвертируем BFloat16 в float32 перед numpy (numpy не поддерживает BFloat16)
                features_float = features.float() if features.dtype == torch.bfloat16 else features
                features_list.append(features_float.cpu().numpy())
                labels_list.append(target.numpy())
                if len(features_list) * data.size(0) >= 2000:  # ограничиваем для скорости
                    break
        
        features_all = np.concatenate(features_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)
        
        print(f"[VIZ] Computing {'t-SNE' if use_tsne else 'PCA'} for {len(features_all)} samples...")
        
        if use_tsne:
            # КРИТИЧНО: в новых версиях sklearn параметр называется max_iter вместо n_iter
            try:
                reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            except TypeError:
                # Fallback для старых версий sklearn
                reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        features_2d = reducer.fit_transform(features_all)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i in range(10):
            mask = labels_all == i
            if mask.any():
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[colors[i]], label=class_names[i], alpha=0.6, s=20)
        
        ax.set_title(f"Latent Space Visualization {title_suffix}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        project_root = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(project_root, "test_outputs", f"latent_space_{title_suffix.lower().replace(' ', '_')}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[VIZ] Saved to {out_path}")
        plt.close()

    # Визуализация после Phase2 (если есть expansion)
    if expansion_count > 0:
        print("\n--- VISUALIZING LATENT SPACE (Post Phase2) ---")
        visualize_latent_space(agent, test_loader_B, device, title_suffix="Post Phase2 Animals")

    # ----------------------------
    # Generative Quality Check: Car ⊕ Cat (структурная связность)
    # ----------------------------
    @torch.no_grad()
    def dream_hybrid_check(agent, loader_all, device, cls_a=1, cls_b=3, n_a=64, n_b=64):
        """
        Проверка качества VAE через гибридные образы.
        Делает среднее в латенте VAE (μ) между классами и декодирует.
        Если получается структурный гибрид - VAE научился связному латенту.
        """
        if not (agent.use_vae_dreams and agent.vae_trained):
            print("[HYBRID] VAE not ready, skipping hybrid check")
            return

        agent.dream_vae.eval()

        xa, xb = [], []
        for x, y in loader_all:
            for i in range(x.size(0)):
                if int(y[i]) == cls_a and len(xa) < n_a:
                    xa.append(x[i:i+1])
                if int(y[i]) == cls_b and len(xb) < n_b:
                    xb.append(x[i:i+1])
            if len(xa) >= n_a and len(xb) >= n_b:
                break

        if len(xa) < n_a or len(xb) < n_b:
            print(f"[HYBRID] Not enough samples: Car={len(xa)}, Cat={len(xb)}")
            return

        xa = torch.cat(xa, dim=0).to(device)
        xb = torch.cat(xb, dim=0).to(device)

        mu_a, lv_a = agent.dream_vae.encode(xa)
        mu_b, lv_b = agent.dream_vae.encode(xb)

        # Среднее в латенте (не по пикселям!)
        mu_mix = 0.5 * mu_a.mean(dim=0, keepdim=True) + 0.5 * mu_b.mean(dim=0, keepdim=True)
        x_mix = agent.dream_vae.decode(mu_mix)
        x_mix = torch.clamp(x_mix, -1, 1)

        project_root = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(project_root, "test_outputs", "hybrid_car_cat.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Сохранить как картинку
        try:
            import torchvision
            grid = torchvision.utils.make_grid((x_mix * 0.5 + 0.5).cpu(), nrow=1)
            torchvision.utils.save_image(grid, out_path)
            print(f"[HYBRID] Saved hybrid Car⊕Cat image to {out_path}")
            print(f"[HYBRID] If image shows structural hybrid (not gray noise), VAE learned coherent latent space")
        except ImportError:
            print(f"[HYBRID] torchvision not available, cannot save image")

    # Выполняем проверку качества VAE
    if use_vae_dreams and agent.vae_trained:
        print("\n--- GENERATIVE QUALITY CHECK (Car ⊕ Cat) ---")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(project_root, "data")
        test_full_temp = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        test_loader_all_temp = DataLoader(test_full_temp, batch_size=500, shuffle=False, num_workers=4, pin_memory=True)
        dream_hybrid_check(agent, test_loader_all_temp, device, cls_a=1, cls_b=3, n_a=64, n_b=64)

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
