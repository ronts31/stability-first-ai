import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
import copy
import copy
try:
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[WARNING] CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

# --- 1. АРХИТЕКТУРА (МАСШТАБИРОВАННАЯ) ---

class ComplexitySensor:
    def __init__(self, sensitivity=2.5):
        self.history = []
        self.mean = 0
        self.std = 1
        self.sensitivity = sensitivity
        self.calibrated = False

    def update(self, loss):
        self.history.append(loss)
        if len(self.history) > 500: self.history.pop(0)

    def calibrate(self):
        if len(self.history) > 10:
            self.mean = np.mean(self.history)
            self.std = np.std(self.history) + 1e-6
            self.calibrated = True
            print(f"[SENSOR] Baseline set. Mean={self.mean:.3f}, Std={self.std:.3f}")

    def is_shock(self, loss):
        if not self.calibrated: return False
        z_score = (loss - self.mean) / self.std
        return z_score > self.sensitivity

# --- SUBJECTIVE TIME CRITIC (из проекта 06) ---
class SubjectiveTimeCritic(nn.Module):
    """
    Мета-когнитивная сеть, которая предсказывает loss на основе features.
    Surprise = |Real_Loss - Predicted_Loss|
    High Surprise -> Low Lambda (высокая пластичность)
    Low Surprise -> High Lambda (высокая стабильность)
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Предсказывает scalar Loss
        )
    
    def forward(self, features):
        return self.net(features).squeeze(-1)  # [batch]
    
    def compute_surprise(self, predicted_loss, real_loss):
        """Вычисляет Surprise = |Real - Predicted|"""
        return torch.abs(real_loss.detach() - predicted_loss).mean()
    
    def compute_lambda(self, surprise, base_lambda=10000.0, sensitivity=10.0):
        """Вычисляет динамический Lambda на основе Surprise"""
        return base_lambda / (1.0 + surprise.item() * sensitivity)

# --- VAE для генерации реалистичных снов (из проекта 01, 05) ---
class DreamVAE(nn.Module):
    """
    VAE для генерации "снов" - более реалистичных изображений вместо белого шума.
    Используется в Active Sleep для генеративного replay.
    """
    def __init__(self, z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        
        # Encoder: CIFAR-10 (3, 32, 32) -> z_dim
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
        
        # Decoder: z_dim -> (3, 32, 32)
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
            nn.Tanh()  # [-1, 1] как CIFAR-10
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
    """VAE loss: reconstruction + KL divergence"""
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss

# C) Общий CNN Backbone (используется всеми головами)
class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # A) CNN для CIFAR-10 (3-4 conv слоя + GAP)
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
        # x: [B, 3, 32, 32]
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)  # 32x32 -> 16x16
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2)  # 16x16 -> 8x8
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pool2d(h, 2)  # 8x8 -> 4x4
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.gap(h)  # 4x4 -> 1x1
        h = h.view(h.size(0), -1)  # [B, 512]
        return h

# C) Расширяемая голова (только классификатор, без backbone)
class ExpandableHead(nn.Module):
    def __init__(self, hidden_size, output_size, prev_dims=[]):
        super().__init__()
        # Адаптеры для интеграции прошлых представлений
        self.adapters = nn.ModuleList([nn.Linear(p, hidden_size) for p in prev_dims])
        # Финальный классификатор
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
    
    def forward(self, backbone_features, prev_hiddens):
        # backbone_features: [B, 512]
        h = backbone_features
        
        # Интеграция прошлого
        for i, adapter in enumerate(self.adapters):
            if i < len(prev_hiddens):
                h = h + adapter(prev_hiddens[i])
        
        return self.fc(h), h

# Старая TemporalColumn для обратной совместимости (используется только в dream_and_compress)
class TemporalColumn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, prev_dims=[]):
        super().__init__()
        # Полная колонка (backbone + head) для сжатия
        self.backbone = SharedBackbone()
        self.head = ExpandableHead(hidden_size, output_size, prev_dims)
        self.hidden_size = hidden_size

    def forward(self, x, prev_hiddens):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        backbone_features = self.backbone(x)
        return self.head(backbone_features, prev_hiddens)

# --- МОДУЛЬ ЛЮБОПЫТСТВА (ORACLE) ---
class CuriosityModule:
    def __init__(self):
        if not CLIP_AVAILABLE:
            self.available = False
            return
            
        self.available = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Загружаем CLIP (это "Интернет" в кармане - знает всё)
        print("[CURIOSITY] Loading World Knowledge (CLIP)...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
        # D) Улучшенные промпты для CLIP ("a photo of..." обычно лучше)
        self.cifar10_concepts = [
            "a photo of an airplane", "a photo of a car", "a photo of a bird", 
            "a photo of a cat", "a photo of a deer", "a photo of a dog", 
            "a photo of a frog", "a photo of a horse", "a photo of a ship", 
            "a photo of a truck"
        ]
        
        # CLIP нормализация (ImageNet статистика)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1,3,1,1)
        self.clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1,3,1,1)
        
        # Превращаем слова в векторы смысла
        self.text_inputs = clip.tokenize(self.cifar10_concepts).to(self.device)
        print("[CURIOSITY] CLIP loaded successfully!")

    def _prep_for_clip(self, x):
        """
        A) Правильная подготовка изображения для CLIP
        x: CIFAR normalized with mean=0.5 std=0.5 => [-1..1]
        """
        x = x * 0.5 + 0.5          # -> [0..1]
        x = torch.clamp(x, 0, 1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        return x

    def what_is_this(self, image_tensor, return_probs=False):
        """
        Спрашивает у 'Мирового Разума', что на картинке.
        image_tensor: [1, 3, 32, 32] - CIFAR-10 изображение
        """
        if not self.available:
            return (None, None, 0.0) if not return_probs else (None, None, 0.0, None)

        try:
            image_in = self._prep_for_clip(image_tensor.to(self.device))

            with torch.no_grad():
                logits_per_image, _ = self.model(image_in, self.text_inputs)
                probs = logits_per_image.softmax(dim=-1).squeeze(0)  # torch [10]

            best_idx = int(torch.argmax(probs).item())
            best_label = self.cifar10_concepts[best_idx]
            confidence = float(probs[best_idx].item())

            if return_probs:
                return best_idx, best_label, confidence, probs.detach()  # 6) Убираем .cpu() для скорости
            return best_idx, best_label, confidence

        except Exception as e:
            print(f"[CURIOSITY] Error: {e}")
            return (None, None, 0.0) if not return_probs else (None, None, 0.0, None)

class RecursiveAgent(nn.Module):
    def __init__(self, use_curiosity=False, use_subjective_time=False, use_vae_dreams=False):
        super().__init__()
        # C) Общий backbone + расширяемые головы
        self.hidden_size = 512  # Размер скрытого представления после CNN
        self.output_size = 11  # 10 классов + 1 "unknown/ambiguous"
        
        # C) Общий backbone (один для всех)
        self.shared_backbone = SharedBackbone()
        
        # C) Расширяемые головы (создаются при expand)
        self.heads = nn.ModuleList([ExpandableHead(self.hidden_size, self.output_size)])
        
        # Для обратной совместимости (используется в dream_and_compress)
        self.columns = nn.ModuleList([TemporalColumn(0, self.hidden_size, self.output_size)])
        
        self.sensor = ComplexitySensor()
        self.active_classes_per_column = {}
        
        # Модуль любопытства (опционально)
        self.use_curiosity = use_curiosity and CLIP_AVAILABLE
        if self.use_curiosity:
            self.curiosity = CuriosityModule()
        
        # SUBJECTIVE TIME CRITIC (из проекта 06)
        self.use_subjective_time = use_subjective_time
        if self.use_subjective_time:
            self.critic = SubjectiveTimeCritic(feature_dim=self.hidden_size)
            self.critic_optimizer = None  # Будет создан при обучении
            self.ref_backbone = None  # Снимок backbone для регуляции
        
        # VAE для генерации реалистичных снов (из проекта 01, 05)
        self.use_vae_dreams = use_vae_dreams
        if self.use_vae_dreams:
            self.dream_vae = DreamVAE(z_dim=128)
            self.vae_trained = False
        
        # 1️⃣ БУФЕР КОНФЛИКТОВ: Запоминаем несоответствия между моделью и CLIP
        self.conflict_buffer = []  # [(confidence_model, entropy_model, clip_label, clip_conf, image, true_label)]
        self.max_conflicts = 100  # Максимальный размер буфера
        
        # REPLAY BUFFER для восстановления памяти (из проекта 05)
        self.replay_buffer = {'X': [], 'Y': []}
        self.max_replay_size = 1000
        
        # 3️⃣ КЛАСС "UNKNOWN": Индекс для неопределенных объектов
        self.unknown_class_idx = 10 

    def set_initial_responsibility(self, classes):
        self.active_classes_per_column[0] = classes

    def freeze_past(self, use_fractal_time=False):
        """
        Замораживание прошлого с опциональным Fractal Time
        (разные уровни защиты для разных слоев backbone)
        """
        print("[FREEZING] Memory (Crystallization)...")
        
        if use_fractal_time:
            # FRACTAL TIME: разные lambda для разных слоев
            # Conv1-2: очень медленно (lambda_fc1=10000.0)
            # Conv3-4: медленно (lambda_fc2=3000.0)
            # Head: быстро (lambda_head=0.0)
            print("[FRACTAL TIME] Different protection levels per layer group")
            # Замораживаем только ранние слои (более консервативно)
            for name, param in self.shared_backbone.named_parameters():
                if 'conv1' in name or 'conv2' in name or 'bn1' in name or 'bn2' in name:
                    param.requires_grad = False
            # Поздние слои остаются обучаемыми (но с регуляризацией)
        else:
            # Стандартное замораживание: все веса backbone
            for param in self.shared_backbone.parameters():
                param.requires_grad = False
        
        # Замораживаем все старые головы кроме последней
        for i in range(len(self.heads) - 1):
            for param in self.heads[i].parameters():
                param.requires_grad = False
        
        # Сохраняем снимок для Subjective Time Critic
        if self.use_subjective_time:
            self.ref_backbone = copy.deepcopy(self.shared_backbone)
            self.ref_backbone.eval()
            for p in self.ref_backbone.parameters():
                p.requires_grad = False
    
    def _set_bn_train(self, train: bool):
        """Переключает только BN модули в train/eval, остальное не трогает"""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train(train)
    
    def recalibrate_bn(self, loader, device, num_batches=20):
        """Калибровка BN статистик на новых данных (без обучения весов)"""
        was_training = self.training
        # Переключаем модель в eval, чтобы головы не "шумели"
        self.eval()
        # Но BN временно в train, чтобы обновить running stats
        self._set_bn_train(True)
        
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= num_batches:
                    break
                x = x.to(device)
                # Калибруем только backbone BN (быстрее и стабильнее)
                _ = self.shared_backbone(x)
        
        # Возвращаем BN в eval
        self._set_bn_train(False)
        # Восстанавливаем общий режим
        if was_training:
            self.train()
        else:
            self.eval()

    def expand(self, new_classes_indices, use_fractal_time=False):
        self.freeze_past(use_fractal_time=use_fractal_time)
        # C) Создаем только новую голову (backbone общий)
        prev_dims = [h.hidden_size for h in self.heads]
        device = next(self.parameters()).device
        
        new_head = ExpandableHead(self.hidden_size, self.output_size, prev_dims).to(device)
        self.heads.append(new_head)
        
        # Для обратной совместимости (dream_and_compress использует columns)
        new_col = TemporalColumn(0, self.hidden_size, self.output_size, prev_dims).to(device)
        self.columns.append(new_col)
        
        self.active_classes_per_column[len(self.heads)-1] = new_classes_indices
        self.sensor = ComplexitySensor() 
        print(f"[EMERGENCE] Head {len(self.heads)} created (shared backbone). Scope: {new_classes_indices}")
        return new_head.parameters()  # Обучаем только новую голову
    
    def add_to_replay_buffer(self, X, Y, max_samples_per_class=100):
        """Добавляет образцы в replay buffer для восстановления памяти"""
        for x, y in zip(X, Y):
            if len(self.replay_buffer['X']) < self.max_replay_size:
                self.replay_buffer['X'].append(x.detach().cpu().clone())
                self.replay_buffer['Y'].append(y.item() if isinstance(y, torch.Tensor) else y)
            else:
                # Заменяем случайный элемент
                idx = np.random.randint(0, self.max_replay_size)
                self.replay_buffer['X'][idx] = x.detach().cpu().clone()
                self.replay_buffer['Y'][idx] = y.item() if isinstance(y, torch.Tensor) else y
    
    def sample_replay_batch(self, batch_size, device):
        """Сэмплирует батч из replay buffer"""
        if len(self.replay_buffer['X']) == 0:
            return None, None
        n = min(batch_size, len(self.replay_buffer['X']))
        indices = np.random.choice(len(self.replay_buffer['X']), n, replace=False)
        X = torch.stack([self.replay_buffer['X'][i] for i in indices]).to(device)
        Y = torch.tensor([self.replay_buffer['Y'][i] for i in indices], dtype=torch.long).to(device)
        return X, Y
    
    def recover_head_only(self, loader, device, epochs=20, lr=0.001):
        """
        Восстановление памяти через обучение только головы (из проекта 04)
        Позволяет восстановить забытые задачи с минимальными данными
        """
        print(f"[RECOVERY] Head-only recovery for {epochs} epochs...")
        # Замораживаем backbone
        for param in self.shared_backbone.parameters():
            param.requires_grad = False
        
        # Обучаем только последнюю голову
        if len(self.heads) > 0:
            optimizer = optim.Adam(self.heads[-1].parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            self.train()
            for epoch in range(epochs):
                total_loss = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    
                    backbone_features = self.shared_backbone(x)
                    logits, _ = self.heads[-1](backbone_features, prev_hiddens=[])
                    loss = criterion(logits[:, :10], y)
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 5 == 0:
                    print(f"   Recovery epoch {epoch+1}/{epochs}: Loss {total_loss/len(loader):.4f}")
        
        # Размораживаем backbone
        for param in self.shared_backbone.parameters():
            param.requires_grad = True
    
    def record_conflict(self, confidence_model, entropy_model, clip_class, clip_label, clip_conf, image, true_label=None):
        """Запоминаем конфликт между моделью и CLIP для дальнейшего использования"""
        conflict = {
            'confidence_model': confidence_model,
            'entropy_model': entropy_model,
            'clip_class': clip_class,  # Индекс класса
            'clip_label': clip_label,  # Строковое название
            'clip_conf': clip_conf,
            'image': image.detach().clone(),
            'true_label': true_label
        }
        self.conflict_buffer.append(conflict)
        
        # Ограничиваем размер буфера
        if len(self.conflict_buffer) > self.max_conflicts:
            self.conflict_buffer.pop(0)
    
    def get_conflict_statistics(self):
        """Статистика по конфликтам"""
        if not self.conflict_buffer:
            return None
        
        total = len(self.conflict_buffer)
        correct_clip = sum(1 for c in self.conflict_buffer 
                          if c['true_label'] is not None and 
                          c['clip_class'] == c['true_label'])
        
        avg_entropy = np.mean([c['entropy_model'] for c in self.conflict_buffer])
        avg_clip_conf = np.mean([c['clip_conf'] for c in self.conflict_buffer])
        
        return {
            'total_conflicts': total,
            'clip_correct': correct_clip,
            'clip_accuracy': correct_clip / total if total > 0 else 0,
            'avg_entropy': avg_entropy,
            'avg_clip_confidence': avg_clip_conf
        }
    
    def get_clip_soft_targets(self, images):
        """2️⃣ Получаем soft targets от CLIP для использования как teacher (D: батчевый проход)"""
        if not self.use_curiosity:
            return None
        
        device = images.device
        batch_size = images.size(0)
        
        # D) Батчевый проход вместо цикла (оптимизированная версия)
        try:
            # Подготавливаем все изображения сразу (батчевая обработка)
            # images: [B, 3, 32, 32] -> [B, 3, 224, 224]
            images_batch = images.to(device)
            # Применяем нормализацию и ресайз ко всему батчу сразу
            images_batch = images_batch * 0.5 + 0.5  # [-1,1] -> [0,1]
            images_batch = torch.clamp(images_batch, 0, 1)
            images_prep = F.interpolate(images_batch, size=(224, 224), mode='bilinear', align_corners=False)
            images_prep = (images_prep - self.curiosity.clip_mean) / self.curiosity.clip_std
            
            with torch.no_grad():
                logits_per_image, _ = self.curiosity.model(images_prep, self.curiosity.text_inputs)
                probs = logits_per_image.softmax(dim=-1)  # [B, 10]
            
            return probs.to(device)
        except Exception as e:
            print(f"[WARNING] Batch CLIP failed: {e}, falling back to per-image")
            # Fallback на старый метод
            probs_list = []
            for i in range(batch_size):
                _, _, _, probs = self.curiosity.what_is_this(images[i:i+1], return_probs=True)
                if probs is None:
                    probs = torch.full((10,), 0.1, device=device)
                else:
                    probs = probs.to(device)
                probs_list.append(probs)
            return torch.stack(probs_list).to(device)
    
    def dream_and_compress(self, num_dreams=1000, dream_batch_size=100):
        """
        🌙 МОДУЛЬ СНОВИДЕНИЙ (CONSOLIDATION) + LAZARUS v3
        Интегрирует механизмы из других экспериментов:
        1. Consistency (Behavior Anchor) - главный компонент Lazarus (91.5% recovery)
        2. Stability (Local Invariance) - устойчивость к шуму входов
        3. Entropy Floor - предотвращение коллапса в "уверенную ошибку"
        4. Knowledge Distillation - сжатие знаний всех голов в одну
        """
        print("\n🌙 ENTERING SLEEP PHASE (Lazarus v3 + Consolidation)...")
        print(f"   Current heads: {len(self.heads)}")
        
        if len(self.heads) <= 1:
            print("   Only one head exists. No compression needed.")
            return
        
        device = next(self.parameters()).device
        
        # 1. Создаем "Студента" - одну компактную сеть
        student_head = ExpandableHead(self.hidden_size, self.output_size).to(device)
        student = TemporalColumn(0, self.hidden_size, self.output_size).to(device)
        optimizer = optim.Adam(student_head.parameters(), lr=0.0005)  # Меньше LR для стабильности
        
        # 2. LAZARUS: Создаем frozen teacher (Consistency Anchor)
        # Это поведенческий якорь - главный компонент восстановления (91.5% recovery)
        teacher_model = copy.deepcopy(self)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        
        print(f"   Generating {num_dreams} dreams with Lazarus v3 protocol...")
        print(f"   Parameters: w_cons=1.0, w_stab=0.5, w_ent=0.05, H0=1.5")
        
        # Lazarus v3 параметры (из эксперимента 07-stability-first-cifar10)
        w_cons = 1.0  # Consistency (главный компонент - 91.5% recovery)
        w_stab = 0.5  # Stability (дополнительная стабилизация)
        w_ent = 0.05  # Entropy Floor (предотвращение коллапса)
        H0 = 1.5      # Минимальная энтропия
        epsilon = 0.05  # Шум для stability loss
        
        for epoch in range(15):  # Больше эпох для лучшей консолидации
            total_loss = 0
            total_cons = 0
            total_stab = 0
            total_ent = 0
            total_distill = 0
            
            for dream_batch in range(num_dreams // dream_batch_size):
                # Генерируем "сны" - VAE или улучшенный шум
                noise = self.sample_dreams(dream_batch_size, device)
                
                # LAZARUS v3: Consistency Anchor (главный компонент)
                with torch.no_grad():
                    teacher_logits = teacher_model(noise)
                    teacher_probs = torch.softmax(teacher_logits[:, :10], dim=1)
                
                # Студент предсказывает
                backbone_features = self.shared_backbone(noise)
                student_logits, _ = student_head(backbone_features, prev_hiddens=[])
                student_probs = torch.softmax(student_logits[:, :10], dim=1)
                
                # 1. Consistency Loss (MSE между student и teacher logits)
                # Это главный компонент Lazarus - поведенческий якорь
                loss_cons = F.mse_loss(student_logits[:, :10], teacher_logits[:, :10])
                
                # 2. Stability Loss (устойчивость к шуму входов)
                # Заставляет модель быть инвариантной к малым возмущениям
                noise_pert = noise + torch.randn_like(noise) * epsilon
                backbone_features_pert = self.shared_backbone(noise_pert)
                student_logits_pert, _ = student_head(backbone_features_pert, prev_hiddens=[])
                loss_stab = F.mse_loss(student_logits[:, :10], student_logits_pert[:, :10])
                
                # 3. Entropy Floor (предотвращение коллапса)
                # Предотвращает коллапс в "уверенную ошибку"
                log_probs = F.log_softmax(student_logits[:, :10], dim=1)
                entropy = -(student_probs * log_probs).sum(dim=1).mean()
                loss_ent = F.relu(H0 - entropy)  # Штраф за слишком низкую энтропию
                
                # 4. Knowledge Distillation (KL divergence для мягких меток)
                # Сжимает знания всех голов в одну
                loss_distill = F.kl_div(
                    F.log_softmax(student_logits[:, :10], dim=1),
                    teacher_probs,
                    reduction='batchmean'
                )
                
                # Итоговый loss (Lazarus v3 + Distillation)
                loss = w_cons * loss_cons + w_stab * loss_stab + w_ent * loss_ent + 0.3 * loss_distill
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_cons += loss_cons.item()
                total_stab += loss_stab.item()
                total_ent += loss_ent.item()
                total_distill += loss_distill.item()
            
            batches = num_dreams // dream_batch_size
            if (epoch + 1) % 3 == 0:
                print(f"   Epoch {epoch+1}/15: Total={total_loss/batches:.4f} "
                      f"(Cons={total_cons/batches:.4f}, Stab={total_stab/batches:.4f}, "
                      f"Ent={total_ent/batches:.4f}, Distill={total_distill/batches:.4f}, "
                      f"H={entropy.item():.3f})")
        
        print("☀️ WAKING UP: Lazarus Consolidation Complete.")
        
        # 4. Заменяем сложный мозг на одного Студента
        self.heads = nn.ModuleList([student_head])
        self.columns = nn.ModuleList([student])
        self.active_classes_per_column = {}
        
        print(f"   Memory compressed: {len(self.heads)} head(s) remaining (shared backbone).")
        return "Knowledge Compressed with Lazarus v3!"

    def forward(self, x, raw_image=None, return_curiosity_info=False, return_features=False):
        # C) Используем общий backbone + расширяемые головы
        # x: [B, 3, 32, 32]
        backbone_features = self.shared_backbone(x)  # [B, 512]
        
        hiddens = []
        final_logits = torch.zeros(x.size(0), self.output_size).to(x.device)
        
        curiosity_info = None
        
        # C) Проходим через все головы с общим backbone
        for i, head in enumerate(self.heads):
            out, h = head(backbone_features, hiddens)
            hiddens.append(h)
            
            if i in self.active_classes_per_column:
                indices = self.active_classes_per_column[i]
                mask = torch.zeros_like(out)
                mask[:, indices] = 1.0
                
                # Добавляем только разрешенные выходы
                final_logits = final_logits + (out * mask)
            else:
                # Если зона не определена, добавляем все (для первого слоя до расширения)
                final_logits = final_logits + out
        
        # 3️⃣ КЛАСС "UNKNOWN": E) Улучшенный механизм с калибровкой
        probs_known = torch.softmax(final_logits[:, :10], dim=1)  # Только известные классы
        entropy = -torch.sum(probs_known * torch.log(probs_known + 1e-9), dim=1)
        max_prob_known, _ = torch.max(probs_known, dim=1)
        
        # E) Адаптивные пороги (смягченные для более практичного использования)
        # Для 10 классов max entropy ~ ln(10)=2.302. Используем более мягкие пороги:
        # max_prob < 0.2 (более чувствительно) ИЛИ entropy > 1.8 (высокая неопределенность)
        unknown_mask = (max_prob_known < 0.2) | (entropy > 1.8)
        
        # E) Unknown logit относительный (устойчивее) + защита от пустого mask
        if unknown_mask.any():
            max_logit_known, _ = final_logits[:, :10].max(dim=1)
            # Делаем unknown немного выше максимального, но не слишком доминирующим
            final_logits[unknown_mask, self.unknown_class_idx] = max_logit_known[unknown_mask] + 1.5
        
        # Модуль любопытства: если энтропия высокая, спрашиваем CLIP
        curiosity_info = None
        if self.use_curiosity and raw_image is not None and return_curiosity_info:
            max_entropy = entropy.max().item()
            if max_entropy > 1.5:
                sample_image = raw_image[0:1]
                result = self.curiosity.what_is_this(sample_image)
                if result[0] is not None:
                    clip_class, clip_label, confidence = result
                    curiosity_info = {
                        'clip_class': clip_class,
                        'clip_label': clip_label,
                        'confidence': confidence,
                        'entropy': max_entropy
                    }
                
        if return_curiosity_info:
            return final_logits, curiosity_info
        if return_features:
            return final_logits, backbone_features
        return final_logits
    
    def train_vae_on_data(self, loader, device, epochs=10, lr=1e-3):
        """Обучает VAE на данных для генерации реалистичных снов"""
        if not self.use_vae_dreams:
            return
        
        print(f"[VAE] Training dream generator on {len(loader)} batches...")
        optimizer = optim.Adam(self.dream_vae.parameters(), lr=lr)
        self.dream_vae.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for x, _ in loader:
                x = x.to(device)
                optimizer.zero_grad()
                
                x_recon, mu, logvar = self.dream_vae(x)
                loss = vae_loss(x, x_recon, mu, logvar, beta=1.0)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 3 == 0:
                print(f"   VAE epoch {epoch+1}/{epochs}: Loss {total_loss/len(loader):.4f}")
        
        self.dream_vae.eval()
        self.vae_trained = True
        print("[VAE] Dream generator ready!")
    
    def sample_dreams(self, n, device):
        """Генерирует сны через VAE или белый шум"""
        if self.use_vae_dreams and self.vae_trained:
            # VAE сны (более реалистичные)
            with torch.no_grad():
                z = torch.randn(n, self.dream_vae.z_dim, device=device)
                dreams = self.dream_vae.decode(z)
                # Нормализуем в диапазон CIFAR-10
                dreams = torch.clamp(dreams, -1, 1)
            return dreams
        else:
            # Белый шум (fallback)
            noise = torch.randn(n, 3, 32, 32, device=device)
            return torch.tanh(noise * 0.5)  # Нормализуем

# --- 2. ДАННЫЕ: МАШИНЫ vs ПРИРОДА ---
def get_cifar_split():
    # B) Аугментации для обучения
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Без аугментаций для теста
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Путь к data относительно корня проекта
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data')
    train_full = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)
    test_full = datasets.CIFAR10(data_path, train=False, transform=test_transform)

    # CIFAR-10 Классы:
    # 0:Plane, 1:Car, 8:Ship, 9:Truck (ТЕХНИКА)
    # 2:Bird, 3:Cat, 4:Deer, 5:Dog, 6:Frog, 7:Horse (ЖИВОТНЫЕ)
    
    vehicles = [0, 1, 8, 9]
    animals = [2, 3, 4, 5, 6, 7]

    def get_indices(dataset, classes):
        indices = []
        for i in range(len(dataset)):
            if dataset.targets[i] in classes:
                indices.append(i)
        return indices

    print("Sorting Data into 'Machines' vs 'Nature'...")
    idx_train_A = get_indices(train_full, vehicles)
    idx_train_B = get_indices(train_full, animals)
    idx_test_A = get_indices(test_full, vehicles)
    idx_test_B = get_indices(test_full, animals)

    return (Subset(train_full, idx_train_A), Subset(train_full, idx_train_B),
            Subset(test_full, idx_test_A), Subset(test_full, idx_test_B),
            vehicles, animals)

# --- Утилита для оценки с маскированием ---
def eval_masked(agent, loader, allowed_classes, device, block_unknown=True):
    """Оценка точности с маскированием классов и правильным eval режимом"""
    was_training = agent.training
    agent.eval()
    correct = total = 0
    with torch.no_grad():
        for d, t in loader:
            d, t = d.to(device), t.to(device)
            out = agent(d)
            out_masked = out.clone()
            out_masked[:, [i for i in range(10) if i not in allowed_classes]] = -float('inf')
            if block_unknown:
                out_masked[:, agent.unknown_class_idx] = -float('inf')
            pred = out_masked.argmax(dim=1)
            correct += (pred == t).sum().item()
            total += t.size(0)
    if was_training:
        agent.train()
    return 100.0 * correct / total

# --- 3. ЗАПУСК ---
def run_drone_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024 ** 3:.2f} GB")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print()

    train_A, train_B, test_A, test_B, classes_A, classes_B = get_cifar_split()
    
    loader_A = DataLoader(train_A, batch_size=128, shuffle=True)
    loader_B = DataLoader(train_B, batch_size=128, shuffle=True)
    test_loader_A = DataLoader(test_A, batch_size=500)
    test_loader_B = DataLoader(test_B, batch_size=500)

    # Создаем агента с максимальной интеграцией всех механизмов
    use_curiosity = CLIP_AVAILABLE
    use_subjective_time = True  # Автоматическая регуляция пластичности
    use_vae_dreams = True       # Реалистичные сны вместо белого шума
    use_fractal_time = True      # Разные уровни защиты для разных слоев
    use_adaptive_pain = True     # Динамический lambda на основе конфликта градиентов
    
    agent = RecursiveAgent(
        use_curiosity=use_curiosity,
        use_subjective_time=use_subjective_time,
        use_vae_dreams=use_vae_dreams
    ).to(device)
    agent.set_initial_responsibility(classes_A)
    
    if use_curiosity:
        print("[INFO] Curiosity Module (CLIP) enabled - agent can query world knowledge!")
    if use_subjective_time:
        print("[INFO] Subjective Time Critic enabled - adaptive plasticity regulation!")
    if use_vae_dreams:
        print("[INFO] VAE Dream Generator enabled - realistic dream generation!")
    if use_fractal_time:
        print("[INFO] Fractal Time enabled - layer-wise protection levels!")
    if use_adaptive_pain:
        print("[INFO] Adaptive Time/Pain enabled - dynamic lambda from gradient conflict!")
    
    # Улучшения для Phase1: AdamW + weight_decay для лучшего разделения Plane/Ship/Car/Truck
    optimizer = optim.AdamW(agent.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Label smoothing для стабильности
    
    # Subjective Time Critic optimizer (если включен)
    critic_optimizer = None
    if use_subjective_time:
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-3)

    acc_A_hist, acc_B_hist = [], []
    step = 0
    phase_transition_step = []

    print(f"\n--- PHASE 1: URBAN ENVIRONMENT (Learning Machines: {classes_A}) ---")
    
    # Обучение VAE на Phase 1 данных (если включен)
    if use_vae_dreams:
        print("[VAE] Pre-training dream generator on Phase 1 data...")
        agent.train_vae_on_data(loader_A, device, epochs=5, lr=1e-3)
    
    # Обучение Фаза 1 (больше эпох для лучшего разделения Plane/Ship/Car/Truck)
    # Cosine LR schedule для стабильного обучения
    steps_per_epoch_A = len(loader_A)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch_A * 15, eta_min=1e-5)
    
    # Сохраняем образцы в replay buffer
    replay_samples_collected = 0
    
    for epoch in range(15):  # Увеличено для лучшего разделения vehicles
        for batch_idx, (data, target) in enumerate(loader_A):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # B1) Фаза 1: loss только по 10 классам
            logits, features = agent(data, return_features=True)
            loss = criterion(logits[:, :10], target)
            
            # Subjective Time Critic (если включен)
            surprise = None
            current_lambda = 0.0
            if use_subjective_time:
                # Предсказываем loss через Critic
                predicted_loss = agent.critic(features.detach())  # [batch]
                real_loss_per_sample = criterion(logits, target)  # [batch] - нужен reduction='none'
                # Используем средний loss для Surprise
                mean_real_loss = loss
                mean_predicted_loss = predicted_loss.mean()
                surprise = agent.critic.compute_surprise(mean_predicted_loss, mean_real_loss)
                current_lambda = agent.critic.compute_lambda(surprise, base_lambda=0.0)  # Phase1: нет защиты
            
                # Обучаем Critic
                critic_loss = nn.MSELoss()(mean_predicted_loss, mean_real_loss.detach())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # Cosine LR schedule
            
            # Добавляем в replay buffer (первые 1000 образцов)
            if replay_samples_collected < agent.max_replay_size:
                agent.add_to_replay_buffer(data[:min(32, len(data))], target[:min(32, len(target))])
                replay_samples_collected += min(32, len(data))
            
            agent.sensor.update(loss.item())
            if step == 50: agent.sensor.calibrate()
            
            if step % 50 == 0:
                # Тест с правильным eval режимом
                acc = eval_masked(agent, test_loader_A, classes_A, device, block_unknown=True)
                acc_A_hist.append(acc); acc_B_hist.append(0)
                surprise_str = f" | Surprise: {surprise.item():.4f}" if surprise is not None else ""
                print(f"Step {step}: Loss {loss.item():.2f} | Acc Machines: {acc:.1f}%{surprise_str}")
            step += 1

    print(f"\n--- PHASE 2: WILDERNESS (Reality Shift to Animals: {classes_B}) ---")
    phase_transition_step.append(len(acc_A_hist))
    expansion_count = 0  # Счетчик расширений (вместо флага expanded)
    
    # --- СОСТОЯНИЕ АГЕНТА И ПРЕДОХРАНИТЕЛИ ---
    last_expansion_step = -1000  # Когда последний раз росли
    COOLDOWN_STEPS = 200         # Рефрактерный период (шагов)
    CLIP_TRUST_THRESHOLD = 0.6   # Верим CLIP только если он уверен > 60%
    MAX_LAYERS = 5               # Защита от переполнения памяти
    
    # Phase2 optimizer и scheduler будут созданы после expansion
    optimizer_phase2 = None
    scheduler_phase2 = None
    steps_per_epoch_B = len(loader_B)
    total_steps_phase2 = steps_per_epoch_B * 8  # Для scheduler после expansion
    
    for epoch in range(8):  # Еще больше эпох для дифференциации животных
        for batch_idx, (data, target) in enumerate(loader_B):
            data, target = data.to(device), target.to(device)
            
            # 1. Предсказание и расчет Loss (Боли)
            # B2) test_loss только по 10 классам
            with torch.no_grad():
                test_out = agent(data)
                test_loss = criterion(test_out[:, :10], target)
            
            # ЛОГИКА АКТИВНОГО ОБУЧЕНИЯ: Доверяй Боли, а не Уверенности
            # ПРОВЕРКА НА ШОК И ЗАЩИТА ОТ ЗАЦИКЛИВАНИЯ
            is_shock = agent.sensor.is_shock(test_loss.item())
            can_expand = (step - last_expansion_step) > COOLDOWN_STEPS
            has_budget = len(agent.heads) < MAX_LAYERS
            
            if is_shock and can_expand and has_budget:
                print(f"\n[VISUAL CORTEX SHOCK] Loss {test_loss.item():.2f} detected (High Surprise).")
                print(f"[SAFETY] Checking expansion conditions: Cooldown OK, Budget OK ({len(agent.heads)}/{MAX_LAYERS} heads)")
                
                # Принудительно спрашиваем CLIP, потому что нам "больно" (высокий Loss)
                if agent.use_curiosity:
                    print("[CURIOSITY] Internal confidence is unreliable. Querying Oracle (CLIP)...")
                    # Берем первый пример из батча, вызвавший шок
                    result = agent.curiosity.what_is_this(data[0:1])
                    
                    if result[0] is not None:
                        best_idx, best_label, conf = result
                        
                        # ПРЕДОХРАНИТЕЛЬ №1: Доверяем ли мы Оракулу?
                        if conf > CLIP_TRUST_THRESHOLD:
                            print(f"[EUREKA] CLIP is confident ({conf*100:.1f}%) it's a '{best_label}'")
                            print(f"[ADAPTATION] Triggering Phase Transition for concept: {best_label}...")
                            
                            # 1️⃣ ЗАПОМИНАЕМ КОНФЛИКТ
                            # E) Используем только 10 классов для энтропии
                            with torch.no_grad():
                                agent.eval()  # Переключаем в eval для BatchNorm
                                model_out = agent(data[0:1])
                                agent.train()  # Возвращаем в train режим
                                model_probs = torch.softmax(model_out[:, :10], dim=1)
                                model_conf, model_pred = torch.max(model_probs, 1)
                                model_entropy = -torch.sum(model_probs * torch.log(model_probs + 1e-9), dim=1).item()
                                print(f"[LOG] Model confidence: {model_conf.item():.3f}, Entropy: {model_entropy:.3f}")
                                
                                agent.record_conflict(
                                    confidence_model=model_conf.item(),
                                    entropy_model=model_entropy,
                                    clip_class=best_idx,
                                    clip_label=best_label,
                                    clip_conf=conf,
                                    image=data[0:1],
                                    true_label=target[0].item() if len(target) > 0 else None
                                )
                            
                            # Расширяем сознание (с Fractal Time если включен)
                            new_params = agent.expand(new_classes_indices=classes_B, use_fractal_time=use_fractal_time)
                            # Калибруем BN статистики на новых данных (Phase2)
                            agent.recalibrate_bn(loader_B, device, num_batches=20)
                            # Создаем новый optimizer и scheduler для Phase2
                            optimizer_phase2 = optim.Adam(new_params, lr=0.002)  # x2 для новой головы
                            # T_max = оставшиеся шаги после expansion (примерно)
                            # Используем общее число шагов Phase2 минус уже пройденные
                            steps_per_epoch_A = len(loader_A)
                            steps_already_done = step - (phase_transition_step[-1] * steps_per_epoch_A if phase_transition_step else 0)
                            remaining_steps = max(total_steps_phase2 - steps_already_done, steps_per_epoch_B)
                            scheduler_phase2 = CosineAnnealingLR(optimizer_phase2, T_max=remaining_steps, eta_min=1e-5)
                            expansion_count += 1
                            last_expansion_step = step
                            
                            # Обновляем Subjective Time Critic после expansion
                            if use_subjective_time:
                                agent.ref_backbone = copy.deepcopy(agent.shared_backbone)
                                agent.ref_backbone.eval()
                                for p in agent.ref_backbone.parameters():
                                    p.requires_grad = False
                        else:
                            print(f"[IGNORE] CLIP is unsure ({conf*100:.1f}% < {CLIP_TRUST_THRESHOLD*100:.0f}%). Skipping expansion to prevent hallucination.")
            
            elif is_shock and not can_expand:
                # Мы в шоке, но у нас "отходняк" после прошлого роста
                if step % 50 == 0:  # Показываем только периодически
                    remaining = COOLDOWN_STEPS - (step - last_expansion_step)
                    print(f"[COOLDOWN] Shock detected but in refractory period ({remaining} steps remaining)")
            
            elif is_shock and not has_budget:
                print(f"\n[CRITICAL] Head Limit ({MAX_LAYERS}) Reached. Brain is full.")
                print(f"[ACTION] Initiating SLEEP PHASE to compress knowledge...")
                
                # 1. ЗАПУСК СНА (Сжатие знаний)
                # Учитель (5 голов) учит Студента (1 голова) на псевдо-снах
                agent.dream_and_compress(num_dreams=1000, dream_batch_size=100)
                
                # 2. ПЕРЕЗАГРУЗКА
                # Так как мы удалили старые слои и создали новый, нужно обновить оптимизатор
                optimizer = optim.Adam(agent.parameters(), lr=0.001)
                
                # 3. СБРОС СОСТОЯНИЯ
                # Мы "выспались", теперь у нас 1 слой и куча свободного места
                expanded = True 
                last_expansion_step = step
                print("[WAKE UP] Agent is ready for new memories.")
                
            # 2. Обучение с использованием всех интегрированных механизмов
            # Используем правильный optimizer
            current_optimizer = optimizer_phase2 if optimizer_phase2 is not None else optimizer
            current_optimizer.zero_grad()
            
            # Forward pass с features для Subjective Time
            outputs, features = agent(data, return_features=True)
            
            # Базовый loss
            loss = criterion(outputs[:, :10], target)  # Только известные классы
            
            # REPLAY BUFFER: добавляем replay loss для защиты памяти (из проекта 05)
            replay_loss = 0.0
            if len(agent.replay_buffer['X']) > 0:
                x_replay, y_replay = agent.sample_replay_batch(batch_size=32, device=device)
                if x_replay is not None:
                    outputs_replay = agent(x_replay)
                    replay_loss = criterion(outputs_replay[:, :10], y_replay)
            
            # SUBJECTIVE TIME CRITIC: динамическая регуляция пластичности
            surprise = None
            current_lambda = 10000.0  # Базовый lambda для Phase2
            stability_loss = 0.0
            if use_subjective_time and agent.ref_backbone is not None:
                # Предсказываем loss через Critic
                predicted_loss = agent.critic(features.detach())  # [batch]
                real_loss_per_sample = criterion(outputs, target)  # Нужен reduction='none'
                mean_real_loss = loss
                mean_predicted_loss = predicted_loss.mean()
                surprise = agent.critic.compute_surprise(mean_predicted_loss, mean_real_loss)
                current_lambda = agent.critic.compute_lambda(surprise, base_lambda=10000.0, sensitivity=10.0)
                
                # Stability Loss (Backbone Anchor) - защита памяти
                backbone_params = list(agent.shared_backbone.parameters())
                backbone_ref_params = list(agent.ref_backbone.parameters())
                for p, p_ref in zip(backbone_params, backbone_ref_params):
                    stability_loss += (p - p_ref).pow(2).sum()
                
                # Обучаем Critic
                critic_loss = nn.MSELoss()(mean_predicted_loss, mean_real_loss.detach())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
            
            # ADAPTIVE TIME/PAIN: динамический lambda на основе конфликта градиентов
            adaptive_lambda = current_lambda
            if use_adaptive_pain and len(agent.replay_buffer['X']) > 0:
                x_replay, y_replay = agent.sample_replay_batch(batch_size=32, device=device)
                if x_replay is not None:
                    # Вычисляем градиенты для нового и старого loss
                    backbone_params = list(agent.shared_backbone.parameters())
                    
                    loss_new = criterion(outputs[:, :10], target)
                    loss_old = criterion(agent(x_replay)[:, :10], y_replay)
                    
                    g_new = torch.autograd.grad(loss_new, backbone_params, retain_graph=True, create_graph=False)
                    g_old = torch.autograd.grad(loss_old, backbone_params, retain_graph=True, create_graph=False)
                    
                    # Вычисляем косинус между градиентами
                    g_new_flat = torch.cat([gi.detach().flatten() for gi in g_new])
                    g_old_flat = torch.cat([gi.detach().flatten() for gi in g_old])
                    
                    dot = torch.dot(g_new_flat, g_old_flat).item()
                    n1 = (g_new_flat.pow(2).sum().item() ** 0.5) + 1e-8
                    n2 = (g_old_flat.pow(2).sum().item() ** 0.5) + 1e-8
                    cos = dot / (n1 * n2)
                    
                    # Pain = (1 - cos) / 2, lambda = lambda_min + (lambda_max - lambda_min) * pain
                    pain = max(0.0, min(1.0, (1.0 - cos) * 0.5))
                    adaptive_lambda = 100.0 + (20000.0 - 100.0) * pain
                    current_lambda = adaptive_lambda  # Используем adaptive lambda
            
            # 2️⃣ ДОБАВЛЯЕМ KL-DIVERGENCE С CLIP (если высокая энтропия)
            kl_loss = 0.0
            if agent.use_curiosity:
                probs_model = torch.softmax(outputs[:, :10], dim=1)
                entropy = -torch.sum(probs_model * torch.log(probs_model + 1e-9), dim=1)
                high_entropy_mask = entropy > 1.5  # Высокая неопределенность
                
                if high_entropy_mask.any():
                    # (2) Ограничить CLIP teacher на шаг
                    MAX_UNCERTAIN = 16
                    idx = torch.where(high_entropy_mask)[0]
                    if idx.numel() > MAX_UNCERTAIN:
                        idx = idx[:MAX_UNCERTAIN]
                    
                    # Получаем soft targets от CLIP
                    uncertain_images = data[idx]
                    clip_targets = agent.get_clip_soft_targets(uncertain_images)
                    
                    if clip_targets is not None:
                        # D) KL divergence между моделью и CLIP (clip_targets уже распределение)
                        clip_probs = clip_targets  # уже prob distribution, не нужно softmax
                        kl_loss = F.kl_div(
                            torch.log(probs_model[idx] + 1e-9),
                            clip_probs,
                            reduction='batchmean'
                        )
                        # Планируемый KL коэффициент: warmup после expansion
                        if expansion_count > 0:
                            steps_since_expand = step - last_expansion_step
                            kl_weight = min(0.3, 0.3 * (steps_since_expand / 500))
                        else:
                            kl_weight = 0.3
                        # Сохраняем взвешенный KL loss
                        kl_loss = kl_weight * kl_loss
                        # 5) Логи только раз в N шагов (не каждый батч)
                        if step % 50 == 0:
                            print(f"[LOG] High entropy samples: {idx.numel()}, KL loss: {kl_loss.item():.4f}")
            
            # ИТОГОВЫЙ LOSS: базовый + replay + stability (Subjective Time) + adaptive pain + KL
            total_loss = loss
            if replay_loss > 0:
                total_loss = total_loss + 0.25 * replay_loss  # Replay fraction
            if stability_loss > 0:
                total_loss = total_loss + current_lambda * stability_loss  # Subjective Time stability
            if kl_loss > 0:
                total_loss = total_loss + kl_loss  # CLIP teacher distillation
            
            # Используем правильный optimizer
            current_optimizer = optimizer_phase2 if optimizer_phase2 is not None else optimizer
            current_optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            
            current_optimizer.step()
            
            if scheduler_phase2 is not None:
                scheduler_phase2.step()  # Cosine LR schedule для Phase2
            elif optimizer_phase2 is None:
                scheduler.step()  # Продолжаем Phase1 scheduler до expansion
            
            agent.sensor.update(total_loss.item())
            
            if step % 50 == 0:
                # Тест Памяти (Машины) и Нового (Животные) с правильным eval режимом
                acc_A = eval_masked(agent, test_loader_A, classes_A, device, block_unknown=True)
                acc_B = eval_masked(agent, test_loader_B, classes_B, device, block_unknown=True)
                
                acc_A_hist.append(acc_A); acc_B_hist.append(acc_B)
                
                # E) Unknown Rate (синхронизировано с новыми порогами в forward)
                with torch.no_grad():
                    test_out = agent(data)
                    probs_test = torch.softmax(test_out[:, :10], dim=1)
                    ent = -torch.sum(probs_test * torch.log(probs_test + 1e-9), dim=1)
                    mp, _ = torch.max(probs_test, dim=1)
                    # Синхронизировано с forward(): (max_prob < 0.2) | (entropy > 1.8)
                    unk_rate = ((mp < 0.2) | (ent > 1.8)).float().mean().item()
                
                print(f"Step {step}: Loss {loss.item():.2f} | Mem (Machines): {acc_A:.1f}% | New (Animals): {acc_B:.1f}% | Heads: {len(agent.heads)} | UnknownRate: {unk_rate*100:.1f}%")
            step += 1
    
    # 🌙 СОН: Консолидация памяти (если накопилось много голов)
    if len(agent.heads) >= 3:
        print(f"\n🌙 SLEEP PHASE: {len(agent.heads)} heads detected. Consolidating memories...")
        agent.dream_and_compress(num_dreams=500, dream_batch_size=50)
        
        # Пересоздаем оптимизатор для нового сжатого мозга
        optimizer = optim.Adam(agent.parameters(), lr=0.001)
        print("☀️ Agent woke up with consolidated knowledge.")

    # Классы CIFAR-10 + Unknown (определяем заранее для использования в анализе)
    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck', 'Unknown']
    
    # --- АНАЛИЗ НЕИЗВЕСТНЫХ ОБЪЕКТОВ С CLIP ---
    if agent.use_curiosity:
        print("\n--- ANALYZING UNKNOWN OBJECTS WITH CLIP ---")
        
        # Переключаем модель в eval режим для анализа
        agent.eval()
        
        # Берем несколько случайных изображений из тестового набора
        unknown_samples = []
        with torch.no_grad():
            for d, t in test_loader_B:
                d = d.to(device)
                # Обрабатываем батч целиком, чтобы избежать проблем с BatchNorm
                outputs = agent(d)
                # 2) Энтропия только по 10 классам (без Unknown)
                probs = torch.softmax(outputs[:, :10], dim=1)
                max_probs, predicted = torch.max(probs, 1)
                entropies = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
                
                # Берем первые 5 изображений для анализа
                for i in range(min(5, len(d))):
                    # Если модель не уверена (низкая вероятность или высокая энтропия)
                    if max_probs[i].item() < 0.5 or entropies[i].item() > 1.5:
                        image = d[i:i+1]  # Берем одно изображение для CLIP
                        true_class = t[i].item()
                        true_label = class_names[true_class]
                        unknown_samples.append({
                            'image': image,
                            'true_class': true_class,
                            'true_label': true_label,
                            'predicted': predicted[i].item(),
                            'confidence': max_probs[i].item(),
                            'entropy': entropies[i].item()
                        })
                        if len(unknown_samples) >= 5:
                            break
                if len(unknown_samples) >= 5:
                    break
        
        if unknown_samples:
            print(f"\nFound {len(unknown_samples)} uncertain objects. Querying CLIP for analysis...\n")
            for idx, sample in enumerate(unknown_samples, 1):
                print(f"--- Sample {idx} ---")
                print(f"True label: {sample['true_label']} (class {sample['true_class']})")
                print(f"Model prediction: {class_names[sample['predicted']]} (confidence: {sample['confidence']*100:.1f}%)")
                print(f"Model entropy: {sample['entropy']:.2f} (high = uncertain)")
                
                # Спрашиваем CLIP
                result = agent.curiosity.what_is_this(sample['image'])
                if result[0] is not None:
                    clip_class, clip_label, clip_confidence = result
                    print(f"CLIP suggestion: '{clip_label}' (confidence: {clip_confidence*100:.1f}%)")
                    
                    # Проверяем, правильно ли CLIP угадал
                    if clip_class == sample['true_class']:
                        print(f"[CORRECT] CLIP identified it correctly! Model was uncertain.")
                    elif clip_class in classes_B:  # CLIP предложил класс из правильной группы
                        print(f"[PARTIAL] CLIP suggested different class from same group (animals).")
                    else:
                        print(f"[WRONG] CLIP also uncertain or wrong.")
                    
                    # 1️⃣ ЗАПОМИНАЕМ КОНФЛИКТ
                    agent.record_conflict(
                        confidence_model=sample['confidence'],
                        entropy_model=sample['entropy'],
                        clip_class=clip_class,
                        clip_label=clip_label,
                        clip_conf=clip_confidence,
                        image=sample['image'],
                        true_label=sample['true_class']
                    )
                print()
        
        # Выводим статистику конфликтов
        conflict_stats = agent.get_conflict_statistics()
        if conflict_stats:
            print("=== CONFLICT BUFFER STATISTICS ===")
            print(f"Total conflicts recorded: {conflict_stats['total_conflicts']}")
            print(f"CLIP accuracy on conflicts: {conflict_stats['clip_accuracy']*100:.1f}%")
            print(f"Average model entropy: {conflict_stats['avg_entropy']:.2f}")
            print(f"Average CLIP confidence: {conflict_stats['avg_clip_confidence']*100:.1f}%")
            print()
        else:
            print("No uncertain objects found in test set.")
    
    # --- ТЕСТ НА ВСЕХ КЛАССАХ (включая невиданные) ---
    print("\n--- TESTING ON ALL CLASSES (Including Unseen) ---")
    
    # Загружаем полный тестовый набор
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data')
    test_full = datasets.CIFAR10(data_path, train=False, transform=transform)
    test_loader_all = DataLoader(test_full, batch_size=500, shuffle=False)
    
    # Статистика по классам (включая Unknown)
    class_correct = {i: 0 for i in range(11)}
    class_total = {i: 0 for i in range(10)}  # Unknown не может быть true_label
    class_predictions = {i: {j: 0 for j in range(11)} for i in range(10)}      # confusion matrix
    unknown_count = 0  # Сколько раз модель сказала "не знаю" (без блокировки)
    unknown_count_blocked = 0  # Сколько раз модель сказала "не знаю" (с блокировкой для accuracy)
    
    with torch.no_grad():
        for data, target in test_loader_all:
            data, target = data.to(device), target.to(device)
            outputs = agent(data)
            
            # Применяем маскирование в зависимости от класса
            for i, (d, t) in enumerate(zip(data, target)):
                out = outputs[i:i+1]
                true_class = t.item()
                
                # Определяем, к какой группе относится класс
                if true_class in classes_A:
                    # Для техники - маскируем все кроме техники
                    out_masked = out.clone()
                    out_masked[:, [j for j in range(10) if j not in classes_A]] = -float('inf')
                    # Для accuracy: блокируем unknown
                    out_masked_blocked = out_masked.clone()
                    out_masked_blocked[:, agent.unknown_class_idx] = -float('inf')
                    _, pred = torch.max(out_masked_blocked, 1)
                    # Для статистики unknown: без блокировки
                    _, pred_unblocked = torch.max(out_masked, 1)
                elif true_class in classes_B:
                    # Для животных - маскируем все кроме животных
                    out_masked = out.clone()
                    out_masked[:, [j for j in range(10) if j not in classes_B]] = -float('inf')
                    # Для accuracy: блокируем unknown
                    out_masked_blocked = out_masked.clone()
                    out_masked_blocked[:, agent.unknown_class_idx] = -float('inf')
                    _, pred = torch.max(out_masked_blocked, 1)
                    # Для статистики unknown: без блокировки
                    _, pred_unblocked = torch.max(out_masked, 1)
                else:
                    # 3) В CIFAR-10 все классы видны (либо фаза 1, либо фаза 2)
                    # Этот блок не сработает, но оставляем для будущих расширений
                    out_masked = out.clone()
                    out_masked_blocked = out_masked.clone()
                    out_masked_blocked[:, agent.unknown_class_idx] = -float('inf')
                    _, pred = torch.max(out_masked_blocked, 1)
                    _, pred_unblocked = torch.max(out_masked, 1)
                
                predicted_class = pred.item()
                predicted_class_unblocked = pred_unblocked.item()
                
                # Статистика unknown (без блокировки)
                if predicted_class_unblocked == agent.unknown_class_idx:
                    unknown_count += 1
                
                # Для accuracy используем blocked версию
                predicted_class = pred.item()
                class_total[true_class] += 1
                class_predictions[true_class][predicted_class] += 1
                if predicted_class == 10:  # Unknown класс
                    unknown_count += 1
                if true_class == predicted_class:
                    class_correct[true_class] += 1
    
    # Выводим результаты
    print("\n=== CLASSIFICATION RESULTS ===")
    print(f"{'Class':<10} {'Name':<10} {'Trained':<10} {'Accuracy':<10} {'Total':<10}")
    print("-" * 50)
    
    for i, name in enumerate(class_names):
        if i < 10:  # Известные классы
            trained = "YES" if i in classes_A or i in classes_B else "NO"
            acc = 100 * class_correct[i] / max(class_total[i], 1)
            print(f"{i:<10} {name:<10} {trained:<10} {acc:>6.1f}%    {class_total[i]:<10}")
        else:  # Unknown класс
            print(f"{i:<10} {name:<10} {'N/A':<10} {'N/A':<10} {unknown_count:<10} (times predicted)")
    
    # Анализ ошибок
    print("\n=== ERROR ANALYSIS ===")
    print("Most common misclassifications:")
    for true_class in range(10):
        if class_total[true_class] > 0:
            errors = [(pred_class, count) for pred_class, count in class_predictions[true_class].items() 
                     if pred_class != true_class and count > 0]
            if errors:
                errors.sort(key=lambda x: x[1], reverse=True)
                top_error = errors[0]
                print(f"{class_names[true_class]} (class {true_class}) -> {class_names[top_error[0]]} (class {top_error[0]}): {top_error[1]} times")
    
    # ВИЗУАЛИЗАЦИЯ
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # График 1: Точность по шагам
    axes[0].plot(acc_A_hist, label='Urban (Machines)', linewidth=3)
    axes[0].plot(acc_B_hist, label='Nature (Animals)', linewidth=3)
    if phase_transition_step:
        axes[0].axvline(x=phase_transition_step[0], color='r', linestyle='--', label='Environment Shift')
    axes[0].set_title("Recursive Emergence: Real-World Data (CIFAR-10)")
    axes[0].set_ylabel("Accuracy %")
    axes[0].set_xlabel("Training Steps (x50)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Точность по классам
    class_accs = [100 * class_correct[i] / max(class_total[i], 1) for i in range(10)]
    colors = ['green' if i in classes_A else ('blue' if i in classes_B else 'red') for i in range(10)]
    bars = axes[1].bar(range(10), class_accs, color=colors, alpha=0.7)
    axes[1].set_title("Accuracy by Class (Green=Trained Phase1, Blue=Trained Phase2, Red=Unseen)")
    axes[1].set_ylabel("Accuracy %")
    axes[1].set_xlabel("Class")
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels([f"{i}\n{name}" for i, name in enumerate(class_names[:10])], rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 100)
    
    # Добавляем значения на столбцы
    for i, (bar, acc) in enumerate(zip(bars, class_accs)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Сохраняем в тестовую папку
    output_path = os.path.join(os.path.dirname(__file__), 'cifar10_drone_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Graph saved as {output_path}")
    plt.show()

if __name__ == "__main__":
    run_drone_simulation()
