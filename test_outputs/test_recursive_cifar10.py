import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
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

class TemporalColumn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, prev_dims=[]):
        super().__init__()
        # A) CNN вместо MLP для CIFAR-10 (3-4 conv слоя + GAP + linear)
        # CIFAR-10: 32x32x3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Адаптеры для интеграции прошлых слоев
        self.adapters = nn.ModuleList([nn.Linear(p, 512) for p in prev_dims])
        
        # Финальный классификатор
        self.fc = nn.Linear(512, output_size)
        self.hidden_size = 512

    def forward(self, x, prev_hiddens):
        # x shape: [batch, 3072] -> reshape to [batch, 3, 32, 32]
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        
        # CNN backbone
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)  # 32x32 -> 16x16
        
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2)  # 16x16 -> 8x8
        
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pool2d(h, 2)  # 8x8 -> 4x4
        
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.gap(h)  # 4x4 -> 1x1
        h = h.view(h.size(0), -1)  # [batch, 512]
        
        # Интеграция прошлого
        for i, adapter in enumerate(self.adapters):
            if i < len(prev_hiddens):
                h = h + adapter(prev_hiddens[i])
        
        return self.fc(h), h

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
    def __init__(self, use_curiosity=False):
        super().__init__()
        # A) CNN работает с изображениями напрямую, не нужен input_size
        self.hidden_size = 512  # Размер скрытого представления после CNN
        self.output_size = 11  # 10 классов + 1 "unknown/ambiguous"
        
        self.columns = nn.ModuleList([TemporalColumn(0, self.hidden_size, self.output_size)])
        self.sensor = ComplexitySensor()
        self.active_classes_per_column = {}
        
        # Модуль любопытства (опционально)
        self.use_curiosity = use_curiosity and CLIP_AVAILABLE
        if self.use_curiosity:
            self.curiosity = CuriosityModule()
        
        # 1️⃣ БУФЕР КОНФЛИКТОВ: Запоминаем несоответствия между моделью и CLIP
        self.conflict_buffer = []  # [(confidence_model, entropy_model, clip_label, clip_conf, image, true_label)]
        self.max_conflicts = 100  # Максимальный размер буфера
        
        # 3️⃣ КЛАСС "UNKNOWN": Индекс для неопределенных объектов
        self.unknown_class_idx = 10 

    def set_initial_responsibility(self, classes):
        self.active_classes_per_column[0] = classes

    def freeze_past(self):
        print("[FREEZING] Memory (Crystallization)...")
        for param in self.parameters():
            param.requires_grad = False

    def expand(self, new_classes_indices):
        self.freeze_past()
        prev_dims = [c.hidden_size for c in self.columns]
        new_col = TemporalColumn(0, self.hidden_size, self.output_size, prev_dims)
        
        # Переносим на то же устройство (GPU/CPU), где живет агент
        device = next(self.parameters()).device
        new_col.to(device)
        
        self.columns.append(new_col)
        self.active_classes_per_column[len(self.columns)-1] = new_classes_indices
        self.sensor = ComplexitySensor() 
        print(f"[EMERGENCE] Layer {len(self.columns)} created. Scope: {new_classes_indices}")
        return new_col.parameters()
    
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
        🌙 МОДУЛЬ СНОВИДЕНИЙ (CONSOLIDATION)
        Генерирует "сны" (псевдо-данные) и сжимает знания всех слоев в один "Студент"
        """
        print("\n🌙 ENTERING SLEEP PHASE (Consolidating Memories)...")
        print(f"   Current layers: {len(self.columns)}")
        
        if len(self.columns) <= 1:
            print("   Only one layer exists. No compression needed.")
            return
        
        device = next(self.parameters()).device
        
        # 1. Создаем "Студента" - одну компактную сеть
        # Она должна быть такой же мощной, как сумма всех прошлых слоев
        student = TemporalColumn(0, self.hidden_size * 2, self.output_size).to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        
        # 2. Генерируем сны (Псевдо-данные)
        # Так как мы не храним картинки (Zero Replay), мы генерируем случайный шум
        # И заставляем нашу текущую сеть (Учителя) разметить этот шум
        
        print(f"   Generating {num_dreams} dreams...")
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(10):  # Быстрый сон (REM sleep)
            total_loss = 0
            
            for dream_batch in range(num_dreams // dream_batch_size):
                # Генерируем "Белый шум" (сны) - для CNN это изображения
                noise = torch.randn(dream_batch_size, 3, 32, 32).to(device)
                
                # Спрашиваем у текущего Мозга (всех слоев): "Что ты видишь в этом шуме?"
                with torch.no_grad():
                    teacher_logits = self.forward(noise)  # Учитель дает свои предсказания
                    teacher_probs = torch.softmax(teacher_logits[:, :10], dim=1)  # Только известные классы
                
                # 3. Учим Студента подражать Учителю
                student_logits, _ = student(noise, prev_hiddens=[])  # Студент пытается угадать
                
                # Loss: Студент должен выдавать те же вероятности, что и Учитель (Distillation Loss)
                loss = kl_loss_fn(
                    F.log_softmax(student_logits[:, :10], dim=1),
                    teacher_probs
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                print(f"   Dream epoch {epoch+1}/10: Loss {total_loss/(num_dreams//dream_batch_size):.4f}")
        
        print("☀️ WAKING UP: Consolidation Complete.")
        
        # 4. Заменяем сложный мозг на одного Студента
        self.columns = nn.ModuleList([student])
        self.active_classes_per_column = {}  # Сброс зон ответственности, теперь Студент знает всё
        
        print(f"   Memory compressed: {len(self.columns)} layer(s) remaining.")
        return "Knowledge Compressed!"

    def forward(self, x, raw_image=None, return_curiosity_info=False):
        # A) CNN работает с изображениями напрямую (x уже [B, 3, 32, 32])
        hiddens = []
        final_logits = torch.zeros(x.size(0), self.output_size).to(x.device)
        
        curiosity_info = None
        
        for i, col in enumerate(self.columns):
            out, h = col(x, hiddens)
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
        
        # 3️⃣ КЛАСС "UNKNOWN": Если энтропия очень высокая, активируем класс "не знаю"
        probs_known = torch.softmax(final_logits[:, :10], dim=1)  # Только известные классы
        entropy = -torch.sum(probs_known * torch.log(probs_known + 1e-9), dim=1)
        max_prob_known, _ = torch.max(probs_known, dim=1)
        
        # Если энтропия высокая И максимальная вероятность низкая -> "не знаю"
        unknown_mask = (entropy > 2.0) & (max_prob_known < 0.3)
        # C) Unknown logit относительный (устойчивее) + защита от пустого mask
        if unknown_mask.any():
            max_logit_known, _ = final_logits[:, :10].max(dim=1)
            final_logits[unknown_mask, self.unknown_class_idx] = max_logit_known[unknown_mask] + 1.0
        
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
        return final_logits

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

    # Создаем агента с модулем любопытства
    use_curiosity = CLIP_AVAILABLE
    agent = RecursiveAgent(use_curiosity=use_curiosity).to(device)
    agent.set_initial_responsibility(classes_A)
    
    if use_curiosity:
        print("[INFO] Curiosity Module (CLIP) enabled - agent can query world knowledge!")
    
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    acc_A_hist, acc_B_hist = [], []
    step = 0
    phase_transition_step = []

    print(f"\n--- PHASE 1: URBAN ENVIRONMENT (Learning Machines: {classes_A}) ---")
    
    # Обучение Фаза 1
    for epoch in range(3): # CIFAR сложнее, нужно пару эпох
        for batch_idx, (data, target) in enumerate(loader_A):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # B1) Фаза 1: loss только по 10 классам
            logits = agent(data)
            loss = criterion(logits[:, :10], target)
            loss.backward()
            optimizer.step()
            
            agent.sensor.update(loss.item())
            if step == 50: agent.sensor.calibrate()
            
            if step % 50 == 0:
                # Тест
                correct = 0; total = 0
                with torch.no_grad():
                    for d, t in test_loader_A:
                        d, t = d.to(device), t.to(device)
                        out = agent(d)
                        # Маскируем только классы техники
                        out_masked = out.clone()
                        out_masked[:, [i for i in range(10) if i not in classes_A]] = -float('inf')
                        out_masked[:, agent.unknown_class_idx] = -float('inf')  # B3) Запрещаем unknown
                        _, pred = torch.max(out_masked, 1)
                        correct += (pred == t).sum().item(); total += t.size(0)
                acc = 100 * correct / total
                acc_A_hist.append(acc); acc_B_hist.append(0)
                print(f"Step {step}: Loss {loss.item():.2f} | Acc Machines: {acc:.1f}%")
            step += 1

    print(f"\n--- PHASE 2: WILDERNESS (Reality Shift to Animals: {classes_B}) ---")
    phase_transition_step.append(len(acc_A_hist))
    expanded = False
    
    # --- СОСТОЯНИЕ АГЕНТА И ПРЕДОХРАНИТЕЛИ ---
    last_expansion_step = -1000  # Когда последний раз росли
    COOLDOWN_STEPS = 200         # Рефрактерный период (шагов)
    CLIP_TRUST_THRESHOLD = 0.6   # Верим CLIP только если он уверен > 60%
    MAX_LAYERS = 5               # Защита от переполнения памяти
    
    # Обучение Фаза 2
    for epoch in range(3):
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
            has_budget = len(agent.columns) < MAX_LAYERS
            
            if not expanded and is_shock and can_expand and has_budget:
                print(f"\n[VISUAL CORTEX SHOCK] Loss {test_loss.item():.2f} detected (High Surprise).")
                print(f"[SAFETY] Checking expansion conditions: Cooldown OK, Budget OK ({len(agent.columns)}/{MAX_LAYERS} layers)")
                
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
                            
                            # Расширяем сознание
                            new_params = agent.expand(new_classes_indices=classes_B)
                            optimizer = optim.Adam(new_params, lr=0.001)
                            expanded = True
                            last_expansion_step = step
                        else:
                            print(f"[IGNORE] CLIP is unsure ({conf*100:.1f}% < {CLIP_TRUST_THRESHOLD*100:.0f}%). Skipping expansion to prevent hallucination.")
            
            elif is_shock and not can_expand:
                # Мы в шоке, но у нас "отходняк" после прошлого роста
                if step % 50 == 0:  # Показываем только периодически
                    remaining = COOLDOWN_STEPS - (step - last_expansion_step)
                    print(f"[COOLDOWN] Shock detected but in refractory period ({remaining} steps remaining)")
            
            elif is_shock and not has_budget:
                print(f"\n[CRITICAL] Layer Limit ({MAX_LAYERS}) Reached. Brain is full.")
                print(f"[ACTION] Initiating SLEEP PHASE to compress knowledge...")
                
                # 1. ЗАПУСК СНА (Сжатие знаний)
                # Учитель (5 слоев) учит Студента (1 слой) на псевдо-снах
                agent.dream_and_compress(num_dreams=1000, dream_batch_size=100)
                
                # 2. ПЕРЕЗАГРУЗКА
                # Так как мы удалили старые слои и создали новый, нужно обновить оптимизатор
                optimizer = optim.Adam(agent.parameters(), lr=0.001)
                
                # 3. СБРОС СОСТОЯНИЯ
                # Мы "выспались", теперь у нас 1 слой и куча свободного места
                expanded = True 
                last_expansion_step = step
                print("[WAKE UP] Agent is ready for new memories.")
                
            # 2. Обучение с использованием CLIP как teacher (если доступно)
            optimizer.zero_grad()
            outputs = agent(data)
            
            # Базовый loss
            loss = criterion(outputs[:, :10], target)  # Только известные классы
            
            # 2️⃣ ДОБАВЛЯЕМ KL-DIVERGENCE С CLIP (если высокая энтропия)
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
                        # Комбинируем losses
                        loss = loss + 0.3 * kl_loss  # Коэффициент для баланса
                        # 5) Логи только раз в N шагов (не каждый батч)
                        if step % 50 == 0:
                            print(f"[LOG] High entropy samples: {idx.numel()}, KL loss: {kl_loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            
            agent.sensor.update(loss.item())
            
            if step % 50 == 0:
                # Тест Памяти (Машины)
                c_A = 0; t_A = 0
                with torch.no_grad():
                    for d, t in test_loader_A:
                        d, t = d.to(device), t.to(device)
                        out = agent(d)
                        # Маскируем только классы техники
                        out_masked = out.clone()
                        out_masked[:, [i for i in range(10) if i not in classes_A]] = -float('inf')
                        out_masked[:, agent.unknown_class_idx] = -float('inf')  # B3) Запрещаем unknown
                        _, pred = torch.max(out_masked, 1)
                        c_A += (pred == t).sum().item(); t_A += t.size(0)
                acc_A = 100 * c_A / t_A
                
                # Тест Нового (Животные)
                c_B = 0; t_B = 0
                with torch.no_grad():
                    for d, t in test_loader_B:
                        d, t = d.to(device), t.to(device)
                        out = agent(d)
                        # Маскируем только классы животных
                        out_masked = out.clone()
                        out_masked[:, [i for i in range(10) if i not in classes_B]] = -float('inf')
                        out_masked[:, agent.unknown_class_idx] = -float('inf')  # B3) Запрещаем unknown
                        _, pred = torch.max(out_masked, 1)
                        c_B += (pred == t).sum().item(); t_B += t.size(0)
                acc_B = 100 * c_B / t_B
                
                acc_A_hist.append(acc_A); acc_B_hist.append(acc_B)
                
                # (3) Лог unknown rate (чтобы понимать, что пороги адекватны)
                with torch.no_grad():
                    pk = torch.softmax(outputs[:, :10], dim=1)
                    ent = -torch.sum(pk * torch.log(pk + 1e-9), dim=1)
                    mp, _ = pk.max(dim=1)
                    unk_rate = ((ent > 2.0) & (mp < 0.3)).float().mean().item()
                
                print(f"Step {step}: Loss {loss.item():.2f} | Mem (Machines): {acc_A:.1f}% | New (Animals): {acc_B:.1f}% | Layers: {len(agent.columns)} | UnknownRate: {unk_rate*100:.1f}%")
            step += 1
    
    # 🌙 СОН: Консолидация памяти (если накопилось много слоев)
    if len(agent.columns) >= 3:
        print(f"\n🌙 SLEEP PHASE: {len(agent.columns)} layers detected. Consolidating memories...")
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
    class_predictions = {i: {j: 0 for j in range(11)} for i in range(10)}  # confusion matrix
    unknown_count = 0  # Сколько раз модель сказала "не знаю"
    
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
                    out_masked[:, agent.unknown_class_idx] = -float('inf')  # B3) Запрещаем unknown
                    _, pred = torch.max(out_masked, 1)
                elif true_class in classes_B:
                    # Для животных - маскируем все кроме животных
                    out_masked = out.clone()
                    out_masked[:, [j for j in range(10) if j not in classes_B]] = -float('inf')
                    out_masked[:, agent.unknown_class_idx] = -float('inf')  # B3) Запрещаем unknown
                    _, pred = torch.max(out_masked, 1)
                else:
                    # 3) В CIFAR-10 все классы видны (либо фаза 1, либо фаза 2)
                    # Этот блок не сработает, но оставляем для будущих расширений
                    out_masked = out.clone()
                    out_masked[:, agent.unknown_class_idx] = -float('inf')  # Запрещаем unknown
                    _, pred = torch.max(out_masked, 1)
                
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
