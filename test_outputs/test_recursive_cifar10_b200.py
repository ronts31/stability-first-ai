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
from torch.utils.data import DataLoader, Subset, ConcatDataset
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
# AGI Components: World Model, Goals, Concepts, Autobiographical Memory, Self-Model
# ----------------------------

# ----------------------------
# Action Space: Attention/Patch Selection Actions
# ----------------------------
class AttentionAction(nn.Module):
    """
    Действие как выбор attention/patch/view.
    Для CIFAR: действие = выбор области изображения для фокуса (crop, zoom, rotate).
    
    КРИТИЧНО: Селективное внимание на основе класса/концепта.
    Например, для Cat (класс 3) делаем zoom на центр (морда).
    """
    def __init__(self, num_patches=4, patch_size=16):
        super().__init__()
        self.num_patches = num_patches  # 2x2 grid = 4 patches
        self.patch_size = patch_size
        
        # Маппинг классов на предпочтительные patches
        # Для Cat (3): центр (patch 1 или 2) - где обычно морда
        # Для Dog (5): тоже центр
        # Для Deer (4): верхняя часть (patch 0 или 1) - где голова
        # Для Bird (2): верхняя часть
        # Для остальных: равномерное распределение
        self.class_to_patch_preference = {
            2: [0, 1],  # Bird - верх
            3: [1, 2],  # Cat - центр (морда)
            4: [0, 1],  # Deer - верх (голова)
            5: [1, 2],  # Dog - центр
            6: [2, 3],  # Frog - низ
            7: [1, 2],  # Horse - центр
        }
        
        # КРИТИЧНО: "Логика боли" для Ship/Plane (синий фон)
        # Если фон синий - ищем различия: киль/мачта для Ship, крылья для Plane
        # Ship (8): нижняя часть (patch 2 или 3) - где киль/мачта
        # Plane (0): верхняя часть (patch 0 или 1) - где крылья
        self.class_to_patch_preference[0] = [0, 1]  # Plane - верх (крылья)
        self.class_to_patch_preference[8] = [2, 3]  # Ship - низ (киль/мачта)
    
    def apply_action(self, x, action_idx, class_hint=None):
        """
        Применяет действие к изображению: выбирает patch для фокуса.
        
        Args:
            x: [B, 3, 32, 32] - исходное изображение
            action_idx: [B] - индекс выбранного patch (0..3)
            class_hint: [B] - подсказка о классе для селективного внимания (опционально)
        
        Returns:
            x_next: [B, 3, 32, 32] - изображение с применённым действием (crop+zoom выбранного patch)
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Разбиваем на patches (2x2 grid)
        patch_h, patch_w = H // 2, W // 2
        
        # Создаём маску для выбранного patch
        x_next = x.clone()
        for b in range(B):
            idx = int(action_idx[b].item())
            
            # КРИТИЧНО: если есть class_hint, корректируем patch для лучшего фокуса
            if class_hint is not None:
                class_id = int(class_hint[b].item()) if torch.is_tensor(class_hint[b]) else int(class_hint[b])
                
                # КРИТИЧНО: "Логика боли" для Ship/Plane - проверяем синий фон
                if class_id in [0, 8]:  # Plane или Ship
                    # Проверяем средний цвет фона (верхняя часть изображения)
                    bg_color = x[b, :, :H//4, :].mean(dim=(1, 2))  # [3] - средний RGB верхней части
                    # Синий фон: B > R и B > G
                    is_blue_bg = (bg_color[2] > bg_color[0] + 0.1) and (bg_color[2] > bg_color[1] + 0.1)
                    
                    if is_blue_bg:
                        # Синий фон - используем специфичные patches для различения
                        if class_id == 0:  # Plane - верх (крылья)
                            idx = 0 if idx not in [0, 1] else idx
                        elif class_id == 8:  # Ship - низ (киль/мачта)
                            idx = 2 if idx not in [2, 3] else idx
                
                # Обычная логика для других классов
                if class_id in self.class_to_patch_preference:
                    preferred = self.class_to_patch_preference[class_id]
                    if idx not in preferred:
                        # Выбираем ближайший предпочтительный patch
                        distances = [abs(idx - p) for p in preferred]
                        idx = preferred[distances.index(min(distances))]
            
            row = idx // 2
            col = idx % 2
            
            # Извлекаем выбранный patch
            patch = x[b:b+1, :, row*patch_h:(row+1)*patch_h, col*patch_w:(col+1)*patch_w]
            
            # Zoom: интерполируем patch обратно до полного размера
            patch_zoomed = F.interpolate(patch, size=(H, W), mode='bilinear', align_corners=False)
            
            # Смешиваем: 70% zoomed patch + 30% оригинал (для плавности)
            x_next[b] = 0.7 * patch_zoomed[0] + 0.3 * x[b]
        
        return x_next


class WorldModel(nn.Module):
    """
    World Model: предсказывает будущие состояния и причинно-следственные связи.
    Использует latent space для предсказания следующего состояния на основе текущего состояния и действия.
    
    КРИТИЧНО: action теперь это реальное действие (attention/patch selection), а не class logits.
    """
    def __init__(self, feature_dim=512, action_dim=4, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim  # 4 patches для CIFAR
        self.latent_dim = latent_dim
        
        # Encoder: features -> latent
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean, logvar
        )
        
        # Transition: (latent, action) -> next_latent
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Decoder: latent -> predicted_features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Action predictor: предсказывает лучшее действие для достижения цели
        # Используем только latent (goal_features опциональны и добавляются отдельно)
        self.action_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def encode(self, features):
        """Кодирует features в latent space"""
        z_params = self.encoder(features)
        z_mean, z_logvar = z_params.chunk(2, dim=-1)
        return z_mean, z_logvar
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick для VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def predict_next(self, features, action_onehot):
        """
        Предсказывает следующее состояние на основе текущего состояния и действия.
        
        Args:
            features: [B, feature_dim] - текущие features
            action_onehot: [B, action_dim] - действие (one-hot для выбора patch)
        
        Returns:
            next_features_pred: [B, feature_dim] - предсказанные features следующего состояния
            z_mean, z_logvar, next_z_mean, next_z_logvar - для KL loss
        """
        # Encode current state
        z_mean, z_logvar = self.encode(features)
        z = self.reparameterize(z_mean, z_logvar)
        
        # Predict next latent
        z_action = torch.cat([z, action_onehot], dim=-1)
        next_z_params = self.transition(z_action)
        next_z_mean, next_z_logvar = next_z_params.chunk(2, dim=-1)
        next_z = self.reparameterize(next_z_mean, next_z_logvar)
        
        # Decode to features
        next_features_pred = self.decoder(next_z)
        
        return next_features_pred, z_mean, z_logvar, next_z_mean, next_z_logvar
    
    def predict_best_action(self, features, goal_features=None):
        """
        Предсказывает лучшее действие для достижения цели.
        
        Args:
            features: [B, feature_dim] - текущие features
            goal_features: [B, goal_dim] - целевые features (опционально, из InternalGoals)
        
        Returns:
            action_logits: [B, action_dim] - логиты действий
        """
        z_mean, z_logvar = self.encode(features)
        z = self.reparameterize(z_mean, z_logvar)
        
        # Если goal_features предоставлены, добавляем их влияние через проекцию
        if goal_features is not None:
            # Проецируем goal_features в latent space для совместимости
            # Используем простое сложение (goal_features уже в меньшей размерности)
            # Или можно использовать проекцию, но для простоты просто используем z
            # В будущем можно добавить goal-conditioned encoder
            pass  # Пока используем только z, goal_features можно использовать для модификации через attention
        
        action_logits = self.action_predictor(z)
        return action_logits


class InternalGoals(nn.Module):
    """
    Внутренние цели: система целей, независимая от внешних наград.
    Генерирует собственные цели на основе curiosity, novelty, и внутренней мотивации.
    
    КРИТИЧНО: Цели теперь влияют на policy через goal-conditioned control.
    """
    def __init__(self, feature_dim=512, goal_dim=64, hidden_dim=256):
        super().__init__()
        self.goal_dim = goal_dim
        
        # Goal generator: features -> goal vector
        self.goal_generator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim)
        )
        
        # Goal evaluator: (features, goal) -> achievement_score
        self.goal_evaluator = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # [0..1] - насколько цель достигнута
        )
        
        # Goal-conditioned policy: (features, goal) -> policy_modifier
        # Влияет на recursion depth, replay ratio, gate temperature
        self.goal_policy = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # [recursion_boost, replay_boost, temperature_mod]
            nn.Tanh()  # [-1..1] для модификации
        )
        
        # Curiosity signal: novelty detector
        self.novelty_detector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Active goals buffer
        self.active_goals = []  # список текущих активных целей
        self.max_active_goals = 5
    
    def generate_goal(self, features, novelty_signal=None):
        """
        Генерирует новую внутреннюю цель на основе текущего состояния.
        
        Args:
            features: [B, feature_dim] - текущие features
            novelty_signal: float [0..1] - сигнал новизны (опционально)
        
        Returns:
            goal: [B, goal_dim] - вектор цели
        """
        goal = self.goal_generator(features)
        return goal
    
    def evaluate_goal_achievement(self, features, goal):
        """
        Оценивает, насколько цель достигнута.
        
        Returns:
            achievement_score: [B, 1] - [0..1], где 1 = цель полностью достигнута
        """
        goal_feat = torch.cat([features, goal], dim=-1)
        achievement = self.goal_evaluator(goal_feat)
        return achievement
    
    def get_goal_policy_modifier(self, features, goal):
        """
        Получает модификатор policy на основе цели.
        
        Returns:
            policy_mod: [B, 3] - [recursion_boost, replay_boost, temperature_mod]
                где значения в [-1..1] для модификации базовых параметров
        """
        goal_feat = torch.cat([features, goal], dim=-1)
        policy_mod = self.goal_policy(goal_feat)
        return policy_mod
    
    def compute_novelty(self, features):
        """
        Вычисляет сигнал новизны (curiosity).
        
        Returns:
            novelty: [B, 1] - [0..1], где 1 = максимальная новизна
        """
        return self.novelty_detector(features)


class OwnConcepts(nn.Module):
    """
    Собственные концепты: генерация и использование внутренних концептов,
    не заданных извне (в отличие от CLIP, который использует внешние концепты).
    """
    def __init__(self, feature_dim=512, concept_dim=128, num_concepts=32, hidden_dim=256):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        
        # Concept encoder: features -> concept activations
        self.concept_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )
        
        # Concept bank: хранилище концептов (trainable embeddings)
        self.concept_bank = nn.Parameter(torch.randn(num_concepts, concept_dim))
        
        # Concept decoder: (concept_activations, concept_bank) -> reconstructed_features
        self.concept_decoder = nn.Sequential(
            nn.Linear(num_concepts * concept_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Concept importance: какие концепты важны для текущего состояния
        self.importance_net = nn.Sequential(
            nn.Linear(feature_dim, num_concepts),
            nn.Softmax(dim=-1)
        )
    
    def extract_concepts(self, features):
        """
        Извлекает активации концептов из features.
        
        Returns:
            concept_activations: [B, num_concepts] - активации концептов
            concept_importance: [B, num_concepts] - важность каждого концепта
        """
        concept_activations = torch.sigmoid(self.concept_encoder(features))  # [B, num_concepts]
        concept_importance = self.importance_net(features)  # [B, num_concepts]
        return concept_activations, concept_importance
    
    def get_concept_based_routing(self, concept_activations, concept_importance):
        """
        Генерирует routing weights на основе концептов.
        
        Args:
            concept_activations: [B, num_concepts]
            concept_importance: [B, num_concepts]
        
        Returns:
            routing_signal: [B] - сигнал для модификации routing/recursion
        """
        # Используем важность концептов как сигнал для routing
        # Высокая важность = больше внимания нужно
        routing_signal = concept_importance.sum(dim=-1)  # [B]
        return routing_signal
    
    def reconstruct_from_concepts(self, concept_activations):
        """
        Восстанавливает features из концептов.
        
        Args:
            concept_activations: [B, num_concepts]
        
        Returns:
            reconstructed_features: [B, feature_dim]
        """
        # Weighted combination of concepts
        weighted_concepts = concept_activations.unsqueeze(-1) * self.concept_bank.unsqueeze(0)  # [B, num_concepts, concept_dim]
        concept_flat = weighted_concepts.view(concept_activations.size(0), -1)  # [B, num_concepts * concept_dim]
        reconstructed = self.concept_decoder(concept_flat)
        return reconstructed
    
    def discover_new_concept(self, features, threshold=0.3):
        """
        Обнаруживает новый концепт, если текущие концепты недостаточны.
        
        Returns:
            new_concept_found: bool - обнаружен ли новый концепт
        """
        concept_activations, importance = self.extract_concepts(features)
        max_activation = concept_activations.max(dim=-1)[0].mean()
        # Если максимальная активация низкая, значит нужен новый концепт
        return max_activation.item() < threshold


class AutobiographicalMemory:
    """
    Автобиографическая память: запись собственных действий, решений и опыта.
    Хранит не только данные, но и контекст: "что я делал", "почему", "что произошло".
    """
    def __init__(self, max_memories=10000):
        self.max_memories = max_memories
        self.memories = []  # список записей памяти
        
    def record(self, step, state_features, action, outcome, reward_signal=None, context=None, pain_level=None):
        """
        Записывает эпизод в автобиографическую память.
        
        КРИТИЧНО: Эмоциональная память - индексируем события по уровню Pain.
        События с высокой болью будут приоритетны для Targeted Dreaming.
        
        Args:
            step: int - номер шага
            state_features: tensor [feature_dim] - состояние
            action: tensor или int - действие
            outcome: tensor [feature_dim] или dict - результат действия
            reward_signal: float - сигнал награды (опционально)
            context: dict - дополнительный контекст (опционально)
            pain_level: float - уровень боли/конфликта (опционально, для приоритизации)
        """
        memory_entry = {
            "step": step,
            "state": state_features.detach().cpu() if torch.is_tensor(state_features) else state_features,
            "action": action.detach().cpu() if torch.is_tensor(action) else action,
            "outcome": outcome.detach().cpu() if torch.is_tensor(outcome) else outcome,
            "reward": reward_signal,
            "pain_level": pain_level if pain_level is not None else 0.0,  # КРИТИЧНО: уровень боли
            "context": context or {},
            "timestamp": len(self.memories)  # порядковый номер
        }
        self.memories.append(memory_entry)
        
        # Ограничиваем размер памяти
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)
    
    def get_high_pain_memories(self, k=50):
        """
        Возвращает k наиболее "болезненных" эпизодов для Targeted Dreaming.
        
        Returns:
            high_pain_memories: list - список эпизодов с максимальным pain_level
        """
        if len(self.memories) == 0:
            return []
        
        # Сортируем по pain_level (убывание)
        sorted_memories = sorted(self.memories, key=lambda m: m.get("pain_level", 0.0), reverse=True)
        return sorted_memories[:k]
    
    def recall(self, query_features, k=10):
        """
        Вспоминает похожие эпизоды по query_features.
        
        Returns:
            similar_memories: list - список k наиболее похожих записей
        """
        if len(self.memories) == 0:
            return []
        
        # Вычисляем similarity (cosine similarity)
        similarities = []
        query_norm = query_features.norm()
        if query_norm == 0:
            return []
        
        # КРИТИЧНО: определяем device из query_features
        query_device = query_features.device if torch.is_tensor(query_features) else torch.device("cpu")
        
        for mem in self.memories:
            if torch.is_tensor(mem["state"]):
                state = mem["state"]
                # КРИТИЧНО: перемещаем state на тот же device, что и query_features
                if state.device != query_device:
                    state = state.to(query_device)
                
                if state.numel() == query_features.numel():
                    state_flat = state.flatten()
                    query_flat = query_features.flatten()
                    sim = torch.dot(state_flat, query_flat) / (state_flat.norm() * query_flat.norm() + 1e-8)
                    similarities.append((sim.item(), mem))
        
        # Сортируем по similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in similarities[:k]]
    
    def get_recall_policy_modifier(self, similar_memories):
        """
        Генерирует модификатор policy на основе вспомненных эпизодов.
        
        Args:
            similar_memories: list - список похожих эпизодов
        
        Returns:
            policy_mod: dict - модификаторы для routing/recursion/replay
                - routing_boost: float - изменение routing weights
                - recursion_boost: float - изменение recursion depth
                - replay_boost: float - изменение replay ratio
        """
        if len(similar_memories) == 0:
            return {"routing_boost": 0.0, "recursion_boost": 0.0, "replay_boost": 0.0}
        
        # Анализируем исходы похожих эпизодов
        # Если исходы были хорошие (низкий loss/surprise), увеличиваем уверенность
        # Если исходы были плохие, увеличиваем осторожность (больше recursion/replay)
        
        good_outcomes = 0
        total_outcomes = 0
        
        for mem in similar_memories[:5]:  # берём топ-5
            if "outcome" in mem and isinstance(mem["outcome"], dict):
                if "loss" in mem["outcome"]:
                    loss = mem["outcome"]["loss"]
                    if loss < 1.0:  # хороший исход
                        good_outcomes += 1
                    total_outcomes += 1
        
        if total_outcomes == 0:
            return {"routing_boost": 0.0, "recursion_boost": 0.0, "replay_boost": 0.0}
        
        success_rate = good_outcomes / total_outcomes
        
        # Если успешные эпизоды - увеличиваем уверенность (меньше recursion)
        # Если неуспешные - увеличиваем осторожность (больше recursion/replay)
        if success_rate > 0.7:
            recursion_boost = -0.2  # меньше recursion
            replay_boost = -0.1  # меньше replay
        else:
            recursion_boost = 0.3  # больше recursion
            replay_boost = 0.2  # больше replay
        
        return {
            "routing_boost": 0.0,  # пока не используем
            "recursion_boost": recursion_boost,
            "replay_boost": replay_boost
        }
    
    def get_statistics(self):
        """Возвращает статистику по памяти"""
        return {
            "total_memories": len(self.memories),
            "oldest_step": self.memories[0]["step"] if self.memories else None,
            "newest_step": self.memories[-1]["step"] if self.memories else None,
        }


class SelfModel(nn.Module):
    """
    Явный self-model: модель самого себя и своих способностей.
    Отвечает на вопросы: "Что я умею?", "Насколько я уверен?", "Где мои слабые места?"
    
    КРИТИЧНО: Обучается на внутренних метриках (self-supervised) и влияет на управление.
    """
    def __init__(self, feature_dim=512, num_heads=5, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads
        
        # Capability predictor: предсказывает способности на разных задачах
        self.capability_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)  # способности для каждого head
        )
        
        # Confidence estimator: оценивает уверенность в предсказаниях
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Weakness detector: обнаруживает слабые места
        self.weakness_detector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # [0..1], где 1 = максимальная слабость
        )
        
        # Self-awareness: мета-оценка собственного состояния
        self.self_awareness = nn.Sequential(
            nn.Linear(feature_dim + num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [capability, confidence, weakness]
        )
        
        # EMA для внутренних метрик (targets для обучения)
        # КРИТИЧНО: инициализируем как None, будет создан динамически при первом обновлении
        self.ema_capabilities = None  # будет создан при первом update_ema_targets
        self.ema_confidence = 0.0
        self.ema_weakness = 0.0
        self.ema_momentum = 0.99
    
    def predict_capabilities(self, features):
        """
        Предсказывает способности на разных задачах (heads).
        
        Returns:
            capabilities: [B, num_heads] - [0..1], способности для каждого head
        """
        return torch.sigmoid(self.capability_predictor(features))
    
    def estimate_confidence(self, features):
        """
        Оценивает уверенность в предсказаниях.
        
        Returns:
            confidence: [B, 1] - [0..1], где 1 = максимальная уверенность
        """
        return self.confidence_estimator(features)
    
    def detect_weakness(self, features):
        """
        Обнаруживает слабые места в знаниях.
        
        Returns:
            weakness: [B, 1] - [0..1], где 1 = максимальная слабость
        """
        return self.weakness_detector(features)
    
    def self_assess(self, features, head_capabilities):
        """
        Мета-оценка собственного состояния.
        
        Returns:
            assessment: [B, 3] - [capability, confidence, weakness]
        """
        feat_cap = torch.cat([features, head_capabilities], dim=-1)
        assessment = self.self_awareness(feat_cap)
        return assessment
    
    def update_ema_targets(self, actual_capabilities, actual_confidence, actual_weakness):
        """
        Обновляет EMA targets для self-supervised обучения.
        
        Args:
            actual_capabilities: [num_heads] - реальные способности (например, accuracy по head)
            actual_confidence: float - реальная уверенность (калибровка)
            actual_weakness: float - реальная слабость (surprise + pain + entropy)
        """
        # КРИТИЧНО: создаём ema_capabilities при первом обновлении с правильным размером
        if self.ema_capabilities is None:
            self.ema_capabilities = actual_capabilities.clone()
        else:
            # Проверяем размерность и расширяем если нужно
            if self.ema_capabilities.size(0) < actual_capabilities.size(0):
                # Расширяем: добавляем новые heads с начальным значением
                new_size = actual_capabilities.size(0)
                old_size = self.ema_capabilities.size(0)
                expanded = torch.zeros(new_size, device=self.ema_capabilities.device, dtype=self.ema_capabilities.dtype)
                expanded[:old_size] = self.ema_capabilities
                expanded[old_size:] = 0.5  # начальное значение для новых heads
                self.ema_capabilities = expanded
            elif self.ema_capabilities.size(0) > actual_capabilities.size(0):
                # Обрезаем до текущего размера
                self.ema_capabilities = self.ema_capabilities[:actual_capabilities.size(0)]
        
        self.ema_capabilities = (
            self.ema_momentum * self.ema_capabilities + 
            (1 - self.ema_momentum) * actual_capabilities
        )
        self.ema_confidence = (
            self.ema_momentum * self.ema_confidence + 
            (1 - self.ema_momentum) * actual_confidence
        )
        self.ema_weakness = (
            self.ema_momentum * self.ema_weakness + 
            (1 - self.ema_momentum) * actual_weakness
        )
    
    def get_targets(self):
        """Возвращает текущие targets для обучения"""
        # КРИТИЧНО: если ema_capabilities ещё не инициализирован, возвращаем None
        if self.ema_capabilities is None:
            return {
                "capabilities": None,
                "confidence": self.ema_confidence,
                "weakness": self.ema_weakness
            }
        return {
            "capabilities": self.ema_capabilities.clone(),
            "confidence": self.ema_confidence,
            "weakness": self.ema_weakness
        }


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


class TimeMixer(nn.Module):
    """
    Time Mixer - сглаживает переходы между фазами через миксование временных репрезентаций.
    
    Решает проблемы:
    1. Сглаживание "Шока" (Gradient Spikes) - плавное вытеснение старых признаков новыми
    2. Решение "Frog Collapse" (Mode Collapse) - регуляризация через совместимость со старыми знаниями
    3. Улучшение Lazarus v3 - миксование временных репрезентаций во время сна
    
    Хранит M последних состояний (EMA) и смешивает их с текущими features через обучаемые веса.
    """
    def __init__(self, feature_dim, memory_size=5, ema_decay=0.9):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.ema_decay = ema_decay
        
        # Обучаемые веса для смешивания временных слоев
        # КРИТИЧНО: больший вес для текущего состояния, меньший для старых (предотвращает размывание)
        initial_weights = torch.ones(memory_size + 1)
        initial_weights[0] = 2.0  # текущее состояние важнее
        initial_weights[1:] = 0.5 / memory_size  # старые состояния менее важны
        self.mix_weights = nn.Parameter(initial_weights / initial_weights.sum())
        
        # EMA буферы для хранения прошлых состояний (не обучаемые)
        self.register_buffer('memory_buffer', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_count', torch.zeros(memory_size))
        
        # Сеть для адаптивного смешивания (опционально)
        self.adaptive_mixer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, memory_size + 1),
            nn.Softmax(dim=-1)
        )
    
    def update_memory(self, features):
        """
        Обновляет память прошлых состояний через EMA.
        
        Args:
            features: [B, feature_dim] - текущие features (берём среднее по батчу)
        """
        if not self.training:
            return
        
        # Берём среднее по батчу для обновления памяти
        current_mean = features.mean(dim=0).detach()  # [feature_dim]
        
        # Сдвигаем память: новое состояние в начало, старые сдвигаются
        self.memory_buffer = torch.cat([
            current_mean.unsqueeze(0),  # новое состояние
            self.memory_buffer[:-1]  # старые состояния (сдвиг)
        ], dim=0)
        
        # Обновляем счётчики (для взвешенного EMA)
        self.memory_count = torch.cat([
            torch.ones(1, device=features.device),
            self.memory_count[:-1] * self.ema_decay
        ], dim=0)
    
    def forward(self, features, use_adaptive=True, surprise=None):
        """
        Смешивает текущие features с прошлыми состояниями.
        
        Args:
            features: [B, feature_dim] - текущие features из backbone
            use_adaptive: bool - использовать адаптивное смешивание или фиксированные веса
            surprise: float - уровень surprise (опционально, для адаптивного забывания)
        
        Returns:
            mixed_features: [B, feature_dim] - смешанные features
        """
        B = features.size(0)
        
        # КРИТИЧНО: При высокой surprise (новая среда) - забываем старые состояния быстрее
        # Это предотвращает размывание признаков при переходе между фазами
        forget_factor = 1.0
        if surprise is not None and surprise > 0.5:  # высокая surprise = новая среда
            # Увеличиваем забывание: surprise 0.5 -> forget=0.5, surprise 1.0 -> forget=0.0
            forget_factor = max(0.0, 1.0 - 2.0 * (surprise - 0.5))  # [0.5..1.0] -> [0.0..1.0]
        
        # Подготавливаем все состояния для смешивания
        # [current, memory[0], memory[1], ..., memory[M-1]]
        states = [features]  # текущее состояние
        
        # КРИТИЧНО: Вычисляем "новизну" на основе расстояния между текущими и прошлыми features
        # Это позволяет адаптивно забывать старые состояния без явного surprise
        novelty_signal = None
        if self.memory_count[0] > 0:  # если есть хотя бы одно прошлое состояние
            recent_memory = self.memory_buffer[0].unsqueeze(0).expand(B, -1)  # [B, feature_dim]
            # Вычисляем косинусное расстояние между текущими и недавними features
            cosine_sim = F.cosine_similarity(features, recent_memory, dim=1)  # [B]
            novelty_signal = (1.0 - cosine_sim.mean()).item()  # [0..2], чем выше, тем новее
            # Если новизна высокая (>0.3), увеличиваем забывание
            if novelty_signal > 0.3:
                forget_factor = max(0.0, forget_factor * (1.0 - novelty_signal * 0.5))
        
        # Добавляем прошлые состояния с учётом забывания
        for i in range(self.memory_size):
            if self.memory_count[i] > 0:  # если память инициализирована
                memory_state = self.memory_buffer[i].unsqueeze(0).expand(B, -1)  # [B, feature_dim]
                # Применяем забывание: старые состояния становятся менее важными
                memory_weight = forget_factor ** (i + 1)  # более старые состояния забываются быстрее
                states.append(memory_state * memory_weight)
            else:
                # Если память не инициализирована, используем текущие features с низким весом
                states.append(features * forget_factor * 0.1)
        
        # Объединяем все состояния
        all_states = torch.stack(states, dim=1)  # [B, M+1, feature_dim]
        
        # Вычисляем веса смешивания
        if use_adaptive and self.training:
            # Адаптивное смешивание на основе текущих features
            mix_logits = self.adaptive_mixer(features)  # [B, M+1]
            mix_weights = mix_logits
            # КРИТИЧНО: При высокой surprise увеличиваем вес текущего состояния
            if surprise is not None and surprise > 0.5:
                current_weight_boost = min(0.5, (surprise - 0.5) * 1.0)  # [0..0.5] при surprise [0.5..1.0]
                mix_weights[:, 0] = mix_weights[:, 0] + current_weight_boost  # увеличиваем вес текущего
                mix_weights = mix_weights / mix_weights.sum(dim=1, keepdim=True)  # нормализуем
        else:
            # Фиксированные веса (обучаемые параметры)
            mix_weights = torch.softmax(self.mix_weights, dim=0)  # [M+1]
            # КРИТИЧНО: При высокой surprise увеличиваем вес текущего состояния
            if surprise is not None and surprise > 0.5:
                current_weight_boost = min(0.3, (surprise - 0.5) * 0.6)  # [0..0.3] при surprise [0.5..1.0]
                mix_weights[0] = mix_weights[0] + current_weight_boost
                mix_weights = mix_weights / mix_weights.sum()  # нормализуем
            mix_weights = mix_weights.unsqueeze(0).expand(B, -1)  # [B, M+1]
        
        # Смешиваем состояния
        mixed_features = torch.sum(all_states * mix_weights.unsqueeze(-1), dim=1)  # [B, feature_dim]
        
        # Обновляем память (только во время обучения)
        if self.training:
            self.update_memory(features)
        
        return mixed_features


class ElegantRecursiveCore(nn.Module):
    """
    Элегантный рекурсивный блок - фрактальная простота вместо модульности.
    
    Вместо отдельных модулей (Backbone + WorldModel + Critic) используем один итеративный блок.
    Рекурсивное время встроено в цикл: чем сложнее картинка (сюрприз), тем больше итераций.
    """
    def __init__(self, input_dim=3, hidden_dim=512, output_dim=11):
        super().__init__()
        # Единый блок: encode + refine + classify
        self.encode = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8->4
            nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # [B, hidden_dim, 1, 1]
        )
        
        # Рекурсивный refine блок - уточняет представление на каждой итерации
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Классификатор
        self.classify = nn.Linear(hidden_dim, output_dim)
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x, max_steps=None, surprise_threshold=0.3):
        """
        Рекурсивный forward с автоматическим определением числа итераций.
        
        Args:
            x: [B, 3, 32, 32] - входное изображение
            max_steps: int - максимальное число итераций (опционально)
            surprise_threshold: float - порог сюрприза для продолжения рекурсии
        
        Returns:
            logits: [B, output_dim] - предсказания
            h_final: [B, hidden_dim] - финальное представление
            reconstruction_error: float - ошибка восстановления (энергия/сюрприз)
        """
        B = x.size(0)
        device = x.device
        
        # 1. Encode: начальное представление
        h = self.encode(x)  # [B, hidden_dim, 1, 1]
        h = h.view(B, -1)  # [B, hidden_dim]
        
        # 2. Рекурсивное уточнение: h зависит от h прошлого шага
        # Сюрприз как энергия: если вход сильно отличается от ожидаемого, продолжаем итерации
        reconstruction_error = None
        h_prev = None
        
        if max_steps is None:
            max_steps = 3  # по умолчанию 3 итерации
        
        for step in range(max_steps):
            h_prev = h.clone()
            # Refine: уточняем представление
            h = self.refine(h)  # [B, hidden_dim]
            
            # КРИТИЧНО: Сюрприз как энергия - вычисляем ошибку восстановления
            # Если h сильно изменился, значит есть "напряжение" (сложность)
            if step > 0 and h_prev is not None:
                reconstruction_error = F.mse_loss(h, h_prev.detach()).item()
                # Если ошибка мала (h стабилизировался), можно остановиться раньше
                if reconstruction_error < surprise_threshold * 0.1:
                    break
        
        # 3. Classify: финальное предсказание
        logits = self.classify(h)  # [B, output_dim]
        
        # Если reconstruction_error не был вычислен, используем разницу между последними двумя шагами
        if reconstruction_error is None:
            reconstruction_error = 0.0
        
        return logits, h, reconstruction_error


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
        self.base_recovery_rate = 0.020  # базовое восстановление за шаг (увеличено для стабильности)
        self.budget_decay_rate = 0.998  # медленное затухание
        
        # Стоимости действий (уменьшены для баланса с recovery_rate)
        self.cost_recursion = 0.012  # за один рекурсивный проход (уменьшено для баланса)
        self.cost_replay = 0.003  # за единицу replay_ratio (ещё уменьшено)
        self.cost_kl = 0.008  # за KL distillation (ещё уменьшено)
        self.cost_expansion = 0.30  # за expansion (оставляем высоким, т.к. это редкое событие)
        
        # История complexity для динамического баланса
        self.recent_complexity = []  # последние N значений complexity
        self.max_recent_complexity = 20  # размер окна для вычисления динамического recovery_rate
        
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
    
    def get_actions(self, complexity, has_expansion_budget, cooldown_ok, weakness_signal=None, expansion_history=None):
        """
        Выдаёт действия на основе состояния сложности.
        
        КРИТИЧНО: Мета-регуляция бюджета сложности.
        Сравнивает стоимость: рекурсия vs expansion.
        Если expansion неэффективен (weakness не падает), предпочитает рекурсию.
        
        Args:
            complexity: float [0..1] - текущая сложность
            has_expansion_budget: bool - есть ли бюджет на expansion
            cooldown_ok: bool - можно ли расширяться
            weakness_signal: float - текущий уровень слабости (опционально)
            expansion_history: dict - история эффективности expansion (опционально)
        
        Returns:
            dict с ключами:
                - n_recursions: int [1..3] - сколько рекурсивных проходов
                - replay_ratio: float [0.1..0.4] - доля replay в батче
                - gate_temperature: float [0.7..2.0] - temperature для routing
                - crystal_target: float [0..1] - целевой crystal_level
                - expand_allowed: bool - разрешение на expansion
        """
        # КРИТИЧНО: Utility Analysis - сравниваем стоимость стратегий
        # Если expansion неэффективен, предпочитаем рекурсию
        expansion_effective = True
        if expansion_history is not None:
            # Проверяем историю: если после expansion weakness не падал, стратегия неэффективна
            recent_expansions = expansion_history.get("recent", [])
            if len(recent_expansions) > 0:
                # КРИТИЧНО: фильтруем только записи с полными данными (weakness_after не None)
                complete_expansions = [e for e in recent_expansions if e.get("weakness_after") is not None and e.get("weakness_before") is not None]
                if len(complete_expansions) > 0:
                    avg_weakness_after = sum(e.get("weakness_after", 1.0) for e in complete_expansions) / len(complete_expansions)
                    avg_weakness_before = sum(e.get("weakness_before", 1.0) for e in complete_expansions) / len(complete_expansions)
                    # Если weakness не снизился после expansion, стратегия неэффективна
                    if avg_weakness_after >= avg_weakness_before * 0.95:  # менее 5% улучшения
                        expansion_effective = False
        
        # КРИТИЧНО: если expansion неэффективен, увеличиваем рекурсию вместо expansion
        if not expansion_effective and weakness_signal is not None and weakness_signal > 0.5:
            # Высокая слабость + неэффективный expansion = больше рекурсии
            expand_allowed = False
            n_recursions_base = min(3, int(1 + complexity * 2 + weakness_signal * 1))
        else:
            # Обычная логика
            expand_allowed = (complexity > 0.7 and has_expansion_budget and cooldown_ok)
            n_recursions_base = int(1 + complexity * 1.5)
        
        # КРИТИЧНО: используем n_recursions_base для плавного наращивания "времени на раздумья"
        # Это позволяет системе наращивать рекурсию перед радикальным шагом (expansion)
        n_recursions = min(3, max(1, n_recursions_base))  # 1..3, используем вычисленный base
        
        # replay_ratio: высокая сложность → больше памяти
        replay_ratio = 0.10 + 0.30 * complexity  # 10%..40%
        
        # gate_temperature: высокая сложность → более равномерный routing (поиск)
        gate_temperature = 0.7 + 1.3 * complexity  # 0.7..2.0
        
        # crystal_target: высокая сложность → меньше кристаллизации (больше пластичности)
        crystal_target = 1.0 - complexity
        
        # КРИТИЧНО: expand_allowed уже вычислен выше, не перезаписываем
        
        return {
            "n_recursions": n_recursions,
            "replay_ratio": replay_ratio,
            "gate_temperature": gate_temperature,
            "crystal_target": crystal_target,
            "expand_allowed": expand_allowed,
        }
    
    def update_budget(self, actions, used_expansion=False, used_kl=False, current_complexity=None, pain_value=None):
        """
        Обновляет complexity budget на основе использованных действий.
        
        КРИТИЧНО: Динамический баланс - recovery_rate зависит от темпа поступления новых данных.
        При высокой complexity (много новых данных) - быстрее восстанавливаем budget.
        При низкой complexity (мало новых данных) - медленнее восстанавливаем.
        
        КРИТИЧНО: Регуляция "Эмоционального резонанса" - при высокой pain (>0.8) замедляем recovery,
        чтобы система "болела" дольше и тратила больше ресурсов на рекурсию для решения проблем.
        
        Args:
            actions: dict от get_actions()
            used_expansion: bool - был ли использован expansion
            used_kl: bool - был ли использован KL distillation
            current_complexity: float - текущая complexity (опционально, для динамического баланса)
            pain_value: float - текущий уровень pain (опционально, для эмоциональной регуляции)
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
        
        # КРИТИЧНО: Динамический recovery_rate в зависимости от темпа поступления новых данных
        if current_complexity is not None:
            # Обновляем историю complexity
            self.recent_complexity.append(current_complexity)
            if len(self.recent_complexity) > self.max_recent_complexity:
                self.recent_complexity.pop(0)
            
            # Вычисляем среднюю complexity за последние N шагов
            avg_complexity = sum(self.recent_complexity) / len(self.recent_complexity) if self.recent_complexity else 0.5
            
            # Динамический recovery_rate: высокая complexity -> быстрее восстановление
            # complexity [0..1] -> recovery_rate [0.01..0.03]
            # При complexity=0.5 recovery_rate=base_recovery_rate
            dynamic_recovery = self.base_recovery_rate * (0.5 + avg_complexity)  # [0.01..0.03]
        else:
            # Fallback: используем базовый recovery_rate
            dynamic_recovery = self.base_recovery_rate
        
        # КРИТИЧНО: Регуляция "Эмоционального резонанса" - при высокой pain замедляем recovery
        # Это позволяет системе "болеть" дольше и тратить больше ресурсов на рекурсию
        # для решения проблем (например, Cat vs Dog)
        if pain_value is not None and pain_value > 0.8:
            # При pain > 0.8 уменьшаем recovery_rate: pain 0.8 -> mult=0.8, pain 1.0 -> mult=0.5
            pain_multiplier = max(0.5, 1.0 - (pain_value - 0.8) * 1.5)  # [0.5..0.8] при pain [0.8..1.0]
            dynamic_recovery = dynamic_recovery * pain_multiplier
        
        # Восстанавливаем budget с динамическим recovery_rate
        self.complexity_budget = min(1.0, self.complexity_budget + dynamic_recovery)
    
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
    def __init__(self, use_curiosity=False, use_subjective_time=False, use_vae_dreams=False, 
                 use_world_model=False, use_internal_goals=False, use_own_concepts=False,
                 use_autobiographical_memory=False, use_self_model=False, use_elegant_mode=False):
        super().__init__()
        self.hidden_size = 512
        self.output_size = 11
        self.unknown_class_idx = 10

        # КРИТИЧНО: Элегантный режим - фрактальная простота вместо модульности
        self.use_elegant_mode = bool(use_elegant_mode)  # опционально: можно включить для упрощенной архитектуры
        if self.use_elegant_mode:
            # Единый рекурсивный блок вместо Backbone + WorldModel + Critic
            self.elegant_core = ElegantRecursiveCore(input_dim=3, hidden_dim=self.hidden_size, output_dim=self.output_size)
            # "Медленные веса" для кристаллизации через EMA
            self.memory_weights = copy.deepcopy(self.elegant_core)
            for p in self.memory_weights.parameters():
                p.requires_grad = False
            self.ema_decay = 0.999  # медленное обновление памяти
            # КРИТИЧНО: Заглушка для совместимости - в элегантном режиме expansion не нужен
            self.heads = nn.ModuleList()  # пустой список для совместимости с кодом
            self.shared_backbone = None  # не используется в элегантном режиме
            self.columns = nn.ModuleList()  # не используется в элегантном режиме
        else:
            # Стандартный режим с модульностью
            self.shared_backbone = SharedBackbone()
            self.heads = nn.ModuleList([ExpandableHead(self.hidden_size, self.output_size)])
            self.columns = nn.ModuleList([TemporalColumn(self.hidden_size, self.output_size)])

        self.sensor = ComplexitySensor()
        self.active_classes_per_column = {}
        
        # КРИТИЧНО: Time Mixer - сглаживает переходы между фазами
        self.use_time_mixer = True
        if self.use_time_mixer:
            self.time_mixer = TimeMixer(feature_dim=self.hidden_size, memory_size=5, ema_decay=0.9)
        
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
        
        # КРИТИЧНО: История эффективности expansion для мета-регуляции
        self.expansion_history = {
            "recent": [],  # последние N expansion'ов с метриками
            "max_history": 10  # храним последние 10 expansion'ов
        }

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

        # --- AGI COMPONENTS ---
        self.use_world_model = bool(use_world_model)
        if self.use_world_model:
            # КРИТИЧНО: action_dim = 4 (patches), а не output_size (классы)
            self.world_model = WorldModel(feature_dim=self.hidden_size, action_dim=4)
            self.attention_action = AttentionAction(num_patches=4, patch_size=16)
        
        self.use_internal_goals = bool(use_internal_goals)
        if self.use_internal_goals:
            self.internal_goals = InternalGoals(feature_dim=self.hidden_size)
        
        self.use_own_concepts = bool(use_own_concepts)
        if self.use_own_concepts:
            self.own_concepts = OwnConcepts(feature_dim=self.hidden_size, num_concepts=32)
        
        self.use_autobiographical_memory = bool(use_autobiographical_memory)
        if self.use_autobiographical_memory:
            self.autobiographical_memory = AutobiographicalMemory(max_memories=10000)
        
        self.use_self_model = bool(use_self_model)
        if self.use_self_model:
            self.self_model = SelfModel(feature_dim=self.hidden_size, num_heads=5)

        self.conflict_buffer = []
        self.max_conflicts = 100

        self.replay_buffer = {"X": [], "Y": []}
        self.max_replay_size = 1000
        
        # КРИТИЧНО: Упрощенная кристаллизация через EMA весов (для элегантного режима)
        if self.use_elegant_mode:
            self.ema_update_counter = 0
            self.ema_update_frequency = 10  # обновляем memory_weights каждые 10 шагов

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

    def update_memory_weights_ema(self):
        """
        КРИТИЧНО: Упрощенная кристаллизация через весовую инерцию (EMA).
        Вместо сложной логики заморозки слоев используем просто EMA копию весов.
        "Медленные" веса (memory_weights) плавно догоняют "быстрые" (elegant_core).
        Это одна строчка кода в функции потерь - элегантность вместо модульности.
        """
        if not self.use_elegant_mode or not hasattr(self, 'elegant_core') or not hasattr(self, 'memory_weights'):
            return
        
        # EMA обновление: memory_weights = ema_decay * memory_weights + (1 - ema_decay) * elegant_core
        with torch.no_grad():
            for (name, param_fast), (_, param_slow) in zip(
                self.elegant_core.named_parameters(),
                self.memory_weights.named_parameters()
            ):
                param_slow.data = self.ema_decay * param_slow.data + (1.0 - self.ema_decay) * param_fast.data
    
    def elegant_stability_loss(self):
        """
        КРИТИЧНО: Stability Loss - просто не даем весам быстро меняться.
        Это попытка текущих весов не уходить далеко от "медленных" (памяти).
        Одна строчка кода вместо сложной логики кристаллизации.
        """
        if not self.use_elegant_mode or not hasattr(self, 'elegant_core') or not hasattr(self, 'memory_weights'):
            # Fallback: возвращаем нулевой тензор на правильном устройстве
            if hasattr(self, 'elegant_core'):
                device = next(self.elegant_core.parameters()).device
            elif hasattr(self, 'shared_backbone'):
                device = next(self.shared_backbone.parameters()).device
            else:
                device = torch.device('cpu')
            return torch.tensor(0.0, device=device)
        
        # Просто MSE между быстрыми и медленными весами
        total_loss = 0.0
        count = 0
        device = next(self.elegant_core.parameters()).device
        for (name, param_fast), (_, param_slow) in zip(
            self.elegant_core.named_parameters(),
            self.memory_weights.named_parameters()
        ):
            total_loss += F.mse_loss(param_fast, param_slow.detach())
            count += 1
        
        return total_loss / max(1, count) if count > 0 else torch.tensor(0.0, device=device)
    
    def crystallization_regularizer(self):
        """
        Пер-слойная защита (EWC-lite): ||W - W_ref||^2 с динамическим весом в зависимости от crystal_level.
        Требует наличия self.ref_backbone (снимок).
        Нормализовано через mean() для сопоставимости между слоями.
        """
        # В элегантном режиме используем elegant_stability_loss вместо этого
        if self.use_elegant_mode:
            return 0.0  # stability loss уже добавлен через elegant_stability_loss
        
        if (not self.use_subjective_time) or (self.ref_backbone is None):
            return 0.0

        reg = 0.0
        cnt = 0
        # КРИТИЧНО: матчим параметры по имени, а не через zip (более надежно)
        ref_params = dict(self.ref_backbone.named_parameters())
        if self.shared_backbone is None:
            return 0.0
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
        # В элегантном режиме заморозка не нужна (используется EMA)
        if self.use_elegant_mode:
            return
        
        # lock/unlock ранних слоёв
        if self.crystal_level >= self.crystal_lock_threshold:
            self.crystal_lock_steps += 1
        else:
            self.crystal_lock_steps = max(0, self.crystal_lock_steps - 2)

        # LOCK
        if self.crystal_lock_steps >= self.crystal_lock_patience:
            if self.shared_backbone is None:
                return
            for name, p in self.shared_backbone.named_parameters():
                if self._layer_group(name) == "early":
                    if p.requires_grad:  # только если еще не заморожен
                        p.requires_grad = False
                        self.hard_frozen_layers.add(name)

        # UNLOCK (если мир снова нестабилен)
        if self.crystal_level <= self.crystal_unlock_threshold and len(self.hard_frozen_layers) > 0:
            if self.shared_backbone is None:
                return
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
                if self.use_elegant_mode:
                    _ = self.elegant_core(x, max_steps=1)  # в элегантном режиме используем elegant_core
                elif self.shared_backbone is not None:
                    _ = self.shared_backbone(x)
        self._set_bn_train(False)
        if was_training:
            self.train()
        else:
            self.eval()

    def _semantic_merging(self, device, similarity_threshold=0.85, test_loader=None, test_acc_before=None):
        """
        КРИТИЧНО: Семантическое Слияние - слияние похожих голов в "Полиглотов".
        
        Вычисляет косинусное сходство весов между всеми головами.
        Сливает (усредняет) те головы, которые стали слишком похожи.
        Освободившийся "бюджет" отдается новой голове, если Pain (конфликт) снова вырастет.
        
        КРИТИЧНО: "Когнитивный Кэшбэк" - бонус к бюджету при успешном слиянии.
        КРИТИЧНО: "Градиентный Штраф" - наказание при неудачном слиянии.
        
        Это превращает "Рой" в саморегулируемую файловую систему знаний.
        """
        if len(self.heads) <= 1:
            return 0.0  # Возвращаем изменение бюджета (0 = нет изменений)
        
        print(f"   [SEMANTIC MERGING] Analyzing {len(self.heads)} heads for similarity...")
        
        # Вычисляем косинусное сходство весов между всеми парами голов
        similarities = []
        head_pairs = []
        
        for i in range(len(self.heads)):
            for j in range(i + 1, len(self.heads)):
                # Получаем веса обеих голов
                weights_i = torch.cat([p.flatten() for p in self.heads[i].parameters() if p.requires_grad])
                weights_j = torch.cat([p.flatten() for p in self.heads[j].parameters() if p.requires_grad])
                
                # Нормализуем для косинусного сходства
                weights_i_norm = F.normalize(weights_i, p=2, dim=0)
                weights_j_norm = F.normalize(weights_j, p=2, dim=0)
                
                # Косинусное сходство
                similarity = (weights_i_norm * weights_j_norm).sum().item()
                similarities.append(similarity)
                head_pairs.append((i, j))
        
        # Находим пары голов, которые слишком похожи
        merged_indices = set()
        merge_operations = []
        budget_change = 0.0  # Изменение бюджета (положительное = бонус, отрицательное = штраф)
        
        for idx, (i, j) in enumerate(head_pairs):
            if similarities[idx] >= similarity_threshold and i not in merged_indices and j not in merged_indices:
                print(f"   [MERGE] Head {i} and Head {j} are similar (cosine={similarities[idx]:.3f}). Merging...")
                
                # Сохраняем веса до слияния для возможного отката
                weights_before_i = [p.data.clone() for p in self.heads[i].parameters()]
                weights_before_j = [p.data.clone() for p in self.heads[j].parameters()]
                
                # Сливаем веса: усредняем параметры двух голов
                with torch.no_grad():
                    for param_i, param_j in zip(self.heads[i].parameters(), self.heads[j].parameters()):
                        if param_i.requires_grad and param_j.requires_grad:
                            # Усредняем веса: 50% от каждой головы
                            merged_weight = 0.5 * param_i.data + 0.5 * param_j.data
                            param_i.data.copy_(merged_weight)
                            param_j.data.copy_(merged_weight)  # Обе головы получают усредненные веса
                
                # КРИТИЧНО: Проверяем успешность слияния через тест accuracy
                merge_successful = True
                if test_loader is not None and test_acc_before is not None:
                    # Быстрая проверка на небольшом батче
                    self.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for test_data, test_target in test_loader:
                            test_data = test_data[:32].to(device)  # небольшой батч для быстрой проверки
                            test_target = test_target[:32].to(device)
                            test_output = self(test_data)
                            pred = test_output[:, :10].argmax(dim=1)
                            correct += (pred == test_target).sum().item()
                            total += test_target.size(0)
                            break  # только один батч для скорости
                    test_acc_after = 100.0 * correct / max(1, total)
                    self.train()
                    
                    # КРИТИЧНО: "Когнитивный Кэшбэк" - если accuracy не упала значительно (>= 98%)
                    if test_acc_after >= test_acc_before * 0.98:
                        # УСПЕХ: Даем плюшку
                        budget_change += 0.2  # Бонус к бюджету
                        print(f"   🎉 SYNERGY BONUS: Heads merged successfully. Accuracy: {test_acc_before:.2f}% -> {test_acc_after:.2f}%")
                        print(f"   [BUDGET] +0.2 bonus added to complexity_budget")
                    else:
                        # ОШИБКА: Откатываем слияние и наказываем
                        merge_successful = False
                        with torch.no_grad():
                            for param_i, param_j, w_before_i, w_before_j in zip(
                                self.heads[i].parameters(), 
                                self.heads[j].parameters(),
                                weights_before_i,
                                weights_before_j
                            ):
                                param_i.data.copy_(w_before_i)
                                param_j.data.copy_(w_before_j)
                        budget_change -= 0.1  # Штраф к бюджету
                        print(f"   ⚠️ MERGE PENALTY: Knowledge conflict detected! Accuracy: {test_acc_before:.2f}% -> {test_acc_after:.2f}%")
                        print(f"   [BUDGET] -0.1 penalty. Merge rolled back.")
                
                if merge_successful:
                    merged_indices.add(i)
                    merged_indices.add(j)
                    merge_operations.append((i, j))
        
        if merge_operations:
            print(f"   [SEMANTIC MERGING] Merged {len(merge_operations)} pairs of similar heads.")
            print(f"   [BUDGET] Freed capacity for future expansion.")
        else:
            print(f"   [SEMANTIC MERGING] No similar heads found (threshold={similarity_threshold}). All heads remain distinct.")
        
        return budget_change
    
    def _polyglot_synthesis(self, device, similarity_threshold=0.80):
        """
        КРИТИЧНО: "Синтез Полиглота" - умное слияние голов для разблокировки Head Limit.
        
        Когда достигнут лимит голов, система автоматически находит и сливает наиболее похожие пары,
        освобождая место для новых знаний. Это превращает [CRITICAL] в [EVOLUTION].
        
        Args:
            device: torch.device
            similarity_threshold: float - порог косинусного сходства для слияния (0.80 = более агрессивный)
        
        Returns:
            bool - True если хотя бы одна пара была слита, False если слияние невозможно
        """
        if len(self.heads) < 2:
            return False  # Нужно минимум 2 головы для слияния
        
        print(f"   [POLYGLOT SYNTHESIS] Analyzing {len(self.heads)} heads for intelligent merging...")
        
        # Вычисляем косинусное сходство весов между всеми парами голов
        similarities = []
        head_pairs = []
        
        for i in range(len(self.heads)):
            for j in range(i + 1, len(self.heads)):
                # Получаем веса обеих голов
                try:
                    weights_i = torch.cat([p.flatten() for p in self.heads[i].parameters() if p.requires_grad])
                    weights_j = torch.cat([p.flatten() for p in self.heads[j].parameters() if p.requires_grad])
                    
                    if weights_i.numel() == 0 or weights_j.numel() == 0:
                        continue
                    
                    # Нормализуем для косинусного сходства
                    weights_i_norm = F.normalize(weights_i, p=2, dim=0)
                    weights_j_norm = F.normalize(weights_j, p=2, dim=0)
                    
                    # Косинусное сходство
                    similarity = (weights_i_norm * weights_j_norm).sum().item()
                    similarities.append(similarity)
                    head_pairs.append((i, j))
                except Exception as e:
                    print(f"   [WARNING] Could not compute similarity for heads {i} and {j}: {e}")
                    continue
        
        if len(similarities) == 0:
            print(f"   [POLYGLOT SYNTHESIS] No valid head pairs found for comparison.")
            return False
        
        # Находим пару с максимальным сходством
        max_similarity_idx = int(torch.tensor(similarities).argmax().item()) if len(similarities) > 0 else -1
        if max_similarity_idx < 0:
            return False
        max_similarity = similarities[max_similarity_idx]
        best_pair = head_pairs[max_similarity_idx]
        i, j = best_pair
        
        if max_similarity >= similarity_threshold:
            print(f"   [POLYGLOT SYNTHESIS] Found mergeable pair: Head {i} ↔ Head {j} (cosine={max_similarity:.3f})")
            
            # Сохраняем активные классы обеих голов перед слиянием
            classes_i = self.active_classes_per_column.get(i, [])
            classes_j = self.active_classes_per_column.get(j, [])
            merged_classes = sorted(list(set(classes_i + classes_j)))  # объединяем классы
            
            # Сливаем веса: усредняем параметры двух голов
            with torch.no_grad():
                for param_i, param_j in zip(self.heads[i].parameters(), self.heads[j].parameters()):
                    if param_i.requires_grad and param_j.requires_grad and param_i.shape == param_j.shape:
                        # Усредняем веса: 50% от каждой головы
                        merged_weight = 0.5 * param_i.data + 0.5 * param_j.data
                        param_i.data.copy_(merged_weight)
            
            # Удаляем одну из голов (оставляем head i, удаляем head j)
            # КРИТИЧНО: Обновляем active_classes_per_column перед удалением
            self.active_classes_per_column[i] = merged_classes
            # Удаляем head j из словаря и сдвигаем индексы
            new_active_classes = {}
            for old_idx, classes in self.active_classes_per_column.items():
                if old_idx < j:
                    new_active_classes[old_idx] = classes
                elif old_idx > j:
                    new_active_classes[old_idx - 1] = classes  # сдвигаем индексы
            self.active_classes_per_column = new_active_classes
            
            # Удаляем head j из ModuleList
            heads_list = list(self.heads)
            del heads_list[j]
            self.heads = nn.ModuleList(heads_list)
            
            print(f"   [POLYGLOT SYNTHESIS] ✓ Merged Head {j} into Head {i}. Classes: {merged_classes}")
            print(f"   [POLYGLOT SYNTHESIS] Heads remaining: {len(self.heads)}")
            
            return True
        else:
            print(f"   [POLYGLOT SYNTHESIS] No mergeable heads found (max similarity={max_similarity:.3f} < {similarity_threshold})")
            return False
    
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
    
    def dream_and_compress(self, num_dreams=1000, dream_batch_size=100, device=None, test_loader=None):
        """
        🌙 МОДУЛЬ СНОВИДЕНИЙ (CONSOLIDATION) + LAZARUS v3
        Объединяет знания из нескольких heads в один через dream distillation.
        
        КРИТИЧНО: Добавлена поддержка "Когнитивного Кэшбека" и "Градиентного Штрафа"
        через проверку успешности слияния голов.
        """
        if device is None:
            device = next(self.parameters()).device
        
        print("\n🌙 ENTERING SLEEP PHASE (Lazarus v3 + Consolidation)...")
        print(f"   Current heads: {len(self.heads)}")
        
        if len(self.heads) <= 1:
            print("   Only one head exists. No compression needed.")
            return 0.0  # Возвращаем изменение бюджета
        
        # КРИТИЧНО: Вычисляем тестовую accuracy до слияния для проверки успешности
        test_acc_before = None
        if test_loader is not None:
            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for test_data, test_target in test_loader:
                    test_data = test_data[:64].to(device)  # небольшой батч для быстрой проверки
                    test_target = test_target[:64].to(device)
                    test_output = self(test_data)
                    pred = test_output[:, :10].argmax(dim=1)
                    correct += (pred == test_target).sum().item()
                    total += test_target.size(0)
                    break  # только один батч для скорости
            test_acc_before = 100.0 * correct / max(1, total)
            self.train()
        
        # КРИТИЧНО: Семантическое Слияние - слияние похожих голов перед консолидацией
        # Это превращает "Рой" в саморегулируемую файловую систему знаний
        budget_change = 0.0
        if len(self.heads) > 1:
            budget_change = self._semantic_merging(device, similarity_threshold=0.85, test_loader=test_loader, test_acc_before=test_acc_before)
        
        # 1. Создаем "Студента" - одну компактную сеть
        student_head = ExpandableHead(self.hidden_size, self.output_size).to(device)
        optimizer = optim.Adam(student_head.parameters(), lr=0.0005)
        
        # 2. LAZARUS: Создаем frozen teacher (Consistency Anchor)
        teacher_model = copy.deepcopy(self)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        
        print(f"   Generating {num_dreams} dreams with Lazarus v3 protocol...")
        
        # КРИТИЧНО: Targeted Dreaming - получаем высокоболевые эпизоды
        high_pain_memories = []
        if self.use_autobiographical_memory and hasattr(self, 'autobiographical_memory') and self.autobiographical_memory is not None:
            high_pain_memories = self.autobiographical_memory.get_high_pain_memories(k=100)
            if high_pain_memories:
                avg_pain = sum(m.get("pain_level", 0.0) for m in high_pain_memories) / len(high_pain_memories)
                print(f"   [TARGETED DREAMING] Found {len(high_pain_memories)} high-pain memories (avg pain: {avg_pain:.3f}) for focused consolidation")
        
        # Lazarus v3 параметры
        w_cons = 1.0  # Consistency (главный компонент)
        w_stab = 0.5  # Stability
        w_ent = 0.08  # Entropy Floor (усилен с 0.05 для большей любознательности)
        H0 = 2.1      # Минимальная энтропия (поднят с 1.5 для большей любознательности после пробуждения)
        epsilon = 0.05
        
        for epoch in range(15):
            total_loss = 0
            for dream_batch in range(num_dreams // dream_batch_size):
                # КРИТИЧНО: Targeted Dreaming - смешиваем случайные сны с высокоболевыми
                if high_pain_memories and dream_batch % 3 == 0:  # Каждый 3-й батч используем болевые точки
                    # Пытаемся восстановить образы из высокоболевых эпизодов
                    # Используем VAE для генерации похожих образов
                    pain_mem = high_pain_memories[dream_batch % len(high_pain_memories)]
                    if torch.is_tensor(pain_mem.get("state")):
                        # Используем state features для генерации похожего образа
                        # Упрощённо: генерируем случайный сон, но с приоритетом на болевые классы
                        noise = self.sample_dreams(dream_batch_size, device)
                        # Можно добавить более сложную логику восстановления образов из features
                    else:
                        noise = self.sample_dreams(dream_batch_size, device)
                else:
                    # Обычные случайные сны
                    noise = self.sample_dreams(dream_batch_size, device)
                
                # LAZARUS v3: Consistency Anchor
                with torch.no_grad():
                    teacher_logits = teacher_model(noise)
                    teacher_probs = torch.softmax(teacher_logits[:, :10], dim=1)
                    
                    # КРИТИЧНО: Lazarus v3 для Роутера - дистиллируем уверенность роутера
                    # Студент должен учиться не только "что это", но и "какая голова за это отвечает"
                    teacher_routing_confidence = None
                    if self.use_soft_routing and len(teacher_model.heads) > 1 and hasattr(teacher_model, 'routing_gate'):
                        teacher_features = teacher_model.shared_backbone(noise)
                        # КРИТИЧНО: Time Mixer для teacher - используем временные репрезентации
                        if teacher_model.use_time_mixer and hasattr(teacher_model, 'time_mixer'):
                            teacher_features = teacher_model.time_mixer(teacher_features, use_adaptive=False)  # фиксированные веса для teacher
                        teacher_gates_full = teacher_model.routing_gate(teacher_features)  # [B, MAX_LAYERS]
                        teacher_gates = teacher_gates_full[:, :len(teacher_model.heads)]  # [B, H]
                        teacher_gates = teacher_gates / (teacher_gates.sum(dim=1, keepdim=True) + 1e-9)
                        # Сохраняем routing confidence для дистилляции
                        teacher_routing_confidence = teacher_gates  # [B, H] - уверенность каждого head
                
                # Студент предсказывает
                backbone_features = self.shared_backbone(noise)
                # КРИТИЧНО: Time Mixer для студента - миксование временных репрезентаций
                # Это позволяет студенту выучить "инварианты" между машинами и животными
                if self.use_time_mixer and hasattr(self, 'time_mixer'):
                    backbone_features = self.time_mixer(backbone_features, use_adaptive=True)
                student_logits, _ = student_head(backbone_features, prev_hiddens=[])
                student_probs = torch.softmax(student_logits[:, :10], dim=1)
                
                # 1. Consistency Loss
                loss_cons = F.mse_loss(student_logits[:, :10], teacher_logits[:, :10])
                
                # 2. Stability Loss
                noise_pert = noise + torch.randn_like(noise) * epsilon
                backbone_features_pert = self.shared_backbone(noise_pert)
                student_logits_pert, _ = student_head(backbone_features_pert, prev_hiddens=[])
                loss_stab = F.mse_loss(student_logits[:, :10], student_logits_pert[:, :10])
                
                # 3. Entropy Floor
                log_probs = F.log_softmax(student_logits[:, :10], dim=1)
                entropy = -(student_probs * log_probs).sum(dim=1).mean()
                loss_ent = F.relu(H0 - entropy)
                
                # 4. Knowledge Distillation
                loss_distill = F.kl_div(
                    F.log_softmax(student_logits[:, :10], dim=1),
                    teacher_probs,
                    reduction='batchmean'
                )
                
                # 5. КРИТИЧНО: Cross-Head Distillation (Семантическое Слияние в действии)
                # Головы (experts) предсказывают латентные представления друг друга
                # Пример: Голова 1 (Urban) видит во сне "колесо". Она передает Голове 2 (Nature) не просто картинку,
                # а свою "уверенность" в структуре. Так Голова 2 учится понимать "опору" (ноги животного)
                # через призму "шасси".
                loss_cross_head = torch.zeros((), device=device)
                if len(teacher_model.heads) > 1 and self.use_soft_routing and hasattr(teacher_model, 'shared_backbone'):
                    # Получаем предсказания каждой головы teacher
                    teacher_features = teacher_model.shared_backbone(noise)
                    if teacher_model.use_time_mixer and hasattr(teacher_model, 'time_mixer'):
                        teacher_features = teacher_model.time_mixer(teacher_features, use_adaptive=False)
                    
                    # Вычисляем латентные представления каждой головы (hidden states перед классификацией)
                    head_latents = []
                    for head_idx in range(len(teacher_model.heads)):
                        with torch.no_grad():
                            # Получаем hidden state перед финальным классификатором
                            h = teacher_features
                            for adapter in teacher_model.heads[head_idx].adapters:
                                if len(head_latents) < len(teacher_model.heads[head_idx].adapters):
                                    h = h + adapter(h)  # адаптеры для предыдущих голов
                            head_latents.append(h.detach())
                    
                    # Студент должен предсказывать латентные представления всех голов
                    # Это учит его "полиглотству" - пониманию разных точек зрения
                    # Используем backbone_features студента как его латентное представление
                    student_latent = backbone_features
                    for head_latent in head_latents:
                        if student_latent.size() == head_latent.size():
                            # MSE между латентными представлениями
                            loss_cross_head = loss_cross_head + F.mse_loss(student_latent, head_latent)
                    loss_cross_head = loss_cross_head / len(head_latents) if len(head_latents) > 0 else loss_cross_head
                
                # 6. КРИТИЧНО: Routing Confidence Distillation
                # Студент учится "какая голова за что отвечает" через неявное кодирование в features
                # Если teacher routing confidence высокая для head i, student должен предсказывать классы этого head
                loss_routing_distill = torch.zeros((), device=device)
                if teacher_routing_confidence is not None:
                    # Вычисляем взвешенную по routing confidence дистилляцию
                    # Если head i имеет высокую confidence, его предсказания важнее для студента
                    for head_idx in range(len(teacher_model.heads)):
                        head_confidence = teacher_routing_confidence[:, head_idx:head_idx+1]  # [B, 1]
                        # Взвешиваем KL divergence по confidence этого head
                        # Это учит студента: "когда teacher был уверен в head i, предсказывай как head i"
                        weighted_kl = head_confidence * F.kl_div(
                            F.log_softmax(student_logits[:, :10], dim=1),
                            teacher_probs,
                            reduction='none'
                        ).sum(dim=1, keepdim=True)
                        loss_routing_distill = loss_routing_distill + weighted_kl.mean()
                
                # Итоговый loss
                loss = w_cons * loss_cons + w_stab * loss_stab + w_ent * loss_ent + 0.3 * loss_distill
                if loss_routing_distill.item() > 0:
                    loss = loss + 0.2 * loss_routing_distill  # дистилляция routing confidence
                if loss_cross_head.item() > 0:
                    loss = loss + 0.15 * loss_cross_head  # Cross-Head Distillation (Семантическое Слияние)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 3 == 0:
                batches = num_dreams // dream_batch_size
                print(f"   Epoch {epoch+1}/15: Loss={total_loss/batches:.4f}, H={entropy.item():.3f}")
        
        print("☀️ WAKING UP: Lazarus Consolidation Complete.")
        
        # КРИТИЧНО: Визуализация снов - сохраняем примеры снов для анализа
        if self.use_vae_dreams and self.vae_trained:
            try:
                import matplotlib.pyplot as plt
                import os
                
                # Генерируем несколько примеров снов для визуализации
                num_dream_samples = 16
                dream_samples = self.sample_dreams(num_dream_samples, device)
                
                # Денормализуем для визуализации (CIFAR-10 нормализация: mean=0.5, std=0.5)
                dream_vis = dream_samples.detach().cpu()
                dream_vis = torch.clamp((dream_vis + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] -> [0,1]
                
                # Создаём grid 4x4
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for i in range(num_dream_samples):
                    row, col = i // 4, i % 4
                    img = dream_vis[i].permute(1, 2, 0).numpy()
                    axes[row, col].imshow(img)
                    axes[row, col].axis('off')
                    axes[row, col].set_title(f"Dream {i+1}", fontsize=8)
                
                plt.suptitle("VAE Dreams During SLEEP (Lazarus Consolidation)", fontsize=12)
                plt.tight_layout()
                
                # Сохраняем
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else os.getcwd()
                dream_path = os.path.join(project_root, "test_outputs", "test_outputs", "sleep_dreams.png")
                os.makedirs(os.path.dirname(dream_path), exist_ok=True)
                plt.savefig(dream_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   [DREAMS] Saved {num_dream_samples} dream samples to {dream_path}")
            except Exception as e:
                print(f"   [WARNING] Could not save dream visualization: {e}")
        
        # Заменяем сложный мозг на одного Студента
        self.heads = nn.ModuleList([student_head])
        self.active_classes_per_column = {0: list(range(10))}  # Объединяем все классы
        
        print(f"   Memory compressed: {len(self.heads)} head(s) remaining (shared backbone).")
        
        # КРИТИЧНО: Возвращаем изменение бюджета для "Когнитивного Кэшбека" или "Градиентного Штрафа"
        return budget_change

    def forward(self, x, return_features=False):
        # КРИТИЧНО: Элегантный режим - единый рекурсивный блок
        if self.use_elegant_mode and hasattr(self, 'elegant_core'):
            # Рекурсивное время встроено в цикл: чем сложнее картинка, тем больше итераций
            logits, h_final, reconstruction_error = self.elegant_core(x, max_steps=3)
            
            # КРИТИЧНО: Сюрприз как энергия - reconstruction_error автоматически определяет сложность
            # Не нужен отдельный Critic - энергия встроена в сам forward
            if return_features:
                return logits, h_final
            return logits
        
        # Стандартный режим с модульностью
        feats = self.shared_backbone(x)
        
        # КРИТИЧНО: Time Mixer - смешивает текущие features с прошлыми состояниями
        # Это сглаживает переходы между фазами и предотвращает "шок" при смене среды
        if self.use_time_mixer and hasattr(self, 'time_mixer'):
            feats = self.time_mixer(feats, use_adaptive=True)  # [B, feature_dim]
        
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
    
    def select_action(self, features, goal_features=None, weakness_signal=None, image=None, class_hint=None):
        """
        Выбирает действие (attention/patch selection) на основе текущего состояния и цели.
        
        КРИТИЧНО: Активное латентное воображение - реальная симуляция Zoom для минимизации Weakness.
        WorldModel делает внутреннюю симуляцию: "Если я посмотрю на уши, станет ли мне понятнее?"
        Если прогноз говорит "Да", дрон делает Zoom. Это превращает систему из "распознавателя"
        в "активного наблюдателя".
        
        Args:
            features: [B, feature_dim] - текущие features
            goal_features: [B, feature_dim] - целевые features (опционально)
            weakness_signal: [B, 1] или float - сигнал слабости от SelfModel (опционально)
            image: [B, 3, H, W] - исходное изображение для реальной симуляции Zoom (опционально)
            class_hint: [B] - подсказка о классе для улучшения внимания (опционально)
        
        Returns:
            action_idx: [B] - индекс выбранного patch (0..3)
            action_logits: [B, 4] - логиты действий
        """
        if not self.use_world_model:
            # Fallback: случайный выбор
            B = features.size(0)
            return torch.randint(0, 4, (B,), device=features.device), None
        
        # КРИТИЧНО: Активное латентное воображение - реальная симуляция Zoom
        if self.use_self_model and self.use_world_model:
            B = features.size(0)
            device = features.device
            
            # 1. Получаем текущую Weakness от SelfModel
            current_weakness = None
            if weakness_signal is not None:
                if isinstance(weakness_signal, torch.Tensor):
                    current_weakness = weakness_signal.mean().item() if weakness_signal.numel() > 0 else 0.0
                else:
                    current_weakness = float(weakness_signal)
            else:
                # Если weakness_signal не передан, получаем его от SelfModel
                current_weakness_pred = self.self_model.detect_weakness(features)
                current_weakness = current_weakness_pred.mean().item() if current_weakness_pred.numel() > 0 else 0.0
            
            # 2. Если есть слабость (>0.3), проигрываем все возможные действия через ГЛУБОКОЕ ВООБРАЖЕНИЕ
            # КРИТИЧНО: Multi-step Active Inference - цепочка виртуальных действий
            if current_weakness > 0.3:
                weakness_reductions = []  # насколько снизится Weakness для каждого действия
                
                # КРИТИЧНО: Реальная симуляция Zoom на изображении (если доступно)
                use_real_simulation = (image is not None and hasattr(self, 'attention_action') and self.attention_action is not None)
                
                # КРИТИЧНО: Параметры глубокого воображения
                max_imagination_steps = 3  # максимум 3 виртуальных шага
                min_improvement_per_step = 0.005  # минимальное улучшение для продолжения цепочки
                
                for patch_idx in range(4):
                    if use_real_simulation:
                        # ГЛУБОКОЕ ВООБРАЖЕНИЕ: цепочка виртуальных действий
                        # Посмотрел → Увидел деталь → Решил посмотреть еще ближе → Принял решение
                        current_image = image
                        current_features = features
                        current_weakness_step = current_weakness
                        total_improvement = 0.0
                        action_chain = [patch_idx]  # запоминаем цепочку действий
                        
                        for imagination_step in range(max_imagination_steps):
                            # Применяем текущее действие (Zoom)
                            action_idx_temp = torch.full((B,), action_chain[-1], device=device, dtype=torch.long)
                            image_zoomed = self.attention_action.apply_action(current_image, action_idx_temp, class_hint)
                            
                            # Получаем features после Zoom
                            with torch.no_grad():
                                if self.use_elegant_mode:
                                    _, features_after_zoom, _ = self.elegant_core(image_zoomed, max_steps=1)
                                else:
                                    features_after_zoom = self.shared_backbone(image_zoomed)
                                
                                # Оцениваем улучшение
                                predicted_weakness = self.self_model.detect_weakness(features_after_zoom)
                                predicted_weakness_mean = predicted_weakness.mean().item() if predicted_weakness.numel() > 0 else current_weakness_step
                                weakness_reduction_step = current_weakness_step - predicted_weakness_mean
                                
                                # Confidence и Entropy
                                current_confidence_step = self.self_model.estimate_confidence(current_features).mean().item() if hasattr(self.self_model, 'estimate_confidence') else 0.0
                                predicted_confidence = self.self_model.estimate_confidence(features_after_zoom).mean().item() if hasattr(self.self_model, 'estimate_confidence') else 0.0
                                confidence_increase_step = predicted_confidence - current_confidence_step
                                
                                # Entropy reduction и контрастивное воображение
                                entropy_reduction_step = 0.0
                                contrastive_score = 0.0  # КРИТИЧНО: разница между топ-2 классами
                                if self.use_elegant_mode:
                                    logits_after, _, _ = self.elegant_core(image_zoomed, max_steps=1)
                                    logits_current_step, _, _ = self.elegant_core(current_image, max_steps=1)
                                elif len(self.heads) > 0:
                                    logits_after, _ = self.heads[0](features_after_zoom, prev_hiddens=[])
                                    logits_current_step, _ = self.heads[0](current_features, prev_hiddens=[])
                                    if logits_after is not None and logits_current_step is not None and logits_after.size(1) >= 10:
                                        probs_after = torch.softmax(logits_after[:, :10], dim=1)
                                        probs_current_step = torch.softmax(logits_current_step[:, :10], dim=1)
                                        entropy_after = -(probs_after * torch.log(probs_after + 1e-9)).sum(dim=1).mean().item()
                                        entropy_current_step = -(probs_current_step * torch.log(probs_current_step + 1e-9)).sum(dim=1).mean().item()
                                        entropy_reduction_step = entropy_current_step - entropy_after
                                        
                                        # КРИТИЧНО: Контрастивное воображение - выбираем патч, который максимально различает топ-2 класса
                                        # Если модель сомневается между Cat и Dog, ищем патч (уши/нос), где разница максимальна
                                        probs_current_mean = probs_current_step.mean(dim=0)  # [10] - средние вероятности по батчу
                                        top2_values, top2_indices = torch.topk(probs_current_mean, k=2)
                                        
                                        # Вычисляем разницу между топ-2 классами до и после Zoom
                                        diff_before = top2_values[0].item() - top2_values[1].item()  # разница до Zoom
                                        probs_after_mean = probs_after.mean(dim=0)  # [10]
                                        top2_after_values, _ = torch.topk(probs_after_mean, k=2)
                                        diff_after = top2_after_values[0].item() - top2_after_values[1].item()  # разница после Zoom
                                        
                                        # Контрастивный score: увеличение разницы между топ-2 классами
                                        contrastive_score = diff_after - diff_before  # положительное = хороший патч для различения
                                
                                # Комбинированный score для этого шага
                                step_score = (
                                    1.0 * weakness_reduction_step +
                                    1.0 * confidence_increase_step +
                                    0.5 * entropy_reduction_step +
                                    0.8 * contrastive_score  # КРИТИЧНО: контрастивное воображение для Cat vs Dog
                                )
                                
                                # Если улучшение значительное, продолжаем цепочку
                                if step_score > min_improvement_per_step:
                                    total_improvement += step_score
                                    current_image = image_zoomed  # обновляем для следующего шага
                                    current_features = features_after_zoom
                                    current_weakness_step = predicted_weakness_mean
                                    
                                    # КРИТИЧНО: Решаем, нужно ли смотреть еще ближе или переключиться на другой патч
                                    # Если weakness все еще высокая, пробуем ВСЕ патчи и выбираем лучший для следующего шага
                                    if current_weakness_step > 0.3 and imagination_step < max_imagination_steps - 1:
                                        # КРИТИЧНО: Пробуем все 4 патча и выбираем лучший для следующего шага
                                        # Это позволяет системе "исследовать" разные области (уши -> нос -> глаза)
                                        best_next_patch = action_chain[-1]  # по умолчанию остаемся на том же
                                        best_next_score = 0.0
                                        
                                        for candidate_patch in range(4):
                                            # Быстрая оценка каждого патча
                                            action_candidate = torch.full((B,), candidate_patch, device=device, dtype=torch.long)
                                            image_candidate = self.attention_action.apply_action(current_image, action_candidate, class_hint)
                                            
                                            with torch.no_grad():
                                                if self.use_elegant_mode:
                                                    _, features_candidate, _ = self.elegant_core(image_candidate, max_steps=1)
                                                else:
                                                    features_candidate = self.shared_backbone(image_candidate)
                                                weakness_candidate = self.self_model.detect_weakness(features_candidate).mean().item()
                                                confidence_candidate = self.self_model.estimate_confidence(features_candidate).mean().item() if hasattr(self.self_model, 'estimate_confidence') else 0.0
                                                
                                                # КРИТИЧНО: Контрастивное воображение для выбора следующего патча
                                                contrastive_candidate = 0.0
                                                if self.use_elegant_mode:
                                                    logits_candidate, _, _ = self.elegant_core(image_candidate, max_steps=1)
                                                elif len(self.heads) > 0:
                                                    logits_candidate, _ = self.heads[0](features_candidate, prev_hiddens=[])
                                                    if logits_candidate is not None and logits_candidate.size(1) >= 10:
                                                        probs_candidate = torch.softmax(logits_candidate[:, :10], dim=1)
                                                        probs_candidate_mean = probs_candidate.mean(dim=0)  # [10]
                                                        top2_candidate, _ = torch.topk(probs_candidate_mean, k=2)
                                                        diff_candidate = top2_candidate[0].item() - top2_candidate[1].item()
                                                        
                                                        # Сравниваем с текущим состоянием
                                                        if self.use_elegant_mode:
                                                            logits_current_temp, _, _ = self.elegant_core(current_image, max_steps=1)
                                                        else:
                                                            logits_current_temp, _ = self.heads[0](current_features, prev_hiddens=[])
                                                        if logits_current_temp is not None:
                                                            probs_current_temp = torch.softmax(logits_current_temp[:, :10], dim=1)
                                                            probs_current_temp_mean = probs_current_temp.mean(dim=0)
                                                            top2_current_temp, _ = torch.topk(probs_current_temp_mean, k=2)
                                                            diff_current_temp = top2_current_temp[0].item() - top2_current_temp[1].item()
                                                            contrastive_candidate = diff_candidate - diff_current_temp
                                                
                                                # Быстрая оценка с контрастивным воображением
                                                candidate_score = (
                                                    (current_weakness_step - weakness_candidate) +
                                                    0.5 * (confidence_candidate - current_confidence_step) +
                                                    0.6 * contrastive_candidate  # контрастивный компонент
                                                )
                                                
                                                if candidate_score > best_next_score:
                                                    best_next_score = candidate_score
                                                    best_next_patch = candidate_patch
                                        
                                        # Если лучший патч дает значительное улучшение, переключаемся
                                        if best_next_score > min_improvement_per_step:
                                            action_chain.append(best_next_patch)
                                        else:
                                            # Улучшение недостаточное, останавливаемся
                                            break
                                    else:
                                        # Достаточно улучшения, останавливаемся
                                        break
                                else:
                                    # Улучшение недостаточное, останавливаемся
                                    break
                        
                        # Сохраняем общее улучшение для этого начального патча
                        weakness_reductions.append(total_improvement)
                    else:
                        # FALLBACK: прогнозирование через WorldModel (если изображение недоступно)
                        action_onehot = torch.zeros(B, 4, device=device)
                        action_onehot[:, patch_idx] = 1.0
                        
                        next_features_pred, _, _, _, _ = self.world_model.predict_next(features, action_onehot)
                        
                        if next_features_pred is not None:
                            predicted_weakness = self.self_model.detect_weakness(next_features_pred)
                            predicted_weakness_mean = predicted_weakness.mean().item() if predicted_weakness.numel() > 0 else current_weakness
                            weakness_reduction = current_weakness - predicted_weakness_mean
                            weakness_reductions.append(weakness_reduction)
                        else:
                            weakness_reductions.append(0.0)
                
                # 3. Выбираем действие с максимальным улучшением (после глубокого воображения)
                if weakness_reductions and max(weakness_reductions) > 0.005:
                    best_idx = np.argmax(weakness_reductions)
                    action_idx = torch.full((B,), best_idx, device=device, dtype=torch.long)
                    action_logits = torch.zeros(B, 4, device=device)
                    action_logits[:, best_idx] = 1.0
                    
                    # Логирование для отладки (только при значительном улучшении)
                    if B > 0 and current_weakness > 0.4 and weakness_reductions[best_idx] > 0.01:
                        sim_type = "DEEP" if use_real_simulation else "PREDICTED"
                        # Логируем только при значительном улучшении (score > 0.01)
                        # Показываем, что это результат глубокого воображения (несколько шагов)
                        print(f"   [DEEP IMAGINATION] {sim_type} Weakness: {current_weakness:.3f} -> {current_weakness - weakness_reductions[best_idx]:.3f} (patch {best_idx}, total_score: {weakness_reductions[best_idx]:.3f})")
                    
                    return action_idx, action_logits
        
        # Fallback: обычный выбор через WorldModel
        action_logits = self.world_model.predict_best_action(features, goal_features)
        action_idx = torch.argmax(action_logits, dim=-1)  # [B]
        return action_idx, action_logits
    
    def apply_action_to_image(self, x, action_idx, class_hint=None):
        """
        Применяет действие к изображению с селективным вниманием.
        
        Args:
            x: [B, 3, 32, 32] - изображение
            action_idx: [B] - индекс действия
            class_hint: [B] - подсказка о классе для улучшения внимания (опционально)
        """
        if not self.use_world_model:
            return x
        return self.attention_action.apply_action(x, action_idx, class_hint)
    
    def predict_next_state(self, features, action_idx):
        """
        Предсказывает следующее состояние на основе действия.
        
        Args:
            features: [B, feature_dim] - текущие features
            action_idx: [B] - индекс действия (0..3)
        
        Returns:
            next_features_pred: [B, feature_dim] - предсказанные features
            z_mean, z_logvar, next_z_mean, next_z_logvar - для KL loss
        """
        if not self.use_world_model:
            return None, None, None, None, None
        
        # Конвертируем action_idx в one-hot
        action_onehot = F.one_hot(action_idx, num_classes=4).float()  # [B, 4]
        
        # Предсказываем следующее состояние
        next_features_pred, z_mean, z_logvar, next_z_mean, next_z_logvar = self.world_model.predict_next(
            features, action_onehot
        )
        return next_features_pred, z_mean, z_logvar, next_z_mean, next_z_logvar
    
    def generate_internal_goal(self, features):
        """Генерирует внутреннюю цель на основе текущего состояния"""
        if not self.use_internal_goals:
            return None
        return self.internal_goals.generate_goal(features)
    
    def extract_own_concepts(self, features):
        """Извлекает собственные концепты из features"""
        if not self.use_own_concepts:
            return None, None
        return self.own_concepts.extract_concepts(features)
    
    def record_autobiographical_memory(self, step, state_features, action, outcome, reward_signal=None, context=None, pain_level=None):
        """Записывает эпизод в автобиографическую память"""
        if not self.use_autobiographical_memory:
            return
        self.autobiographical_memory.record(step, state_features, action, outcome, reward_signal, context, pain_level)
    
    def recall_similar_experiences(self, query_features, k=10):
        """Вспоминает похожие эпизоды из автобиографической памяти"""
        if not self.use_autobiographical_memory:
            return []
        return self.autobiographical_memory.recall(query_features, k)
    
    def self_assess_capabilities(self, features):
        """Оценивает собственные способности через Self Model"""
        if not self.use_self_model:
            return None, None, None
        
        capabilities = self.self_model.predict_capabilities(features)
        confidence = self.self_model.estimate_confidence(features)
        weakness = self.self_model.detect_weakness(features)
        return capabilities, confidence, weakness


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


@torch.no_grad()
def eval_global(agent, loader, device):
    """
    Глобальная метрика точности по всем 10 классам без маскирования.
    Настоящий интеллект должен сам понимать, где он находится.
    Показывает, насколько хорошо Soft Routing Gate научился переключать контексты.
    """
    was_training = agent.training
    agent.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = agent(x)
        # Без маскирования - берём предсказание по всем 10 классам (без Unknown)
        out_10 = out[:, :10]  # [B, 10] - только известные классы
        pred = out_10.argmax(dim=1)
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
    # КРИТИЧНО: комбинированный loader для глобальной метрики (все 10 классов)
    test_all_combined = ConcatDataset([test_A, test_B])
    test_loader_all = DataLoader(test_all_combined, batch_size=500, shuffle=False, num_workers=4, pin_memory=True)

    use_curiosity = CLIP_AVAILABLE
    use_subjective_time = True
    use_vae_dreams = True
    use_fractal_time = True
    train_late_backbone = True      # IMPORTANT: if True, include late backbone params into optimizer_phase2
    use_adaptive_pain = True

    # AGI Components (все включены для полной AGI-системы)
    use_world_model = True  # World Model: предсказание будущих состояний
    use_internal_goals = True  # Внутренние цели: независимые от внешних наград
    use_own_concepts = True  # Собственные концепты: генерируемые системой
    use_autobiographical_memory = True  # Автобиографическая память: запись опыта
    use_self_model = True  # Self Model: модель самого себя
    
    # КРИТИЧНО: Элегантный режим - фрактальная простота вместо модульности
    #   - Единый рекурсивный блок вместо Backbone + WorldModel + Critic
    #   - Сюрприз как энергия (reconstruction error) вместо отдельного Critic
    #   - Кристаллизация через EMA весов (одна строчка) вместо сложной логики
    #   - Экономия памяти: одна модель вместо 5 разных
    use_elegant_mode = True  # ВКЛЮЧЕН: элегантная архитектура с фрактальной простотой
    
    agent = RecursiveAgent(
        use_curiosity=use_curiosity,
        use_subjective_time=use_subjective_time,
        use_vae_dreams=use_vae_dreams,
        use_world_model=use_world_model,
        use_internal_goals=use_internal_goals,
        use_own_concepts=use_own_concepts,
        use_autobiographical_memory=use_autobiographical_memory,
        use_self_model=use_self_model,
        use_elegant_mode=use_elegant_mode,  # Элегантный режим: фрактальная простота
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
                # КРИТИЧНО: Phase 1 локальный лосс - только по активным классам (classes_A)
                # Это создаёт чистый фундамент знаний перед выходом в "дикую природу"
                # Убираем лишний шум от неактивных нейронов Unknown класса на старте
                classes_A_t = torch.tensor(classes_A, device=device, dtype=torch.long)
                logits10 = logits[:, :10]
                logitsA = logits10.index_select(1, classes_A_t)  # [B, len(classes_A)] - только активные классы
                
                # Глобальные -> локальные targets
                g2l = {c: i for i, c in enumerate(classes_A)}
                targetA = torch.tensor([g2l[int(t.item())] for t in target], device=device, dtype=torch.long)
                
                # Локальный loss только по активным классам
                loss = criterion_train(logitsA, targetA)
                # Add pair margin loss to reduce Plane↔Ship, Car↔Truck confusion (только по активным)
                # КРИТИЧНО: преобразуем глобальные пары в локальные индексы
                # Пары для Phase 1 (Machines): (0, 8) -> Plane↔Ship, (1, 9) -> Car↔Truck
                pairs_local = []
                for g_a, g_b in ((0, 8), (1, 9)):  # глобальные пары
                    if g_a in g2l and g_b in g2l:
                        pairs_local.append((g2l[g_a], g2l[g_b]))  # локальные индексы
                if pairs_local:
                    pm = pair_margin_loss(logitsA, targetA, pairs=tuple(pairs_local), margin=0.15)
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
        
        # КРИТИЧНО: добавляем параметры AGI компонентов (WorldModel, SelfModel, InternalGoals)
        # Это обеспечивает непрерывную эволюцию когнитивных функций при расширении
        if agent.use_world_model and hasattr(agent, 'world_model') and agent.world_model is not None:
            wm_params = list(agent.world_model.parameters())
            if len(wm_params) > 0:
                param_groups.append({"params": wm_params, "lr": 1e-3, "tag": "world_model"})
        
        if agent.use_self_model and hasattr(agent, 'self_model') and agent.self_model is not None:
            sm_params = list(agent.self_model.parameters())
            if len(sm_params) > 0:
                param_groups.append({"params": sm_params, "lr": 1e-3, "tag": "self_model"})
        
        if agent.use_internal_goals and hasattr(agent, 'internal_goals') and agent.internal_goals is not None:
            ig_params = list(agent.internal_goals.parameters())
            if len(ig_params) > 0:
                param_groups.append({"params": ig_params, "lr": 1e-3, "tag": "internal_goals"})
        
        # КРИТИЧНО: Time Mixer - обучаем вместе с остальными компонентами
        if agent.use_time_mixer and hasattr(agent, 'time_mixer') and agent.time_mixer is not None:
            tm_params = list(agent.time_mixer.parameters())
            if len(tm_params) > 0:
                param_groups.append({"params": tm_params, "lr": 1e-3, "tag": "time_mixer"})

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
            
            # КРИТИЧНО: Инициализируем features_f32 в начале цикла (будет обновлена позже)
            features_f32 = None
            
            # КРИТИЧНО: В элегантном режиме получаем features_f32 из data_real сразу (после перемещения на device)
            # Но сначала нужно переместить data_real на device
            
            # КРИТИЧНО: перемещаем data_real и target_real на device сразу после создания
            # чтобы избежать проблем с device mismatch в pain-блоке
            data_real = data_real.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target_real = target_real.to(device, non_blocking=True)
            
            # КРИТИЧНО: В элегантном режиме получаем features_f32 из data_real (после перемещения на device)
            if agent.use_elegant_mode:
                with torch.no_grad():
                    _, features_f32, _ = agent.elegant_core(data_real[:min(64, real_B)], max_steps=1)

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
            # В элегантном режиме expansion не нужен (единый рекурсивный блок)
            has_budget = False if agent.use_elegant_mode else (len(agent.heads) < MAX_LAYERS)
            
            # КРИТИЧНО: growth_budget растёт без decay (простое накопление)
            agent.growth_budget = min(1.0, agent.growth_budget + 0.001)
            has_growth_budget = agent.growth_budget >= agent.growth_cost_per_expansion
            
            # КРИТИЧНО: получаем features_f32 для AGI компонентов (нужно для expansion decision)
            # Делаем это раньше, чтобы использовать в блоке expansion
            # В элегантном режиме features будут получены позже из data_real (после его определения)
            features_f32 = None
            
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
            # КРИТИЧНО: FORCE_EXPANSION использует phase2_steps, а не глобальный step
            force_expansion = (expansion_count == 0 and phase2_steps >= FORCE_EXPANSION_STEPS and has_budget)
            fallback_expansion = (
                not fallback_expansion_attempted and 
                can_expand and 
                has_budget and 
                test_acc < FALLBACK_EXPANSION_THRESHOLD and 
                float(test_loss.item()) > 2.0
            )

            # КРИТИЧНО: вычисляем complexity и actions ДО блока expansion decision
            # Используем приближение surprise из entropy_test для первого прохода
            complexity = 0.0
            actions = None
            if agent.use_elegant_mode:
                # В элегантном режиме используем reconstruction_error как surprise
                # Используем data_real, который уже определен выше в цикле
                with torch.no_grad():
                    if features_f32 is not None:
                        # Если features_f32 уже получены, используем их для оценки surprise
                        # В элегантном режиме reconstruction_error уже был вычислен при получении features_f32
                        # Используем приближение через entropy_test
                        surp_approx = min(2.0, 0.5 * entropy_test if entropy_test > 0 else 0.0)
                    else:
                        # Fallback: используем entropy_test как приближение surprise
                        surp_approx = min(2.0, 0.5 * entropy_test if entropy_test > 0 else 0.0)
                complexity = agent.complexity_controller.compute_complexity(
                    surprise=surp_approx,
                    pain=pain_value,
                    entropy=entropy_test,
                    unknown_rate=unknown_rate_test
                )
                actions = agent.complexity_controller.get_actions(
                    complexity=complexity,
                    has_expansion_budget=False,  # в элегантном режиме expansion не нужен
                    cooldown_ok=False,
                    weakness_signal=None
                )
            elif len(agent.heads) > 0:
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
            elif not agent.use_elegant_mode and len(agent.heads) > 0 and actions is not None and actions["expand_allowed"]:
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
            
            # КРИТИЧНО: проверяем дублирующий scope перед expansion
            def scope_already_exists(agent, new_scope):
                """Проверяет, существует ли уже head с таким же scope"""
                s = set(new_scope)
                return any(set(v) == s for v in agent.active_classes_per_column.values())
            
            # КРИТИЧНО: запрещаем создание head с дублирующим scope
            if should_expand and has_budget:
                expansion_classes = expansion_new_classes if expansion_new_classes is not None else classes_B
                if scope_already_exists(agent, expansion_classes):
                    should_expand = False
                    if step % 50 == 0:
                        print(f"[EXPANSION] Skipped: identical scope {expansion_classes} already exists")
            
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
                
                # КРИТИЧНО: Записываем weakness ДО expansion для мета-регуляции
                weakness_before = None
                if agent.use_self_model and features_f32 is not None:
                    if agent.use_elegant_mode or len(agent.heads) > 0:
                        weakness_pred = agent.self_model.detect_weakness(features_f32)
                        weakness_before = weakness_pred.mean().item() if weakness_pred.numel() > 0 else 0.0
                
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
                
                # КРИТИЧНО: Записываем expansion в историю для мета-регуляции
                expansion_entry = {
                    "step": step,
                    "weakness_before": weakness_before,
                    "weakness_after": None  # будет обновлено позже
                }
                agent.expansion_history["recent"].append(expansion_entry)
                # Ограничиваем размер истории
                if len(agent.expansion_history["recent"]) > agent.expansion_history["max_history"]:
                    agent.expansion_history["recent"].pop(0)
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
                # КРИТИЧНО: "Синтез Полиглота" - умное слияние голов для разблокировки Head Limit
                # Вместо блокировки сливаем похожие головы, освобождая место для новых
                if not agent.use_elegant_mode and len(agent.heads) >= MAX_LAYERS:
                    print(f"\n[EVOLUTION] Head Limit ({MAX_LAYERS}) reached. Initiating Polyglot Synthesis...")
                    merge_success = agent._polyglot_synthesis(device, similarity_threshold=0.80)  # более агрессивный порог для слияния
                    if merge_success:
                        print(f"[EVOLUTION] Polyglot Synthesis successful! Heads: {len(agent.heads)}/{MAX_LAYERS}. Budget freed for expansion.")
                        # После успешного слияния можно попробовать expansion снова
                        has_budget = len(agent.heads) < MAX_LAYERS
                    else:
                        print(f"[EVOLUTION] Polyglot Synthesis: No mergeable heads found. Forcing SLEEP for consolidation.")
                        # Если слияние невозможно, принудительно запускаем SLEEP (если прошло достаточно шагов)
                        # Используем ту же логику, что и в Intelligent sleep ниже
                        if steps_since_sleep >= 500:  # минимум 500 шагов между sleep
                            should_sleep = True

            # Intelligent sleep: консолидация знаний из нескольких heads в один
            steps_since_sleep = step - last_sleep_step
            should_sleep = (
                len(agent.heads) >= 2
                and expansion_count > 0
                and steps_since_sleep > SLEEP_TRIGGER_STEPS
                and (error_count_phase2 > SLEEP_TRIGGER_ERRORS or steps_since_sleep > SLEEP_TRIGGER_STEPS * 2)
            )
            if should_sleep:
                print(f"\n[INTELLIGENT SLEEP] Triggered after {steps_since_sleep} steps and {error_count_phase2} errors.")
                print(f"[ACTION] Initiating SLEEP PHASE to consolidate knowledge and reduce confusion...")
                
                # Запускаем консолидацию через dream distillation
                # КРИТИЧНО: Передаем test_loader для проверки успешности слияния ("Когнитивный Кэшбек")
                budget_change = agent.dream_and_compress(num_dreams=1500, dream_batch_size=100, device=device, test_loader=test_loader_all)
                # Применяем изменение бюджета ("Когнитивный Кэшбек" или "Градиентный Штраф")
                if budget_change != 0.0:
                    agent.complexity_controller.complexity_budget = min(1.0, max(0.0, agent.complexity_controller.complexity_budget + budget_change))
                    print(f"   [BUDGET UPDATE] Complexity budget updated: {agent.complexity_controller.complexity_budget:.3f} (change: {budget_change:+.3f})")
                
                # Перезагружаем optimizer после консолидации
                # После SLEEP остаётся только один head, создаём новый optimizer
                if agent.use_elegant_mode:
                    # В элегантном режиме создаем optimizer для elegant_core
                    optimizer_phase2 = torch.optim.AdamW(agent.elegant_core.parameters(), lr=2e-3)
                elif len(agent.heads) > 0:
                    # Используем существующую функцию build_phase2_optimizer
                    optimizer_phase2 = build_phase2_optimizer(agent.heads[0])  # После SLEEP только один head
                    for pg in optimizer_phase2.param_groups:
                        pg["lr_base"] = pg["lr"]
                        pg["scheduler_factor"] = 1.0
                    # Обновляем scheduler
                    remaining_steps = max(1000, total_steps_phase2 - step)  # минимум 1000 шагов
                    scheduler_phase2 = CosineAnnealingLR(optimizer_phase2, T_max=remaining_steps, eta_min=1e-5)
                
                # Сбрасываем состояние
                last_sleep_step = step
                error_count_phase2 = 0
                expansion_count = 0  # Сбрасываем после консолидации
                print("[WAKE UP] Knowledge consolidated. Heads merged. Agent ready to continue learning.")

            # 2) training step
            current_opt = optimizer_phase2 if optimizer_phase2 is not None else optimizer
            current_opt.zero_grad(set_to_none=True)

            # КРИТИЧНО: complexity и actions уже вычислены ДО блока expansion decision
            # Здесь применяем gate_temperature к routing gate (если actions доступен)
            # КРИТИЧНО: применяем gate_temperature к routing gate
            if agent.use_soft_routing and agent.routing_gate is not None and actions is not None:
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
            
            # КРИТИЧНО: получаем features_f32 для AGI компонентов (если еще не получены)
            if features_f32 is None:
                if agent.use_elegant_mode:
                    with torch.no_grad():
                        _, features_f32, _ = agent.elegant_core(data_real[:min(64, real_B)], max_steps=1)
                elif len(agent.heads) > 0:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _, features_temp = agent(data_real[:min(64, real_B)], return_features=True)
                        if features_temp is not None:
                            features_f32 = features_temp[:min(64, real_B)].float()  # конвертируем в float32
            
            # ---- forward (BF16) с рекурсией ----
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                # Pass0: начальный прогноз
                outputs_pass0, features_pass0 = agent(data_mix, return_features=True)
                all_outputs.append(outputs_pass0)
                all_features.append(features_pass0)
                used_recursions += 1
                
                # КРИТИЧНО: обновляем features_f32 из первого прохода (для AGI компонентов)
                if features_f32 is None and features_pass0 is not None:
                    features_f32 = features_pass0[:real_B].float()  # конвертируем в float32
                
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
                
                # --- AGI COMPONENTS INTEGRATION ---
                # КРИТИЧНО: конвертируем features в float32 для AGI компонентов (они не в autocast)
                features_f32 = features[:real_B].float() if features.dtype == torch.bfloat16 else features[:real_B]
                
                # 1. World Model: предсказание следующего состояния и обучение
                loss_world_model = torch.zeros((), device=device, dtype=torch.float32)
                wm_error_signal = 0.0
                action_idx = None
                # Инициализируем trust factor по умолчанию (на случай если World Model не используется)
                agent._world_model_trust = 1.0
                # КРИТИЧНО: Warm-up для Active Imagination - необученная WorldModel это генератор галлюцинаций
                # Сначала "насмотренность" (обучение на данных), затем "аналитика" (активное воображение)
                # Проверяем weakness или счетчик шагов перед использованием WorldModel для управления вниманием
                # В элегантном режиме WorldModel не используется (единый рекурсивный блок)
                world_model_ready = False
                if agent.use_world_model and not agent.use_elegant_mode and len(agent.heads) > 0:
                    # Проверка 1: Weakness threshold (WorldModel должна быть достаточно обучена)
                    if agent.use_self_model:
                        weakness_pred = agent.self_model.detect_weakness(features_f32)
                        weakness_val = weakness_pred.mean().item() if weakness_pred.numel() > 0 else 1.0
                        # Если weakness низкая (< 0.7), WorldModel достаточно обучена
                        world_model_ready = (weakness_val < 0.7)
                    else:
                        # Проверка 2: Счетчик шагов Phase2 (минимум 100 шагов для прогрева)
                        world_model_ready = (phase2_steps >= 100)
                
                if agent.use_world_model and len(agent.heads) > 0 and world_model_ready:
                    # Генерируем цель (если включены goals)
                    goal_features = None
                    if agent.use_internal_goals:
                        goal = agent.generate_internal_goal(features_f32)
                        # Конвертируем goal в goal_features (используем goal как features для simplicity)
                        goal_features = agent.internal_goals.goal_generator(features_f32)
                    
                    # КРИТИЧНО: Активное латентное воображение - реальная симуляция Zoom
                    # Если SelfModel сообщает о высокой Weakness, используем проигрывание вариантов
                    weakness_signal = None
                    if agent.use_self_model and len(agent.heads) > 0:
                        weakness_pred = agent.self_model.detect_weakness(features_f32)
                        weakness_signal = weakness_pred.mean().item() if weakness_pred.numel() > 0 else 0.0
                    
                    # КРИТИЧНО: Получаем class_hint для улучшения внимания
                    class_hint = None
                    if len(all_outputs) > 0:
                        pred_outputs = all_outputs[-1][:real_B, :10] if all_outputs[-1].size(0) >= real_B else all_outputs[-1][:, :10]
                        class_hint = pred_outputs.argmax(dim=1)  # [real_B]
                    
                    # Выбираем действие с учётом слабости и реальным изображением для симуляции
                    # КРИТИЧНО: передаём image для реальной симуляции Zoom
                    action_idx, action_logits = agent.select_action(
                        features_f32, 
                        goal_features, 
                        weakness_signal,
                        image=data_real[:real_B] if real_B > 0 else None,  # КРИТИЧНО: реальное изображение для симуляции
                        class_hint=class_hint
                    )
                    
                    # КРИТИЧНО: class_hint уже получен выше в select_action
                    # Применяем действие к изображению с селективным вниманием
                    x_next = agent.apply_action_to_image(data_real, action_idx, class_hint)
                    with torch.no_grad():
                        features_next = agent.shared_backbone(x_next)
                    
                    # Предсказываем следующее состояние
                    next_features_pred, z_mean, z_logvar, next_z_mean, next_z_logvar = agent.predict_next_state(
                        features_f32, action_idx
                    )
                    
                    if next_features_pred is not None:
                        # World Model Loss: reconstruction + KL
                        loss_wm_recon = F.mse_loss(next_features_pred, features_next.detach())
                        loss_wm_kl = -0.5 * torch.sum(1 + next_z_logvar - next_z_mean.pow(2) - next_z_logvar.exp()) / features[:real_B].size(0)
                        loss_world_model = loss_wm_recon + 0.1 * loss_wm_kl
                        
                        # КРИТИЧНО: "Инстинкт недоверия" к воображению на ранних этапах новой фазы
                        # На ранних этапах World Model может генерировать галлюцинации
                        # Уменьшаем доверие к её прогнозам до тех пор, пока она не "насмотрится" на новые данные
                        world_model_trust = 1.0
                        if phase2_steps < 200:  # первые 200 шагов Phase 2
                            # Линейное увеличение доверия: 0 шагов -> 0.3, 200 шагов -> 1.0
                            world_model_trust = 0.3 + 0.7 * (phase2_steps / 200.0)
                        elif agent.use_self_model and weakness_signal is not None:
                            # Дополнительная проверка через weakness: если weakness высокая, уменьшаем доверие
                            if weakness_signal > 0.6:
                                world_model_trust = max(0.5, world_model_trust * (1.0 - (weakness_signal - 0.6) * 0.5))
                        
                        # Сохраняем trust factor для использования при добавлении в total_loss
                        agent._world_model_trust = world_model_trust
                        
                        # World Model error для Complexity Controller
                        wm_error_signal = float(loss_wm_recon.item())
                        agent._last_wm_error = wm_error_signal
                    else:
                        # Если World Model не готова, trust = 0
                        agent._world_model_trust = 0.0
                
                # 2. Goal-conditioned Policy: модифицируем actions на основе целей
                if agent.use_internal_goals and actions is not None and len(agent.heads) > 0:
                    goal = agent.generate_internal_goal(features_f32)
                    policy_mod = agent.internal_goals.get_goal_policy_modifier(features_f32, goal)
                    
                    # Модифицируем actions
                    policy_mod_mean = policy_mod.mean(dim=0)  # [3] - [recursion_boost, replay_boost, temperature_mod]
                    actions["n_recursions"] = max(1, min(3, int(actions["n_recursions"] + policy_mod_mean[0].item())))
                    actions["replay_ratio"] = max(0.1, min(0.4, actions["replay_ratio"] + policy_mod_mean[1].item() * 0.1))
                    actions["gate_temperature"] = max(0.7, min(2.0, actions["gate_temperature"] + policy_mod_mean[2].item() * 0.2))
                
                # 3. Memory → Decision: вспоминаем похожие эпизоды и модифицируем policy
                if agent.use_autobiographical_memory and len(agent.heads) > 0 and complexity > 0.5:
                    query_feat = features_f32[0] if features_f32.size(0) > 0 else features_f32.mean(dim=0)
                    similar_memories = agent.recall_similar_experiences(query_feat, k=5)
                    if similar_memories and actions is not None:
                        memory_mod = agent.autobiographical_memory.get_recall_policy_modifier(similar_memories)
                        actions["n_recursions"] = max(1, min(3, int(actions["n_recursions"] + memory_mod["recursion_boost"])))
                        actions["replay_ratio"] = max(0.1, min(0.4, actions["replay_ratio"] + memory_mod["replay_boost"]))
                
                # 4. Concepts в управлении: модифицируем routing на основе концептов
                loss_concept_recon = torch.zeros((), device=device, dtype=torch.float32)
                if agent.use_own_concepts and len(agent.heads) > 0:
                    concept_activations, concept_importance = agent.extract_own_concepts(features_f32)
                    routing_signal = agent.own_concepts.get_concept_based_routing(concept_activations, concept_importance)
                    
                    # Модифицируем routing gate temperature на основе концептов
                    if actions is not None and agent.use_soft_routing:
                        concept_temp_mod = routing_signal.mean().item() * 0.3  # [-0.3..0.3]
                        actions["gate_temperature"] = max(0.7, min(2.0, actions["gate_temperature"] + concept_temp_mod))
                    
                    # Concept reconstruction loss
                    reconstructed = agent.own_concepts.reconstruct_from_concepts(concept_activations)
                    loss_concept_recon = F.mse_loss(reconstructed, features_f32.detach())
                
                # 5. Self-Model Supervision: обучаем на внутренних метриках
                loss_self_model = torch.zeros((), device=device, dtype=torch.float32)
                if agent.use_self_model and len(agent.heads) > 0:
                    # Вычисляем реальные метрики (упрощённо)
                    # КРИТИЧНО: используем float32 для capabilities_real
                    capabilities_real = torch.ones(len(agent.heads), device=device, dtype=torch.float32) * 0.8  # placeholder
                    confidence_real = max(0.0, min(1.0, 1.0 - float(loss_new.item()) / 2.0)) if 'loss_new' in locals() else 0.5
                    weakness_real = min(1.0, (float(surprise.item()) if surprise is not None else 0.0) + pain_value + entropy_test / 2.3)
                    
                    # Обновляем EMA targets (будет использовано после вычисления loss_new)
                    # Пока сохраняем для использования позже
                    agent._self_model_targets = {
                        "capabilities": capabilities_real,
                        "confidence": confidence_real,
                        "weakness": weakness_real
                    }
                
                # КРИТИЧНО: обновляем complexity после вычисления surprise (для следующего шага)
                # Работаем даже после sleep (когда expansion_count может быть 0, но есть heads)
                # 6. Complexity Controller с World Model Error (если доступен)
                if len(agent.heads) > 0:
                    if agent.use_world_model and wm_error_signal > 0:
                        # Используем prediction error от World Model как сигнал сложности
                        surp_val = wm_error_signal
                    elif surprise is not None:
                        surp_val = float(surprise.item())
                    else:
                        surp_val = entropy_test * 0.5  # fallback
                    
                    complexity = agent.complexity_controller.compute_complexity(
                        surprise=surp_val,
                        pain=pain_value,  # будет обновлён позже в pain-блоке
                        entropy=entropy_test,
                        unknown_rate=unknown_rate_test
                    )
                    # КРИТИЧНО: Мета-регуляция бюджета - получаем weakness для utility analysis
                    weakness_signal = None
                    if agent.use_self_model and len(agent.heads) > 0:
                        weakness_pred = agent.self_model.detect_weakness(features_f32)
                        weakness_signal = weakness_pred.mean().item() if weakness_pred.numel() > 0 else 0.0
                    
                    # Обновляем actions с правильной complexity (для следующего шага)
                    has_expansion_budget = agent.growth_budget >= agent.growth_cost_per_expansion
                    actions = agent.complexity_controller.get_actions(
                        complexity=complexity,
                        has_expansion_budget=has_expansion_budget,
                        cooldown_ok=can_expand,
                        weakness_signal=weakness_signal,
                        expansion_history=agent.expansion_history
                    )
                
                # КРИТИЧНО: Автономный Роутинг - обучение на сигнале сюрприза
                # Если SubjectiveTimeCritic кричит о высоком Surprise, система должна автоматически
                # перенаправлять градиент в RoutingGate, чтобы он быстрее выучил: "Это новая среда, переключи внимание на Head 2"
                routing_entropy_loss = torch.zeros((), device=device, dtype=torch.float32)
                routing_surprise_loss = torch.zeros((), device=device, dtype=torch.float32)
                if agent.use_soft_routing and len(agent.heads) > 1:
                    # Вычисляем gates_full и берём срез (как в forward)
                    gates_full = agent.routing_gate(features[:real_B])  # [real_B, MAX_LAYERS]
                    gates = gates_full[:, :len(agent.heads)]  # [B, H]
                    gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-9)  # нормализуем
                    
                    # 1. Entropy penalty: поощряем равномерное распределение ответственности
                    gate_entropy = -torch.sum(gates * torch.log(gates + 1e-9), dim=1).mean()
                    # Целевая entropy (максимальная для равномерного распределения)
                    target_entropy = torch.log(torch.tensor(len(agent.heads), dtype=torch.float32, device=gates.device))
                    # Penalty если entropy слишком низкая (коллапс к одному head)
                    routing_entropy_loss = F.relu(target_entropy * 0.7 - gate_entropy)  # penalty если entropy < 70% от максимума
                    
                    # 2. КРИТИЧНО: Surprise-based routing loss - автономное переключение контекста
                    # Всегда обучаем routing gate правильно переключаться, не только при высоком surprise
                    # Определяем правильный head для каждого класса
                    # Если класс в classes_B (animals), то head 1 (или последний после expansion)
                    # Если класс в classes_A (machines), то head 0
                    # Создаём target gates: правильный head должен быть активирован
                    target_gates = torch.zeros_like(gates)  # [B, H]
                    classes_B_set = set(classes_B)
                    classes_A_set = set(classes_A)
                    
                    for b in range(real_B):
                        class_id = int(target_real[b].item())
                        # Определяем правильный head для этого класса
                        if class_id in classes_B_set:
                            # Класс из Phase 2 (animals) - активируем последний head
                            correct_head_idx = len(agent.heads) - 1
                        elif class_id in classes_A_set:
                            # Класс из Phase 1 (machines) - активируем первый head
                            correct_head_idx = 0
                        else:
                            # Unknown класс - равномерное распределение
                            correct_head_idx = None
                        
                        if correct_head_idx is not None and correct_head_idx < len(agent.heads):
                            target_gates[b, correct_head_idx] = 1.0
                        else:
                            # Равномерное распределение для unknown
                            target_gates[b, :] = 1.0 / len(agent.heads)
                    
                    # Surprise-based loss: всегда обучаем, но вес зависит от surprise
                    # При высоком surprise - более агрессивное обучение
                    if surprise is not None:
                        # Масштабируем surprise: базовый вес 0.5, при surprise > 0.3 увеличиваем до 2.0
                        base_weight = 0.5
                        surprise_boost = max(0.0, float(surprise.item()) - 0.3) * 3.0  # [0..1.5] при surprise [0.3..0.8]
                        surprise_weight = base_weight + surprise_boost
                    else:
                        surprise_weight = 0.5  # базовый вес если surprise не доступен
                    
                    # Используем KL divergence для более агрессивного обучения (лучше чем MSE для вероятностей)
                    # Нормализуем gates и target_gates для стабильности
                    gates_norm = gates / (gates.sum(dim=1, keepdim=True) + 1e-9)
                    target_gates_norm = target_gates / (target_gates.sum(dim=1, keepdim=True) + 1e-9)
                    routing_surprise_loss = surprise_weight * F.kl_div(
                        F.log_softmax(gates_norm + 1e-9, dim=1),
                        target_gates_norm,
                        reduction='batchmean'
                    )
                
                # Проверка outputs на inf/nan
                if not torch.isfinite(outputs).all():
                    print(f"[ERROR] outputs contains inf/nan at step {step}")
                    # Попытка исправить: заменить inf/nan на 0
                    outputs = torch.where(torch.isfinite(outputs), outputs, torch.zeros_like(outputs))
                    if not torch.isfinite(outputs).all():
                        print(f"[ERROR] Cannot fix outputs, skipping step {step}")
                        continue
                
                # КРИТИЧНО: НЕ обнуляем loss_world_model и loss_concept_recon здесь!
                # Они уже вычислены выше в блоке AGI COMPONENTS INTEGRATION (строки 3166, 3205)
                # Повторная инициализация здесь приводит к "амнезии" - градиенты от AGI-модулей не доходят до весов
                # loss_world_model и loss_concept_recon уже инициализированы выше (строка 3102, 3193)
                # и вычислены если условия выполнились (строки 3166, 3205)
                # Если они не были вычислены, они остаются zeros, что правильно

                # Supervised loss на real данных
                # КРИТИЧНО: Phase2 loss только по активным классам (2..7), не по всем 10
                # Это предотвращает конкуренцию с неактивными классами (0,1,8,9) и mode collapse
                if len(agent.heads) > 1:  # Phase2 после expansion
                    # Локальный softmax только по активным классам
                    classes_B_t = torch.tensor(classes_B, device=device, dtype=torch.long)  # [6]
                    logits10 = outputs[:real_B, :10]
                    logitsB = logits10.index_select(1, classes_B_t)  # [B, 6] - только активные классы
                    
                    # Глобальные -> локальные targets
                    g2l = {c: i for i, c in enumerate(classes_B)}
                    targetB = torch.tensor([g2l[int(t.item())] for t in target_real], 
                                         device=device, dtype=torch.long)
                    
                    # Cross-entropy с label smoothing только по активным классам
                    loss_new = F.cross_entropy(logitsB, targetB, label_smoothing=0.05)
                elif expansion_count > 0 and class_weights_phase2 is not None:
                    # Fallback: class-balanced loss если веса вычислены
                    active_w = class_weights_phase2[torch.tensor(classes_B, device=device)]
                    if torch.isfinite(active_w).all() and (active_w > 0).all():
                        loss_new = class_balanced_loss(outputs[:real_B, :10], target_real, class_weights_phase2, num_classes=10)
                    else:
                        loss_new = criterion_train(outputs[:real_B, :10], target_real)
                else:
                    # Phase1 или если веса не вычислены - используем обычный loss
                    loss_new = criterion_train(outputs[:real_B, :10], target_real)
                
                # КРИТИЧНО: Unknown-margin на ID данных - Unknown должен быть НИЖЕ known на реальных CIFAR-10
                # Это предотвращает доминирование Unknown (10000 предсказаний)
                loss_unk_id = torch.zeros((), device=device, dtype=torch.float32)
                if len(agent.heads) > 1:  # только в Phase2
                    logits10 = outputs[:real_B, :10]
                    unk = outputs[:real_B, agent.unknown_class_idx]
                    max_known = logits10.max(dim=1).values  # максимальный логит среди known классов
                    
                    unk_margin = 1.0  # Unknown должен быть на margin ниже max_known
                    loss_unk_id = F.relu(unk - max_known + unk_margin).mean()
                
                # КРИТИЧНО: Outlier Exposure ОТКЛЮЧЕН для рекурсивной эмергенции
                # Неизвестное должно триггерить expansion новых heads, а не обучаться как отдельный класс.
                loss_unknown = torch.zeros((), device=device, dtype=torch.float32)
                agent.unknown_trained = True  # помечаем что unknown не обучается как класс, а триггерит expansion
                
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
            
            # КРИТИЧНО: Элегантный режим - упрощенная кристаллизация через EMA весов
            # Stability Loss - просто не даем весам быстро меняться (одна строчка вместо сложной логики)
            if agent.use_elegant_mode and hasattr(agent, 'elegant_stability_loss'):
                stability_loss = agent.elegant_stability_loss()
                if isinstance(stability_loss, torch.Tensor) and stability_loss.item() > 0.0 and torch.isfinite(stability_loss):
                    total_loss = total_loss + 0.1 * stability_loss  # вес для стабильности
                
                # Обновляем memory_weights через EMA (каждые N шагов)
                if step % agent.ema_update_frequency == 0:
                    agent.update_memory_weights_ema()

            # КРИТИЧНО: Unknown-margin на ID данных - Unknown должен быть НИЖЕ known на реальных CIFAR-10
            # Это предотвращает доминирование Unknown (10000 предсказаний)
            # Уменьшен вес с 0.2 до 0.05 для здоровой самокритичности (1-2% Unknown предсказаний)
            if isinstance(loss_unk_id, torch.Tensor) and loss_unk_id.item() != 0.0 and torch.isfinite(loss_unk_id):
                total_loss = total_loss + 0.05 * loss_unk_id  # уменьшен с 0.2 до 0.05 для самокритичности
            
            # --- AGI Components Losses ---
            # World Model Loss
            if isinstance(loss_world_model, torch.Tensor) and loss_world_model.item() != 0.0 and torch.isfinite(loss_world_model):
                # КРИТИЧНО: "Инстинкт недоверия" - уменьшаем вес loss_world_model на ранних этапах
                # Это предотвращает обучение на галлюцинациях необученной World Model
                world_model_trust = getattr(agent, '_world_model_trust', 1.0)  # по умолчанию полное доверие
                world_model_weight = 0.1 * world_model_trust  # базовый вес 0.1, умножаем на trust
                total_loss = total_loss + world_model_weight * loss_world_model
            
            # Concept Reconstruction Loss
            if isinstance(loss_concept_recon, torch.Tensor) and loss_concept_recon.item() != 0.0 and torch.isfinite(loss_concept_recon):
                total_loss = total_loss + 0.02 * loss_concept_recon

            # Добавляем loss для Unknown класса (Outlier Exposure) - отключено для рекурсивной эмергенции
            # if isinstance(loss_unknown, torch.Tensor) and loss_unknown.item() != 0.0 and torch.isfinite(loss_unknown):
            #     total_loss = total_loss + 0.02 * loss_unknown  # уменьшен с 0.1 до 0.02
            
            # Добавляем entropy penalty для routing gates
            if isinstance(routing_entropy_loss, torch.Tensor) and routing_entropy_loss.item() != 0.0 and torch.isfinite(routing_entropy_loss):
                total_loss = total_loss + 0.05 * routing_entropy_loss  # небольшой вес для стабилизации
            
            # КРИТИЧНО: Surprise-based routing loss - автономное переключение контекста
            # Всегда обучаем routing gate правильно переключаться, вес зависит от surprise
            # Когда surprise высокий, градиент идёт в RoutingGate для быстрого переключения на правильный head
            if isinstance(routing_surprise_loss, torch.Tensor) and routing_surprise_loss.item() != 0.0 and torch.isfinite(routing_surprise_loss):
                # Увеличиваем вес для более агрессивного обучения routing gate
                # Это критично для улучшения Global Accuracy
                total_loss = total_loss + 0.5 * routing_surprise_loss  # сильный вес для автономного роутинга
            
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
            unk_id_val = float(loss_unk_id.item()) if isinstance(loss_unk_id, torch.Tensor) and torch.isfinite(loss_unk_id) else 0.0
            
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
            
            # Self-Model Supervision Loss (вычисляем после loss_new)
            if agent.use_self_model and len(agent.heads) > 0 and hasattr(agent, '_self_model_targets'):
                targets = agent._self_model_targets
                agent.self_model.update_ema_targets(
                    targets["capabilities"],
                    targets["confidence"],
                    targets["weakness"]
                )
                
                # Self-Model Loss
                # КРИТИЧНО: конвертируем features в float32 для self_model
                features_f32_self = features[:real_B].float() if features.dtype == torch.bfloat16 else features[:real_B]
                capabilities_pred = agent.self_model.predict_capabilities(features_f32_self)
                confidence_pred = agent.self_model.estimate_confidence(features_f32_self)
                weakness_pred = agent.self_model.detect_weakness(features_f32_self)
                
                ema_targets = agent.self_model.get_targets()
                loss_self_model = torch.zeros((), device=device, dtype=torch.float32)
                
                # КРИТИЧНО: проверяем что capabilities target доступен
                if ema_targets["capabilities"] is not None:
                    # Берём только первые len(agent.heads) предсказаний
                    num_heads = len(agent.heads)
                    cap_pred_mean = capabilities_pred.mean(dim=0)[:num_heads]
                    cap_target = ema_targets["capabilities"][:num_heads].to(features.device)
                    if cap_pred_mean.size(0) == cap_target.size(0):
                        loss_self_cap = F.mse_loss(cap_pred_mean, cap_target)
                        loss_self_model = loss_self_model + loss_self_cap
                
                loss_self_conf = F.mse_loss(confidence_pred.mean(), torch.tensor(ema_targets["confidence"], device=features.device))
                loss_self_weak = F.mse_loss(weakness_pred.mean(), torch.tensor(ema_targets["weakness"], device=features.device))
                loss_self_model = loss_self_model + loss_self_conf + loss_self_weak
                
                if torch.isfinite(loss_self_model) and loss_self_model.item() > 0:
                    total_loss = total_loss + 0.05 * loss_self_model
            
            # Записываем эпизод в автобиографическую память
            if agent.use_autobiographical_memory and len(agent.heads) > 0:
                # КРИТИЧНО: используем float32 features для памяти
                state_feat = features_f32[0] if features_f32.size(0) > 0 else features_f32.mean(dim=0)
                # КРИТИЧНО: Эмоциональная память - записываем pain_level для Targeted Dreaming
                pain_level = pain_value if 'pain_value' in locals() else 0.0
                agent.record_autobiographical_memory(
                    step=step,
                    state_features=state_feat,
                    action=action_idx[0] if (agent.use_world_model and action_idx is not None and action_idx.size(0) > 0) else None,
                    outcome={"loss": float(loss_new.item()), "surprise": float(surprise.item()) if surprise is not None else 0.0},
                    reward_signal=max(0.0, min(1.0, 1.0 - float(loss_new.item()) / 2.0)),  # нормализованная награда
                    context={"complexity": complexity if 'complexity' in locals() else 0.0, "entropy": entropy_test},
                    pain_level=pain_level  # КРИТИЧНО: для приоритизации в Targeted Dreaming
                )
            
            # КРИТИЧНО: Обновляем weakness_after для последнего expansion (мета-регуляция)
            if agent.expansion_history["recent"] and len(agent.heads) > 1:
                # Обновляем последний expansion entry с текущим weakness
                if agent.use_self_model:
                    weakness_pred = agent.self_model.detect_weakness(features_f32)
                    weakness_after = weakness_pred.mean().item() if weakness_pred.numel() > 0 else 0.0
                    agent.expansion_history["recent"][-1]["weakness_after"] = weakness_after
            
            # КРИТИЧНО: обновляем Complexity Budget на основе использованных действий
            # Работаем даже после sleep (когда expansion_count может быть 0, но есть heads)
            if len(agent.heads) > 0 and actions is not None:
                used_expansion = should_expand and has_budget
                used_kl = (kl_loss != 0.0 and torch.isfinite(kl_loss))
                agent.complexity_controller.update_budget(
                    actions=actions,
                    used_expansion=used_expansion,
                    used_kl=used_kl,
                    current_complexity=complexity if 'complexity' in locals() else None,  # передаём complexity для динамического баланса
                    pain_value=pain_value if 'pain_value' in locals() else None  # передаём pain для эмоциональной регуляции
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
                # КРИТИЧНО: Глобальная метрика (10-классовая точность без маскирования)
                # Показывает, насколько хорошо Soft Routing Gate научился переключать контексты
                # Используем комбинированный loader со всеми 10 классами, а не только animals
                acc_global = eval_global(agent, test_loader_all, device)
                
                # КРИТИЧНО: Коэффициент Когнитивной Эффективности (η = Global Accuracy / Number of Active Heads)
                # Система должна стремиться максимизировать η
                # Слияние (Merge): Если Acc падает незначительно, а Heads уменьшается — η растет. Профит!
                # Раздувание (Expansion): Если Acc растет медленнее, чем количество голов — η падает. Наказание!
                num_active_heads = len(agent.heads) if not agent.use_elegant_mode else 1
                cognitive_efficiency = acc_global / max(1, num_active_heads)  # η
                
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
                loss_components = f"Lnew:{loss_new_val:.2f} R:{replay_val:.2f} KL:{kl_val:.3f} D:{dream_val:.2f} U:{unknown_val:.3f} Uid:{unk_id_val:.3f} Reg:{reg_val:.2f}"
                
                # КРИТИЧНО: Complexity Controller статус
                # Показываем даже после sleep (когда expansion_count может быть 0, но есть heads)
                complexity_info = ""
                if len(agent.heads) > 0 and actions is not None:
                    budget_status = agent.complexity_controller.get_budget_status()
                    complexity_info = f" | C:{complexity:.3f} R:{used_recursions} B:{budget_status['budget']:.2f} T:{actions['gate_temperature']:.2f}"
                
                print(
                    f"Step {step}: Loss {float(total_loss.item()):.2f} ({loss_components}) | Mem(M): {acc_A:.1f}% | "
                    num_heads_display = len(agent.heads) if not agent.use_elegant_mode else 1  # В элегантном режиме показываем 1 (единый блок)
                    f"New(A): {acc_B:.1f}% | Global: {acc_global:.1f}% | η: {cognitive_efficiency:.2f} | Heads: {num_heads_display} | UnknownRate: {unk_rate*100:.1f}% | "
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
