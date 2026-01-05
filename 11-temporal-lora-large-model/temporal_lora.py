# =========================
# Temporal LoRA: Scaling to Large Language Models with Time Mixer
# =========================
"""
Recursive time theory for large language models:
- Backbone (LLaMA-3, Mistral, etc.): "Eternity" - frozen
- LoRA A (Shakespeare): "Renaissance Era"
- LoRA B (Python): "IT Era"
- Time Mixer: Dynamic switching between epochs
"""

import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Dict, Optional, Tuple

# -------------------------
# Settings
# -------------------------
SEED = 42
FAST_MODE = True  # True = quick test, False = full experiment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Detected device: {DEVICE}", flush=True)
print(f"FAST_MODE: {FAST_MODE}", flush=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -------------------------
# LoRA Adapter (same as GPT-2 version)
# -------------------------
class LoRAAdapter(nn.Module):
    """
    LoRA adapter for one temporal domain.
    Implements low-rank adaptation: W' = W + alpha * (A @ B)
    """
    def __init__(self, base_dim: int, rank: int = 8, alpha: float = 16.0, name: str = "", dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.name = name
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices (low-rank decomposition) с правильным dtype
        if dtype is None:
            dtype = torch.float32
        self.lora_A = nn.Parameter(torch.randn(rank, base_dim, dtype=dtype) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(base_dim, rank, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns LoRA delta (adaptation): Δ = scaling * (x @ A^T @ B^T)
        """
        lora_adaptation = (x @ self.lora_A.T) @ self.lora_B.T
        return self.scaling * lora_adaptation
    
    def get_adapter_weights(self) -> torch.Tensor:
        """Returns adapter weights for analysis"""
        return self.scaling * (self.lora_B @ self.lora_A)

# -------------------------
# Time Mixer: Mechanism for switching between epochs
# -------------------------
class TimeMixer(nn.Module):
    """
    Time Mixer v2: Uses input embeddings to determine domain.
    Adapted for large language models (larger vocab_size).
    """
    def __init__(self, hidden_dim: int, num_adapters: int, vocab_size: int = 128256, strategy: str = "gating"):
        super().__init__()
        self.num_adapters = num_adapters
        self.strategy = strategy
        self.hidden_dim = hidden_dim
        
        # Own embedding for domain classification (lightweight - 64 dim)
        embed_dim = 64
        self.domain_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Domain classifier works at token level
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_adapters)
        )
        
        # Also keep gating for backward compatibility
        if strategy == "gating":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_adapters),
                nn.Softmax(dim=-1)
            )
    
    def reset_weights(self):
        """Reinitializes weights to remove bias towards first adapter"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.domain_classifier.apply(init_weights)
        # Important: reinitialize embeddings with normal distribution
        nn.init.normal_(self.domain_embed.weight, mean=0.0, std=0.02)
        
        if hasattr(self, 'gate'):
            self.gate.apply(init_weights)
    
    def forward(self, hidden_states: torch.Tensor, adapter_outputs: List[torch.Tensor], 
                input_ids: torch.Tensor = None, return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            adapter_outputs: List of [batch, seq_len, hidden_dim]
            input_ids: [batch, seq_len] - CRITICAL for domain determination!
            return_logits: If True, also return domain_logits (before softmax)
        
        Returns:
            mixed_delta: [batch, seq_len, hidden_dim]
            weights: [batch, seq_len, num_adapters]
            domain_logits: [batch, seq_len, num_adapters] (only if return_logits=True)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # NEW LOGIC: use input_ids to determine domain
        if input_ids is not None:
            # Get lightweight token embeddings
            token_embeds = self.domain_embed(input_ids)  # [batch, seq_len, 64]
            # Classify domain by tokens
            domain_logits = self.domain_classifier(token_embeds)  # [batch, seq_len, num_adapters]
            weights = torch.softmax(domain_logits, dim=-1)  # [batch, seq_len, num_adapters]
        else:
            # Fallback to old logic (hidden_states based)
            domain_logits = None
            if hasattr(self, 'gate'):
                weights = self.gate(hidden_states)
                # Привести к dtype hidden_states
                weights = weights.to(hidden_states.dtype)
            else:
                # Uniform weights if no gate
                weights = torch.ones(batch_size, seq_len, self.num_adapters, 
                                   device=hidden_states.device, dtype=hidden_states.dtype)
                weights = weights / self.num_adapters
        
        # Weighted combination of adapter deltas (Δ)
        # Привести weights к dtype hidden_states для совместимости
        weights = weights.to(hidden_states.dtype)
        mixed_delta = torch.zeros_like(hidden_states)
        for i, adapter_delta in enumerate(adapter_outputs):
            # Weight each adapter's delta by its domain probability
            mixed_delta += weights[:, :, i:i+1] * adapter_delta
        
        if return_logits:
            return mixed_delta, weights, domain_logits
        return mixed_delta, weights

# -------------------------
# Temporal LoRA Model for Large Language Models
# -------------------------
class TemporalLoRAModel(nn.Module):
    """
    Main model with frozen large language model backbone and multiple LoRA adapters.
    Supports LLaMA-3, Mistral, and other transformer architectures.
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        mixer_strategy: str = "gating",
        freeze_backbone: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        
        # Определяем dtype для B200
        if torch_dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        
        # Load base model
        print(f"[INFO] Loading model: {model_name}", flush=True)
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Freeze backbone (Eternity)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[OK] Backbone frozen (Eternity)", flush=True)
        
        # Large models use config.hidden_size instead of n_embd
        self.hidden_dim = getattr(self.config, 'hidden_size', getattr(self.config, 'n_embd', 4096))
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.model_dtype = torch_dtype  # Save model dtype for adapters
        
        # Dictionary of LoRA adapters for different epochs
        self.adapters: Dict[str, LoRAAdapter] = nn.ModuleDict()
        
        # Time Mixer
        self.time_mixer = None  # Initialized after adding adapters
        
        # List of adapter names for ordering
        self.adapter_names: List[str] = []
        
    def add_adapter(self, name: str, epoch_description: str = ""):
        """Adds a new LoRA adapter for a temporal epoch"""
        if name in self.adapters:
            print(f"[WARN] Adapter '{name}' already exists", flush=True)
            return
        
        # Create adapters for each transformer layer с правильным dtype
        adapter = LoRAAdapter(
            base_dim=self.hidden_dim,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            name=name,
            dtype=self.model_dtype  # Использовать тот же dtype что и модель
        )
        
        # Move adapter to same device as backbone
        try:
            device = next(self.backbone.parameters()).device
            adapter = adapter.to(device)
        except StopIteration:
            # If backbone has no parameters (shouldn't happen), use CUDA
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            adapter = adapter.to(device)
        
        self.adapters[name] = adapter
        self.adapter_names.append(name)
        
        # Initialize Time Mixer after first adapter
        vocab_size = getattr(self.config, 'vocab_size', 128256)  # Large model vocab_size
        if self.time_mixer is None:
            self.time_mixer = TimeMixer(
                hidden_dim=self.hidden_dim,
                num_adapters=1,
                vocab_size=vocab_size,
                strategy="gating"
            )
            # Move Time Mixer to correct device
            self.time_mixer = self.time_mixer.to(device)
        
        # Update Time Mixer for new number of adapters
        if len(self.adapters) > 1:
            self.time_mixer = TimeMixer(
                hidden_dim=self.hidden_dim,
                num_adapters=len(self.adapters),
                vocab_size=vocab_size,
                strategy="gating"
            )
            # Move Time Mixer to correct device
            self.time_mixer = self.time_mixer.to(device)
        
        print(f"[OK] Added adapter '{name}' ({epoch_description})", flush=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_mixer: bool = True,
        adapter_weights: Optional[Dict[str, float]] = None,
        return_mixer_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with LoRA adapters and Time Mixer applied.
        Adapted for large transformer architectures (LLaMA, Mistral, etc.).
        """
        # Use standard model forward for proper RoPE handling
        # This ensures all parameters (position_embeddings, etc.) are processed correctly
        inputs_embeds = self.backbone.model.embed_tokens(input_ids)
        
        # Создаем position_ids для RoPE
        seq_len = inputs_embeds.size(1)
        batch_size = inputs_embeds.size(0)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        
        # Collect Time Mixer weights for each layer
        all_mixer_weights = []
        
        # Use pre_hook to apply LoRA before each block
        # This allows modifying hidden_states before the block processes them
        def make_hook(layer_idx):
            def pre_hook(module, input_tuple):
                # input_tuple содержит (hidden_states, attention_mask, position_ids, ...)
                # Для Mistral/LLaMA первый аргумент - это hidden_states
                if not isinstance(input_tuple, tuple) or len(input_tuple) == 0:
                    return input_tuple
                
                hidden_states = input_tuple[0]
                
                # Apply LoRA adapters to hidden states before block
                adapter_outputs = []
                for name in self.adapter_names:
                    adapter = self.adapters[name]
                    adapted = adapter(hidden_states)
                    adapter_outputs.append(adapted)
                
                # Time Mixer or simple summation
                if use_mixer and self.time_mixer is not None and len(adapter_outputs) > 1:
                    mixed_delta, mixer_weights = self.time_mixer(hidden_states, adapter_outputs, input_ids=input_ids)
                    modified_hidden = hidden_states + mixed_delta
                    if return_mixer_weights:
                        all_mixer_weights.append(mixer_weights)
                elif adapter_weights is not None:
                    modified_hidden = hidden_states.clone()
                    for i, (name, weight) in enumerate(adapter_weights.items()):
                        if name in self.adapter_names:
                            idx = self.adapter_names.index(name)
                            modified_hidden += weight * adapter_outputs[idx]
                elif len(adapter_outputs) > 0:
                    modified_hidden = hidden_states.clone()
                    for adapter_delta in adapter_outputs:
                        modified_hidden += adapter_delta / len(adapter_outputs)
                else:
                    modified_hidden = hidden_states
                
                # Return modified input tuple
                return (modified_hidden,) + input_tuple[1:]
            return pre_hook
        
        # Register pre_hooks for each layer
        hooks = []
        for layer_idx, layer in enumerate(self.backbone.model.layers):
            hook = layer.register_forward_pre_hook(make_hook(layer_idx))
            hooks.append(hook)
        
        try:
            # Use standard model forward - это правильно обработает RoPE и все параметры
            outputs = self.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
            )
            
            hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Large models use model.norm before lm_head
        if hasattr(self.backbone.model, 'norm'):
            hidden_states = self.backbone.model.norm(hidden_states)
        
        # Get logits
        logits = self.backbone.lm_head(hidden_states)
        
        result = {"logits": logits}
        if return_mixer_weights and all_mixer_weights:
            # Use last layer weights (closest to output, most refined decisions)
            result["mixer_weights"] = all_mixer_weights[-1]  # [batch, seq_len, num_adapters]
        
        return result
    
    def get_adapter_params(self):
        """Returns only adapter parameters (for optimization)"""
        params = []
        for adapter in self.adapters.values():
            params.extend(list(adapter.parameters()))
        if self.time_mixer is not None:
            params.extend(list(self.time_mixer.parameters()))
        return params
    
    def freeze_adapter(self, name: str):
        """Freezes a specific adapter"""
        if name in self.adapters:
            for param in self.adapters[name].parameters():
                param.requires_grad = False
            print(f"[OK] Adapter '{name}' frozen", flush=True)
    
    def unfreeze_adapter(self, name: str):
        """Unfreezes a specific adapter"""
        if name in self.adapters:
            for param in self.adapters[name].parameters():
                param.requires_grad = True
            print(f"[OK] Adapter '{name}' unfrozen", flush=True)

# -------------------------
# Dataset for different domains
# -------------------------
class DomainDataset(Dataset):
    """Dataset for one temporal domain"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0)
        }

# -------------------------
# Data generation for different epochs
# -------------------------
def generate_shakespeare_data(n_samples: int = 100) -> List[str]:
    """Generates data in Shakespeare style (Renaissance Era)"""
    templates = [
        "To code or not to code, that is the question.",
        "What light through yonder window breaks? It is the east, and Python is the sun.",
        "Romeo, Romeo, wherefore art thou, Romeo? In the repository, my love.",
        "All the world's a stage, and all the developers merely players.",
        "Friends, Romans, countrymen, lend me your functions!",
        "A rose by any other name would smell as sweet, but a variable must be properly named.",
        "Double, double toil and trouble; Fire burn and cauldron bubble. The code compiles, the tests pass.",
        "Is this a dagger which I see before me? No, 'tis but a pointer to memory.",
        "Out, out, brief candle! Life's but a walking shadow, a poor player that struts and frets his hour upon the stage of GitHub.",
        "The fault, dear Brutus, is not in our stars, but in ourselves, that we are underlings... of the compiler."
    ]
    
    # Generate variations
    data = []
    for _ in range(n_samples):
        base = random.choice(templates)
        # Add small variations
        if random.random() < 0.3:
            base = base.replace(".", "!")
        if random.random() < 0.2:
            base = base.lower()
        data.append(base)
    
    return data

def generate_python_data(n_samples: int = 100) -> List[str]:
    """Generates Python programming data (IT Era)"""
    templates = [
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "class TemporalLoRA: def __init__(self): self.adapters = {}",
        "import torch; model = TemporalLoRA(); optimizer = optim.Adam(model.parameters())",
        "for epoch in range(num_epochs): loss = train_step(); print(f'Epoch {epoch}: {loss}')",
        "def time_mixer(hidden_states, adapters): return weighted_sum(adapters)",
        "if __name__ == '__main__': main()",
        "tensor = torch.randn(batch_size, seq_len, hidden_dim)",
        "loss.backward(); optimizer.step(); optimizer.zero_grad()",
        "model.eval(); with torch.no_grad(): predictions = model(inputs)",
        "def forward(self, x): return self.layer(self.activation(x))"
    ]
    
    data = []
    for _ in range(n_samples):
        base = random.choice(templates)
        # Add comments
        if random.random() < 0.4:
            base = f"# {base}"
        data.append(base)
    
    return data

# -------------------------
# Training
# -------------------------
def train_adapter(
    model: TemporalLoRAModel,
    tokenizer,
    dataset: DomainDataset,
    adapter_name: str,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
    previous_datasets: Optional[List[DomainDataset]] = None,
    teacher_mixer: Optional[TimeMixer] = None,
    use_active_sleep: bool = True
):
    """
    Trains one LoRA adapter on a specific domain.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Training adapter '{adapter_name}' (Epoch: {adapter_name})", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Ensure adapter exists
    if adapter_name not in model.adapters:
        model.add_adapter(adapter_name)
    
    # Unfreeze only the needed adapter
    for name in model.adapter_names:
        if name == adapter_name:
            model.unfreeze_adapter(name)
        else:
            model.freeze_adapter(name)
    
    # Unfreeze Time Mixer
    if model.time_mixer is not None:
        for param in model.time_mixer.parameters():
            param.requires_grad = True
    
    # Optimizer only for active adapter and Time Mixer
    optimizer = optim.Adam(model.get_adapter_params(), lr=lr)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Preparation for Active Sleep
    previous_loaders = None
    if use_active_sleep and previous_datasets is not None and len(previous_datasets) > 0:
        previous_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in previous_datasets]
        print(f"[INFO] Active Sleep enabled: {len(previous_datasets)} previous epochs", flush=True)
    
    model.train()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    T = 2.0  # Temperature for distillation
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_adapter_loss = 0.0
        total_mixer_distill_loss = 0.0
        n_batches = 0
        
        # Create iterators for previous epochs
        previous_iterators = None
        if previous_loaders:
            previous_iterators = [iter(loader) for loader in previous_loaders]
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass with Time Mixer
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_mixer=True,
                return_mixer_weights=True
            )
            
            logits = outputs["logits"]
            
            # Shift for language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Main loss for current adapter
            adapter_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Active Sleep for Time Mixer (simplified for large models)
            mixer_distill_loss = 0.0
            if (use_active_sleep and teacher_mixer is not None and 
                previous_iterators is not None and len(previous_iterators) > 0 and
                model.time_mixer is not None):
                # Simplified Active Sleep (can be enhanced)
                for prev_iter in previous_iterators:
                    try:
                        prev_batch = next(prev_iter)
                        prev_input_ids = prev_batch["input_ids"].to(DEVICE)
                        
                        # Get hidden states for previous epoch
                        with torch.no_grad():
                            prev_embeds = model.backbone.model.embed_tokens(prev_input_ids)
                            prev_hidden = prev_embeds
                            
                            # Get weights from teacher
                            prev_adapter_outputs = []
                            for name in model.adapter_names:
                                if name != adapter_name:
                                    adapter = model.adapters[name]
                                    prev_adapter_outputs.append(adapter(prev_hidden))
                            
                            if len(prev_adapter_outputs) > 0:
                                _, teacher_weights = teacher_mixer(prev_hidden, prev_adapter_outputs, input_ids=prev_input_ids)
                            else:
                                continue
                        
                        # Get weights from current Time Mixer
                        current_adapter_outputs = []
                        for name in model.adapter_names:
                            if name != adapter_name:
                                adapter = model.adapters[name]
                                current_adapter_outputs.append(adapter(prev_hidden))
                        
                        if len(current_adapter_outputs) > 0 and len(prev_adapter_outputs) > 0:
                            _, current_weights = model.time_mixer(prev_hidden, current_adapter_outputs, input_ids=prev_input_ids)
                            
                            # Distillation of Time Mixer weights
                            teacher_probs = F.softmax(teacher_weights.mean(dim=1) / T, dim=-1)
                            current_log_probs = F.log_softmax(current_weights.mean(dim=1) / T, dim=-1)
                            
                            mixer_distill_loss += (T * T) * kl_loss_fn(current_log_probs, teacher_probs)
                            break
                    except (StopIteration, RuntimeError, KeyError):
                        pass
            
            # Total loss
            loss = adapter_loss + 0.5 * mixer_distill_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_adapter_loss += adapter_loss.item()
            total_mixer_distill_loss += mixer_distill_loss.item() if isinstance(mixer_distill_loss, torch.Tensor) else mixer_distill_loss
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_adapter_loss = total_adapter_loss / n_batches
        avg_mixer_loss = total_mixer_distill_loss / n_batches
        mixer_info = f" | Mixer Distill: {avg_mixer_loss:.4f}" if avg_mixer_loss > 0 else ""
        print(f"Epoch {epoch}/{epochs} | Total Loss: {avg_loss:.4f} | Adapter: {avg_adapter_loss:.4f}{mixer_info}", flush=True)
    
    # Freeze adapter after training
    model.freeze_adapter(adapter_name)
    print(f"[OK] Adapter '{adapter_name}' trained and frozen", flush=True)

# -------------------------
# Phase 3: Time Mixer Calibration
# -------------------------
def calibrate_mixer(
    model: TemporalLoRAModel,
    tokenizer,
    datasetA: DomainDataset,
    datasetB: DomainDataset,
    adapter_nameA: str,
    adapter_nameB: str,
    epochs: int = 5,
    lr: float = 2e-3,
    batch_size: int = 16
):
    """
    Final calibration of Time Mixer using NLL Loss.
    """
    print("\n" + "="*80, flush=True)
    print("PHASE 3: Final Calibration (Contrastive Calibration)", flush=True)
    print("="*80, flush=True)
    
    # 1. Check adapter indices
    if adapter_nameA not in model.adapter_names or adapter_nameB not in model.adapter_names:
        print(f"[ERROR] Adapters {adapter_nameA} or {adapter_nameB} not found in model!", flush=True)
        return

    idx_A = model.adapter_names.index(adapter_nameA)
    idx_B = model.adapter_names.index(adapter_nameB)
    
    print(f"[INFO] Mapping: '{adapter_nameA}' -> Index {idx_A}", flush=True)
    print(f"[INFO] Mapping: '{adapter_nameB}' -> Index {idx_B}", flush=True)
    print("[INFO] Freezing adapters, training only Time Mixer to distinguish epochs", flush=True)
    
    # 2. Freeze everything except Time Mixer
    for name in model.adapter_names:
        model.freeze_adapter(name)
    
    if model.time_mixer is not None:
        print("[INFO] Resetting Time Mixer weights (removing initial bias)", flush=True)
        model.time_mixer.reset_weights()
        for param in model.time_mixer.parameters():
            param.requires_grad = True
    else:
        print("[ERROR] Time Mixer not found!", flush=True)
        return
    
    # 3. Create mixed dataset
    mixed_texts = []
    domain_labels = []
    
    for i in range(len(datasetA)):
        mixed_texts.append(datasetA.texts[i])
        domain_labels.append(idx_A)
    
    for i in range(len(datasetB)):
        mixed_texts.append(datasetB.texts[i])
        domain_labels.append(idx_B)
    
    # Shuffle
    combined = list(zip(mixed_texts, domain_labels))
    random.shuffle(combined)
    mixed_texts, domain_labels = zip(*combined)
    
    # Dataset
    mixed_dataset = DomainDataset(list(mixed_texts), tokenizer, max_length=64)
    
    class LabeledDataset(Dataset):
        def __init__(self, dataset, labels):
            self.dataset = dataset
            self.labels = labels
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            item = self.dataset[idx]
            item["target_adapter_idx"] = self.labels[idx]
            return item

    labeled_ds = LabeledDataset(mixed_dataset, list(domain_labels))
    loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    
    # 4. Training
    optimizer = optim.Adam(model.time_mixer.parameters(), lr=lr)
    
    print(f"[INFO] Starting calibration: {len(labeled_ds)} examples, {len(loader)} batches", flush=True)
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        print(f"[INFO] Epoch {ep}/{epochs}: processing batches...", flush=True)
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}...", end="\r", flush=True)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target_idx = batch["target_adapter_idx"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Use domain_classifier directly (faster)
            token_embeds = model.time_mixer.domain_embed(input_ids)
            domain_logits = model.time_mixer.domain_classifier(token_embeds)
            domain_probs = torch.softmax(domain_logits, dim=-1)
            
            # Masked tail-weighted aggregation
            B, S, K = domain_probs.shape
            attention_mask_float = attention_mask.float()
            
            pos = torch.arange(S, device=DEVICE).float().unsqueeze(0).expand(B, S)
            lengths = attention_mask_float.sum(dim=1).clamp(min=1.0)
            last_pos = (lengths - 1.0).unsqueeze(1)
            
            tau = 0.35
            dist_to_end = (pos - last_pos)
            time_weights = torch.exp(dist_to_end / (tau * S))
            time_weights = time_weights * attention_mask_float
            time_weights = time_weights / (time_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            agg_weights = (domain_probs * time_weights.unsqueeze(-1)).sum(dim=1)
            
            # Loss: NLL Loss on aggregated probabilities
            eps = 1e-8
            agg_weights = torch.clamp(agg_weights, eps, 1.0)
            loss = F.nll_loss(torch.log(agg_weights), target_idx)
            
            loss.backward()
            optimizer.step()
            
            # Calculate router classification accuracy
            pred_idx = agg_weights.argmax(dim=1)
            acc = (pred_idx == target_idx).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1
        
        print(f"\n[OK] Calibration Epoch {ep}/{epochs} | Router Loss: {total_loss/n_batches:.4f} | Router Acc: {total_acc/n_batches*100:.1f}%", flush=True)

    print(f"[OK] Time Mixer recalibrated", flush=True)

