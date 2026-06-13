"""Track B — Transformer Fine-Tuning Pipeline.

Trains a user-selected transformer model on raw email text using
focal loss, FGM adversarial training, and optional curriculum learning.
Evaluates on the shared holdout split.

Model candidates ranked by expected Spam F1 on 342k balanced dataset:
  DeBERTa-v3 → 0.993-0.994
  RoBERTa    → 0.990-0.992
  ELECTRA    → 0.985-0.988
  ModernBERT → 0.984-0.987
  DistilBERT → 0.975-0.978
  BERT-base  → 0.978-0.981
"""

from __future__ import annotations

import gc
import importlib.util
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup,
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.shared import EvalMetrics, ram_report

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 64
EPOCHS = 3
MAX_LENGTH = 512
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
ADVERSARIAL_EPSILON = 0.5
ADVERSARIAL_ALPHA = 0.3
CURRICULUM_EPOCHS = 1
CURRICULUM_EASY_FRAC = 0.5
FP16_ENABLED = True
MAX_GRAD_NORM = 1.0


MODEL_IDS: dict[str, str] = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "DeBERTa-v3": "microsoft/deberta-v3-base",
    "DistilRoBERTa": "distilbert/distilroberta-base",
    "ELECTRA": "google/electra-base-discriminator",
    "ModernBERT": "answerdotai/ModernBERT-base",
}

MODEL_MAX_LENGTHS: dict[str, int] = {
    "DeBERTa-v3": 512, "DistilRoBERTa": 512, "RoBERTa": 514,
    "ELECTRA": 512, "ModernBERT": 8192, "DistilBERT": 512, "BERT": 512,
}


@dataclass
class TransformerConfig:
    model_name: str
    model_id: str
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    focal_gamma: float
    focal_alpha: float
    adversarial_epsilon: float
    adversarial_alpha: float
    curriculum_epochs: int
    curriculum_easy_frac: float
    fp16: bool
    max_grad_norm: float
    fast_dev_run: bool = False
    auto_batch_size: bool = False
    compile_model: bool = False


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if importlib.util.find_spec("torch_directml") is not None:
        return torch.device("directml")
    return torch.device("cpu")


def _get_ddp_config() -> tuple[bool, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1 or not torch.cuda.is_available():
        return False, 0, 1
    world_size = int(os.environ.get("WORLD_SIZE", int(torch.cuda.device_count())))
    if world_size < 2:
        return False, local_rank, 1
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    return True, local_rank, world_size


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _apply_fast_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")


def _get_optimal_workers() -> int:
    cpu_count = os.cpu_count() or 4
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        return min(max(cpu_count // (gpu_count * 2), 4), 8)
    return min(cpu_count, 8)


def _probe_vram_batch_size(
    model: nn.Module,
    tokenizer,
    max_length: int,
    device: torch.device,
    start_batch: int = 8,
    safety_margin: float = 0.85,
) -> int:
    total_vram = torch.cuda.get_device_properties(device).total_memory
    target_vram = int(total_vram * safety_margin)
    batch = start_batch
    while True:
        try:
            torch.cuda.empty_cache()
            dummy_ids = torch.randint(0, 1000, (batch, max_length), device=device)
            dummy_mask = torch.ones((batch, max_length), device=device)
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=dummy_ids, attention_mask=dummy_mask)
                loss = outputs.logits.sum()
            loss.backward()
            used = torch.cuda.max_memory_allocated(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            headroom = target_vram - used
            extra = headroom // (used // batch) if used > 0 else 4
            if extra >= 4:
                batch = min(batch + int(extra), 128)
            elif extra >= 1:
                batch += 1
            else:
                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            if batch <= 4:
                return max(batch - 2, 1)
            batch = max(batch // 2, start_batch)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return batch


def _detect_environment() -> str:
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle"):
        if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
            return "online"
        return "kaggle"
    if os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
        return "cached"
    if os.getenv("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        return "cached"
    return "online"


def _resolve_cache_dir() -> Path:
    env_cache = os.getenv("TRANSFORMERS_CACHE")
    if env_cache:
        return Path(env_cache)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cached_locally(model_id: str) -> bool:
    cache_dir = _resolve_cache_dir()
    model_dir = cache_dir / ("models--" + model_id.replace("/", "--"))
    if not model_dir.is_dir():
        return False
    if (model_dir / "refs" / "main").exists():
        return True
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir() and any(snapshots.iterdir()):
        return True
    return False


def _download_model_if_needed(model_id: str, env: str) -> None:
    if env != "online":
        return
    if _model_cached_locally(model_id):
        return
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, resume_download=True)
    except ImportError:
        pass


def _load_transformer_assets(
    config: TransformerConfig,
    env: str,
) -> tuple:
    device = _device()
    model_id = config.model_id
    cache_dir = str(_resolve_cache_dir())

    if env == "online":
        if not _model_cached_locally(model_id):
            print(f"  Downloading {model_id} from HuggingFace Hub...")
            _download_model_if_needed(model_id, env)
        else:
            print(f"  Found {model_id} in cache — using local copy")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
            hf_config = AutoConfig.from_pretrained(model_id, num_labels=2, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=2, ignore_mismatched_sizes=True, cache_dir=cache_dir,
            )
        except OSError as e:
            if not _model_cached_locally(model_id):
                raise OSError(
                    f"Failed to download {model_id}. Network unavailable and model not cached.\n"
                    f"  Solutions:\n"
                    f"  1. Set HF_HUB_OFFLINE=1 and pre-download the model\n"
                    f"  2. On Kaggle: download in an Interactive notebook first, then submit\n"
                    f"  3. Use --track-a-only for classical ML (no download needed)"
                ) from e
            print(f"  Download failed — falling back to cached {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True, cache_dir=cache_dir)
            hf_config = AutoConfig.from_pretrained(model_id, num_labels=2, local_files_only=True, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=2, ignore_mismatched_sizes=True, local_files_only=True, cache_dir=cache_dir,
            )

    elif env in ("cached", "kaggle"):
        local_kwargs = {"local_files_only": True, "cache_dir": cache_dir}
        if not _model_cached_locally(model_id):
            env_label = "Kaggle" if env == "kaggle" else "offline"
            raise FileNotFoundError(
                f"Model {model_id} not found in cache and {env_label} mode disables downloads.\n"
                f"  Expected: {_resolve_cache_dir() / ('models--' + model_id.replace('/', '--'))}\n"
                f"  Solutions:\n"
                f"  1. Pre-download on a machine with internet:\n"
                f"     python -c \"from transformers import AutoModelForSequenceClassification;\\\n"
                f"        AutoModelForSequenceClassification.from_pretrained('{model_id}')\"\n"
                f"  2. On Kaggle: run in Interactive mode first to cache the model\n"
                f"  3. Set TRANSFORMERS_CACHE to point to pre-downloaded models\n"
                f"  4. Use --track-a-only for classical ML (no download needed)"
            )
        print(f"  Loading {model_id} from local cache (offline mode)")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **local_kwargs)
        hf_config = AutoConfig.from_pretrained(model_id, num_labels=2, **local_kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, ignore_mismatched_sizes=True, **local_kwargs,
        )

    else:
        raise ValueError(f"Unknown environment: {env}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model = model.float()
    return tokenizer, hf_config, model, device


class EmailDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int,
                 pre_tokenize: bool = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._difficulty: np.ndarray | None = None
        self._pre_tokenized: tuple[np.ndarray, np.ndarray] | None = None

        if pre_tokenize:
            self._pre_tokenize_all()

    def __len__(self):
        return len(self.texts)

    def _pre_tokenize_all(self):
        all_ids: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        batch_size = 4096
        for i in range(0, len(self.texts), batch_size):
            batch = self.texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=self.max_length, return_tensors="np",
            )
            all_ids.append(enc["input_ids"])
            all_masks.append(enc["attention_mask"])
        self._pre_tokenized = (np.concatenate(all_ids), np.concatenate(all_masks))

    def __getitem__(self, idx):
        if self._pre_tokenized is not None and self._difficulty is None:
            input_ids, attn_mask = self._pre_tokenized
            return {
                "input_ids": torch.tensor(input_ids[idx], dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }
        text = self.texts[idx]
        if self._difficulty is not None:
            text = f"Difficulty: {self._difficulty[idx]:.2f}. {text}"
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def set_difficulty(self, difficulties: np.ndarray):
        self._difficulty = difficulties


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Mitigates the dominance of easy examples in spam datasets
    where simple keyword matches are abundant and the model
    needs to focus on hard (subtle phishing) samples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FGM:
    """Fast Gradient Method — adversarial training.

    Injects worst-case perturbation into embeddings during training.
    For spam, this makes the model robust to slight text variations
    that spammers use to evade keyword filters.

    This is explicitly aimed at the real adversarial domain of spam
    filtering — spammers actively modify text to evade detection,
    making adversarial training more impactful here than in typical NLP.
    """
    def __init__(self, model: nn.Module, epsilon: float = 0.5):
        self.model = model
        self.epsilon = epsilon
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self, emb_name: str = "word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name: str = "word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def _compute_difficulty_scores(texts: list[str]) -> np.ndarray:
    """Heuristic difficulty: longer + more URLs + more special chars = harder.

    Used for curriculum learning — start with short, clean emails
    and progressively introduce longer, messier ones.
    """
    scores = []
    for text in texts:
        score = len(text) / 500.0
        score += text.count("http") * 0.3
        score += text.count("@") * 0.2
        score += sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1) * 5.0
        scores.append(min(score, 10.0))
    return np.array(scores)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    fgm: FGM | None = None,
    scaler: Any | None = None,
    grad_accum: int = GRADIENT_ACCUMULATION_STEPS,
    ddp_model: Any | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    rank = _get_rank()

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False, disable=(rank != 0))):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        is_accum_boundary = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
        sync_ctx = ddp_model.no_sync() if ddp_model is not None and not is_accum_boundary else nullcontext()

        with sync_ctx:
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=scaler is not None):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels) / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if fgm is not None:
                fgm.attack()
                with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=scaler is not None):
                    adv_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    adv_loss = ADVERSARIAL_ALPHA * criterion(adv_outputs.logits, labels) / grad_accum
                if scaler is not None:
                    scaler.scale(adv_loss).backward()
                else:
                    adv_loss.backward()
                fgm.restore()

        total_loss += loss.item() * grad_accum

        if is_accum_boundary:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    rank = _get_rank()

    for batch in tqdm(loader, desc="Evaluating", leave=False, disable=(rank != 0)):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_preds.append(outputs.logits)
        all_labels.append(labels)

    logits = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy(), labels.cpu().numpy()


def train_transformer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TransformerConfig,
    *,
    checkpoint_dir: str | None = None,
) -> tuple[EvalMetrics, dict[str, Any], Any, Any]:
    use_ddp, local_rank, world_size = _get_ddp_config()
    is_main = _is_main_process()
    _apply_fast_cuda()

    if is_main:
        print("\n" + "=" * 60)
        print(f"  TRACK B — Transformer Fine-Tuning: {config.model_name}")
        print(f"  Model ID: {config.model_id}")
        if config.fast_dev_run:
            print("  FAST DEV RUN — 500 samples only")
        if use_ddp:
            print(f"  DDP: {world_size} GPUs (rank {local_rank})")
        print("=" * 60)

    env = _detect_environment()
    tokenizer, hf_config, model, device = _load_transformer_assets(config, env)

    if is_main:
        print(f"  Device: {device}")
        print(f"  FP16: {config.fp16 and device.type == 'cuda'}")
        if env != "online":
            print(f"  Env: {env} — offline, using cached model only")

    batch_size = config.batch_size
    if config.auto_batch_size and device.type == "cuda":
        searched = _probe_vram_batch_size(model, tokenizer, config.max_length, device, start_batch=batch_size)
        batch_size = searched
        if is_main:
            print(f"  Auto batch size: {config.batch_size} → {batch_size} (VRAM-optimized)")

    ddp_model = None
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, broadcast_buffers=False,
                    gradient_as_bucket_view=True)
        ddp_model = model

    if config.compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
        if is_main:
            print("  torch.compile: enabled (reduce-overhead)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if config.fast_dev_run:
        train_df = train_df.sample(n=min(500, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=42)

    train_texts = train_df["message"].tolist()
    test_texts = test_df["message"].tolist()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    train_dataset = EmailDataset(train_texts, y_train, tokenizer, config.max_length, pre_tokenize=True)
    test_dataset = EmailDataset(test_texts, y_test, tokenizer, config.max_length, pre_tokenize=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank,
                                       shuffle=True, drop_last=True) if use_ddp else None

    optimal_workers = _get_optimal_workers()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=optimal_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=True, prefetch_factor=4, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True,
        persistent_workers=False, prefetch_factor=4,
    )

    if is_main:
        effective_batch = batch_size * world_size * config.gradient_accumulation_steps
        print(f"  Train batches: {len(train_loader)}/gpu (effective batch={effective_batch} "
              f"= {batch_size} × {world_size}gpu × {config.gradient_accumulation_steps}accum)")
        print(f"  Test batches:  {len(test_loader)}")
        print(f"  Data workers:  {optimal_workers} (CPU cores: {os.cpu_count()})")
        print(f"  Total layers:  {hf_config.num_hidden_layers}")
        print(ram_report("Before model load"))
        print(ram_report("After model load"))

    criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, reduction="mean")
    fgm = FGM(model, epsilon=config.adversarial_epsilon) if config.adversarial_epsilon > 0 else None

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=config.learning_rate)
    total_steps = len(train_loader) // config.gradient_accumulation_steps * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", init_scale=16384.0,
                                   growth_interval=2000) if config.fp16 and device.type == "cuda" else None

    if is_main:
        print(f"\n  Training {config.epochs} epochs...")
        print(f"  Steps/rank: {total_steps} | Warmup: {warmup_steps} | Focal γ={config.focal_gamma}")
        if config.curriculum_epochs > 0:
            print(f"  Curriculum: {config.curriculum_epochs} epochs starting with "
                  f"{config.curriculum_easy_frac:.0%} easy samples")

    t0 = time.perf_counter()
    best_f1 = 0.0
    best_state = None
    ckpt_path = None
    if checkpoint_dir and is_main:
        ckpt_path = Path(checkpoint_dir) / f"{config.model_name}_best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if config.curriculum_epochs > 0 and epoch <= config.curriculum_epochs:
            difficulties = _compute_difficulty_scores(train_texts)
            sorted_idx = np.argsort(difficulties)
            keep_n = int(len(sorted_idx) * config.curriculum_easy_frac)
            easy_idx = sorted_idx[:keep_n]
            train_dataset.set_difficulty(difficulties)
            sub_dataset = torch.utils.data.Subset(train_dataset, easy_idx)
            sub_loader = DataLoader(
                sub_dataset, batch_size=config.batch_size, shuffle=True,
                num_workers=4, pin_memory=(device.type == "cuda"),
                persistent_workers=True, prefetch_factor=4,
            )
            active_loader = sub_loader
            if is_main:
                print(f"  Curriculum epoch {epoch}/{config.curriculum_epochs}: "
                      f"using {keep_n}/{len(train_texts)} easiest samples")
        else:
            train_dataset.set_difficulty(np.zeros(len(train_texts)))
            active_loader = train_loader

        avg_loss = train_epoch(model, active_loader, optimizer, scheduler, criterion, device,
                               fgm, scaler, grad_accum=config.gradient_accumulation_steps,
                               ddp_model=ddp_model)

        probs, labels = evaluate_model(model, test_loader, device)
        preds = probs.argmax(axis=1)
        from sklearn.metrics import f1_score
        epoch_f1 = f1_score(labels, preds, pos_label=1)

        if is_main:
            print(f"  Epoch {epoch}/{config.epochs} | Loss: {avg_loss:.4f} | "
                  f"Spam F1: {epoch_f1:.4f} | {ram_report('')}")
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if ckpt_path is not None:
                    torch.save(best_state, ckpt_path)
                    print(f"  Checkpoint saved to {ckpt_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    elif use_ddp:
        if is_main:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0

    probs, labels = evaluate_model(model, test_loader, device)
    preds = probs.argmax(axis=1)

    if not is_main:
        return EvalMetrics(model_name=config.model_name, accuracy=0, spam_precision=0,
                          spam_recall=0, spam_f1=0, roc_auc=None, train_time_seconds=train_time,
                          support=0, track="transformer", eval_method="holdout"), {}, model, tokenizer

    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    report = classification_report(labels, preds, target_names=["Ham", "Spam"],
                                   output_dict=True, zero_division=0)
    try:
        roc_auc = float(roc_auc_score(labels, probs[:, 1]))
    except ValueError:
        roc_auc = None

    spam_metrics = report["Spam"]
    cm = confusion_matrix(labels, preds)

    metrics = EvalMetrics(
        model_name=config.model_name,
        track="transformer",
        accuracy=float(report["accuracy"]),
        spam_precision=float(spam_metrics["precision"]),
        spam_recall=float(spam_metrics["recall"]),
        spam_f1=float(spam_metrics["f1-score"]),
        roc_auc=roc_auc,
        train_time_seconds=train_time,
        support=int(spam_metrics["support"]),
        confusion_matrix=cm.tolist(),
        eval_method="holdout",
    )

    print(f"\n--- [{metrics.track}] {metrics.model_name} ---")
    print(f"Accuracy        : {metrics.accuracy:.4f}")
    print(f"Spam F1         : {metrics.spam_f1:.4f}")
    print(f"Spam Precision  : {metrics.spam_precision:.4f}")
    print(f"Spam Recall     : {metrics.spam_recall:.4f}")
    print(f"ROC-AUC         : {metrics.roc_auc}")
    print(f"Train time      : {metrics.train_time_seconds:.1f}s")
    print("Confusion matrix:")
    print(cm)

    package_info = {
        "model_name": config.model_name,
        "model_id": config.model_id,
        "max_length": config.max_length,
        "focal_gamma": config.focal_gamma,
        "adversarial_epsilon": config.adversarial_epsilon,
        "curriculum_epochs": config.curriculum_epochs,
    }

    return metrics, package_info, model, tokenizer


def get_transformer_config(model_name: str, fast_dev_run: bool = False) -> TransformerConfig:
    model_id = MODEL_IDS.get(model_name)
    if model_id is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_IDS)}")
    return TransformerConfig(
        model_name=model_name,
        model_id=model_id,
        max_length=MODEL_MAX_LENGTHS.get(model_name, 512),
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        focal_gamma=FOCAL_GAMMA,
        focal_alpha=FOCAL_ALPHA,
        adversarial_epsilon=ADVERSARIAL_EPSILON,
        adversarial_alpha=ADVERSARIAL_ALPHA,
        curriculum_epochs=CURRICULUM_EPOCHS,
        curriculum_easy_frac=CURRICULUM_EASY_FRAC,
        fp16=FP16_ENABLED,
        max_grad_norm=MAX_GRAD_NORM,
        fast_dev_run=fast_dev_run,
        auto_batch_size=torch.cuda.is_available(),
        compile_model=torch.cuda.is_available() and hasattr(torch, 'compile'),
    )
