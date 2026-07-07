# Kaggle Training & Recovery Guide

This guide covers training the Spam Email Detection pipeline on Kaggle with GPU acceleration, including checkpoint recovery procedures if training is interrupted.

---

## Kaggle Notebook Setup

### 1. Create a Kaggle Notebook

- Go to [kaggle.com](https://www.kaggle.com/) → Code → New Notebook
- **File** → **Notebook options** → Select **GPU T4 x2** accelerator
- **Persistence**: Notebook outputs persist for 9 hours in interactive mode; batch mode sessions have separate limits

### 2. Attach the Dataset

In the notebook's **Add Data** panel, attach your spam dataset. The training pipeline auto-discovers CSV files from:

1. `KAGGLE_INPUT_DIR` environment variable
2. `/kaggle/input/` directory
3. Manual `--csv-path` CLI argument

**Expected dataset**: A CSV file with columns `label` (values: `spam`/`ham` or `0`/`1`) and `text` (email body).

### 3. Clone the Repository

```bash
!git clone https://github.com/AVijit005/Spam-Email-Detection.git
!cd Spam-Email-Detection && pip install -r requirements.txt
```

### 4. Verify GPU Availability

```bash
!python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
Device: Tesla T4
```

---

## Training Commands

### Full Ensemble Training (XGBoost + DeBERTa-v3)

```bash
python model/train_model.py --model DeBERTa-v3
```

This runs the complete 6-stage pipeline. Requires GPU. Expected runtime: ~2.5 hours.

### Classical Only (No GPU Required)

```bash
python model/train_model.py --track-a-only --competition
```

Runs only Track A (XGBoost, SGD, LightGBM with Optuna). CPU-only. Expected runtime: ~35 minutes.

### Transformer Only

```bash
python model/train_model.py --track-b-only --model DeBERTa-v3
```

Runs only Track B (transformer fine-tuning). Requires GPU. Expected runtime: ~90 minutes.

### Fast Development Run (500 Rows, ~5 Minutes)

```bash
python model/train_model.py --fast-dev --model DeBERTa-v3
```

Uses only 500 rows. All stages run but complete quickly. Use this to verify the pipeline works before a full run.

### Custom CSV Path

```bash
python model/train_model.py --csv-path /kaggle/input/your-dataset/spam.csv
```

### Skip Optuna (Faster Classical Training)

```bash
python model/train_model.py --skip-optuna --track-a-only
```

Skips hyperparameter optimization. Uses default parameters. Stage 2 runtime: ~5 minutes instead of ~35.

### Custom Output Directory

```bash
python model/train_model.py --output-dir /kaggle/working/models
```

### Competition Mode

```bash
python model/train_model.py --competition
```

Increases TF-IDF `max_features` to 50,000 and `ngram_range` to (1, 3) for maximum coverage on large datasets.

---

## GPU & VRAM Requirements

| Model (fp16) | Min GPU | VRAM Required | Notes |
|---|---|---|---|
| **DeBERTa-v3-base** | T4 | 8 GB | Primary model; probes VRAM and selects batch size automatically |
| DeBERTa-v3-base (fp32) | A10G / A100 | 16 GB | Without mixed precision |
| RoBERTa-base | T4 | 6 GB | Smaller than DeBERTa |
| ELECTRA-base | T4 | 6 GB | Fastest transformer option |
| ModernBERT-base | T4 | 6 GB | Good for long emails |
| DistilBERT-base | T4 | 4 GB | Smallest; runs on limited GPUs |
| Classical only (Track A) | None | 0 GB | CPU-only, no GPU needed |

### Kaggle T4 Configuration

- **GPU**: Tesla T4 (16 GB VRAM)
- **Enable**: Notebook Settings → Accelerator → GPU T4 x2
- **Availability**: T4 GPUs are shared on Kaggle. Expect queue time during peak hours (sometimes 5-15 minutes to start a session).
- **Timeout**: Interactive sessions auto-stop after 9 hours of runtime. Batch mode jobs have configurable limits.

## RAM Requirements

| Stage | Peak RAM | Notes |
|---|---|---|
| Stage 1 — Load | ~2 GB | 342,178 rows in pandas DataFrame |
| Stage 2 — Classical | ~8 GB | TF-IDF sparse matrix + XGBoost candidates |
| Stage 3 — Transformer | ~12 GB | Model weights + DataLoader + activations |
| Stage 4 — Ensemble | ~6 GB | Sparse features + probability arrays |
| Stage 5 — Retrain | ~8 GB | Full dataset TF-IDF + XGBoost fit |
| **Peak Combined** | **~16 GB** | Stages 2 + 3 overlap in memory |

Kaggle notebooks typically provide 13-16 GB RAM. If you encounter OOM errors, try:
- Run `--track-a-only` first, then `--track-b-only` separately
- Reduce batch size via VRAM probing (automatic)
- Use `--fast-dev` to verify everything works before full run

---

## Expected Runtime (342,178 Rows)

| Stage | Hardware | Time | Can Resume? |
|---|---|---|---|
| Stage 1 — Load & Preprocess | CPU | ~45s | No (restart from Stage 1) |
| Stage 2 — Classical (3 candidates) | 8-core CPU | ~5 min (default) or ~35 min with Optuna |
| Stage 3 — Transformer (3 epochs, fp16) | T4 GPU | ~60-90 min | **Yes** (checkpoint auto-resume) |
| Stage 4 — Ensemble Grid Search | CPU | ~3 min | No (restart from Stage 4) |
| Stage 5 — Retrain Winner | CPU | ~10 min | No (restart from Stage 5) |
| Stage 6 — Export Artifacts | CPU | ~10s | No (restart from Stage 6) |
| **Total (full ensemble)** | — | **~2.5 hours** | — |

Runtime variables:
- **GPU contention** on Kaggle may add 5-15 minutes of queue time at session start
- **Internet speed** for HuggingFace model download (first run only; models are cached to `~/.cache/huggingface/`)
- **Optuna trials** add ~25 minutes; use `--skip-optuna` to skip
- **Dataset size** — the above assumes 342,178 rows; smaller datasets scale proportionally

---

## Checkpoint Locations

| Artifact | Path | Contents |
|---|---|---|
| Best weights | `model/checkpoints/{model}_best.pt` | state_dict with best F1 so far |
| Full checkpoint | `model/checkpoints/{model}_checkpoint.pt` | Full training state (model, optimizer, scheduler, scaler, RNG) |
| Emergency save | `model/checkpoints/{model}_emergency.pt` | Saved on SIGTERM/SIGINT |
| Token cache | `model/checkpoints/token_cache/` | Pre-tokenized datasets in safetensors |
| TF-IDF vectorizer | `model/vectorizer.pkl` | Stage 2 fitted vectorizer |
| XGBoost model | `model/spam_model.pkl` | Final trained classifier |
| Transformer model | `model/transformer_model.pt` | Final state_dict for inference |
| Tokenizer | `model/transformer_tokenizer/` | HuggingFace `save_pretrained` |
| Metadata | `model/model_metadata.json` | Training config, metrics, timestamps |

The `model/checkpoints/` directory is created automatically by the training orchestrator.

---

## Recovery Procedures

### Stage 3 Interrupted: Transformer Training

**Best case — checkpoint exists and auto-resume works:**

```bash
python model/train_model.py --model DeBERTa-v3 --resume
```

The `--resume` flag triggers the resume logic:
1. Load `model/checkpoints/DeBERTa-v3_checkpoint.pt`
2. Restore model, optimizer, scheduler, scaler, and RNG states
3. Resume from `epoch + 1`
4. Skip curriculum learning (all samples from this point)

**If `--resume` is not available in the current code version:**

Manually resume by loading the best checkpoint and continuing:

```python
import torch
from model.train_transformer import train_transformer, TransformerConfig

config = TransformerConfig(
    model_name="microsoft/deberta-v3-base",
    epochs=3,
    checkpoint_dir="model/checkpoints",
)

# This will detect the existing checkpoint and resume
train_transformer(
    train_texts, train_labels, val_texts, val_labels,
    model_name="DeBERTa-v3",
    config=config,
)
```

**Worst case — no checkpoint saved:**

Restart from Stage 1. The training pipeline will overwrite existing checkpoints.

### Stage 2 Interrupted: Classical Training

**No checkpointing for classical training.** You must restart from Stage 1.

To speed up the retry:
```bash
# Skip Optuna HPO to save ~25 minutes
python model/train_model.py --track-a-only --skip-optuna
```

### Stage 4 Interrupted: Ensemble Grid Search

Restart from Stage 4. The ensemble grid search is fast (~3 minutes) — simply rerun the full training command. The orchestrator will skip Stage 3 if the transformer model is already trained (detected via existing `transformer_model.pt`).

### Stage 5 Interrupted: Retrain Winner

Restart from Stage 5. The retrain is fast (~10 minutes) — simply rerun. The orchestrator will skip earlier stages if artifacts exist.

### Kaggle Session Timeout During Transformer Training

Kaggle's interactive session limit is 9 hours. If you're training a large dataset on transformer and approaching the limit:

1. **Save progress**: The checkpoint is saved every epoch (if it's the best F1 epoch). SIGTERM handler also saves an emergency checkpoint.
2. **Download checkpoint**: Before the session ends, download the checkpoint file:
   ```python
   from IPython.display import FileLink
   FileLink("model/checkpoints/DeBERTa-v3_checkpoint.pt")
   ```
3. **New session**: Upload the checkpoint, clone the repo, and:
   ```bash
   python model/train_model.py --model DeBERTa-v3 --resume
   ```

### GPU OOM During Training

If you hit CUDA Out of Memory:

1. The VRAM probing automatically selects the maximum stable batch size
2. If probing fails, manually reduce batch size:
   ```python
   config = TransformerConfig(batch_size=4, gradient_accumulation_steps=16)  # eff batch = 64
   ```
3. Enable `torch.cuda.empty_cache()` between stages
4. Consider using a lighter model (`DistilBERT`) if GPU memory is consistently tight

---

## After Training: Artifact Locations

```
model/
├── spam_model.pkl                    # XGBoost classifier
├── spam_model.pkl.sha256             # SHA-256 integrity hash
├── vectorizer.pkl                    # TF-IDF vectorizer + meta feature config
├── vectorizer.pkl.sha256             # SHA-256 integrity hash
├── model_metadata.json               # Training metrics and config
├── transformer_model.pt              # DeBERTa-v3 state_dict for inference
├── transformer_model.pt.sha256       # SHA-256 integrity hash
├── transformer_tokenizer/            # HuggingFace tokenizer files
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.json
│   └── config.json
└── checkpoints/
    ├── DeBERTa-v3_best.pt            # Best F1 checkpoint
    ├── DeBERTa-v3_checkpoint.pt      # Full training state
    └── token_cache/                  # Pre-tokenized datasets
```

### Downloading Artifacts from Kaggle

```python
import shutil

# Compress model directory
shutil.make_archive("/kaggle/working/model_export", "zip", "model")

# Download
from IPython.display import FileLink
FileLink("/kaggle/working/model_export.zip")
```

---

## Troubleshooting

### HuggingFace Model Download Timeout

On first run, the transformer model must download from HuggingFace Hub (~184 MB for DeBERTa-v3). If the download times out:

```bash
# Option 1: Pre-download the model as a Kaggle dataset
# Upload the model files as a Kaggle dataset and attach it to the notebook

# Option 2: Use a lighter model that downloads faster
python model/train_model.py --model DistilBERT --fast-dev

# Option 3: Set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
python model/train_model.py --model DeBERTa-v3
```

### "No module named 'slowapi'" or Import Errors

```bash
pip install -r requirements.txt --quiet
```

### Dataset Not Found

The pipeline looks for the CSV in this order:
1. `--csv-path` CLI argument
2. `KAGGLE_INPUT_DIR` env var
3. `/kaggle/input/`
4. `data/spam.csv`

If the dataset is attached to the notebook but not auto-discovered:
```bash
python model/train_model.py --csv-path /kaggle/input/your-dataset-name/spam.csv
```

### Training Runs But No GPU Usage

Check that CUDA is properly initialized:
```python
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"Device: {torch.cuda.get_device_name(0)}")
```

If CUDA appears unavailable, enable GPU in Notebook Settings → Accelerator → GPU T4 x2 and restart the session.

### Python Process Killed (OOM)

The system RAM (not GPU VRAM) was exhausted. Kaggle provides ~13-16 GB. Solutions:
- Run `--track-a-only` and `--track-b-only` in separate sessions
- Use `--fast-dev` to verify on 500 rows first
- Reduce dataset size if using a custom CSV
