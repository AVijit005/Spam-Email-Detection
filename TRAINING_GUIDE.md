# TRAINING GUIDE — Spam Detection Pipeline v3.0

## Kaggle Commands

### Full ensemble training (XGBoost + DeBERTa-v3)
```bash
python model/train_model.py --model DeBERTa-v3
```

### Classical only (no GPU needed)
```bash
python model/train_model.py --track-a-only --competition
```

### Transformer only
```bash
python model/train_model.py --track-b-only --model DeBERTa-v3
```

### Fast dev run (500 rows, ~5 minutes)
```bash
python model/train_model.py --fast-dev --model DeBERTa-v3
```

### Custom CSV path
```bash
python model/train_model.py --csv-path /kaggle/input/spam-dataset/spam.csv
```

### Custom output directory
```bash
python model/train_model.py --output-dir /kaggle/working/models
```

## GPU Requirements

| Component | GPU | VRAM | Required |
|---|---|---|---|
| DeBERTa-v3-base (fp16) | T4 / A10G / A100 | 8 GB | Yes for Track B |
| DeBERTa-v3-base (fp32) | A10G / A100 | 16 GB | Without fp16 |
| RoBERTa-base (fp16) | T4 | 8 GB | Yes for Track B |
| ELECTRA-base (fp16) | T4 | 8 GB | Yes for Track B |
| Classical only (Track A) | None | 0 | No GPU needed |

### Kaggle T4 configuration
- VRAM: 16 GB (sufficient for all transformer models in fp16)
- Enable GPU in Kaggle notebook: Settings → Accelerator → GPU T4 x2

## RAM Requirements

| Stage | RAM | Notes |
|---|---|---|
| Stage 1 (Load) | ~2 GB | 342k rows in pandas DataFrame |
| Stage 2 (Classical) | ~8 GB | TF-IDF sparse matrix + XGBoost candidates |
| Stage 3 (Transformer) | ~12 GB | Model weights + DataLoader + activations |
| Stage 4 (Ensemble) | ~6 GB | Sparse features + probability arrays |
| Stage 5 (Retrain) | ~8 GB | Full dataset TF-IDF + XGBoost fit |
| **Peak combined** | **~16 GB** | Stages 3 + 4 overlap |

## Expected Runtime (on 342,178 rows)

| Stage | Hardware | Time |
|---|---|---|
| Stage 1 — Load | CPU | ~45s |
| Stage 2 — Classical | 8-core CPU | ~35 min (4 candidates with Optuna) |
| Stage 3 — Transformer | T4 GPU | ~60-90 min (DeBERTa-v3, 3 epochs) |
| Stage 4 — Ensemble | CPU | ~3 min |
| Stage 5 — Retrain | CPU | ~10 min |
| Stage 6 — Export | CPU | ~10s |
| **Total** | — | **~2.5 hours** |

Exact times depend on:
- T4 availability (GPU contention on Kaggle adds queue time)
- Internet speed for HuggingFace model download (first run only; cached after)
- Optuna HPO trials (configurable via `--skip-optuna` to skip, reducing Stage 2 to ~5 min)

## Checkpoint Locations

| Artifact | Path | Format |
|---|---|---|
| Transformer checkpoint | `model/checkpoints/DeBERTa-v3_best.pt` | `torch.save(state_dict)` |
| TF-IDF vectorizer | `model/vectorizer.pkl` | pickle |
| XGBoost model | `model/spam_model.pkl` | pickle |
| Transformer final | `model/transformer_model.pt` | `torch.save(state_dict)` |
| Tokenizer | `model/transformer_tokenizer/` | HuggingFace save_pretrained |
| Metadata | `model/model_metadata.json` | JSON |
| SHA-256 integrity | `model/transformer_model.pt.sha256` | hex hash |

Checkpoint directory is created at `PROJECT_ROOT/model/checkpoints/` and created automatically by the orchestrator.

## Recovery Procedure

### If training interrupted during Stage 3 (Transformer)

1. The checkpoint file `model/checkpoints/DeBERTa-v3_best.pt` contains the best weights up to the last completed epoch.
2. Re-run `python model/train_model.py` — the checkpoint is not currently loaded automatically (resume requires code change to check for existing checkpoint).
3. To manually resume: load the checkpoint and continue from the last epoch.

### If training interrupted during Stage 2 (Classical)

1. No checkpointing for classical training — restart is necessary.
2. Use `--fast-dev` to verify the pipeline works before a full run.

### If training interrupted during Stage 5 (Retrain)

1. No checkpointing for retrain — restart is necessary.
2. The retrain is the fastest stage (~10 minutes).

## Artifact Locations After Training

```
model/
├── spam_model.pkl              # XGBoost classifier (ensemble mode)
├── vectorizer.pkl              # TF-IDF vectorizer + meta feature config
├── model_metadata.json         # Training metadata, metrics, timestamps
├── transformer_model.pt        # DeBERTa-v3 state_dict
├── transformer_model.pt.sha256 # SHA-256 integrity hash
├── transformer_tokenizer/      # HuggingFace tokenizer files
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.json / vocab.txt
│   └── config.json
└── checkpoints/
    └── DeBERTa-v3_best.pt      # Best F1 checkpoint during training
```
