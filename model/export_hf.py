"""Export trained DeBERTa-v3 model to Hugging Face native format.

Reads transformer_model.pt (state_dict) + transformer_tokenizer/
Produces hf_model/ directory with:
  - model.safetensors  (full model weights)
  - config.json
  - tokenizer.json
  - tokenizer_config.json
  - special_tokens_map.json
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def export_hf_model(model_dir: Path, output_dir: Path) -> None:
    pt_path = model_dir / "transformer_model.pt"
    tokenizer_src = model_dir / "transformer_tokenizer"

    if not pt_path.exists():
        print(f"ERROR: {pt_path} not found. Run training first.")
        sys.exit(1)
    if not tokenizer_src.is_dir():
        print(f"ERROR: {tokenizer_src} not found.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load base config ────────────────────────────────────
    print(f"Loading config for microsoft/deberta-v3-base...")
    config = AutoConfig.from_pretrained("microsoft/deberta-v3-base", num_labels=2)
    config.id2label = {0: "HAM", 1: "SPAM"}
    config.label2id = {"HAM": 0, "SPAM": 1}

    # ── Step 2: Create model & load fine-tuned state_dict ────────────
    print(f"Creating base model structure...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", config=config
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    print(f"Loading fine-tuned state_dict from {pt_path}...")
    state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print(f"  All keys matched perfectly.")
    model.eval()

    # ── Step 3: Save to HF native format ─────────────────────────────
    print(f"Saving full model to {output_dir}...")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    print(f"  [OK] model.safetensors")
    print(f"  [OK] config.json")

    # ── Step 4: Copy tokenizer files ─────────────────────────────────
    shutil.copy(tokenizer_src / "tokenizer.json", output_dir / "tokenizer.json")
    shutil.copy(tokenizer_src / "tokenizer_config.json", output_dir / "tokenizer_config.json")

    # Fix model_max_length (currently a sentinel value)
    with open(output_dir / "tokenizer_config.json", "r") as f:
        tok_config = json.load(f)
    tok_config["model_max_length"] = 512
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tok_config, f, indent=2)

    # ── Step 5: Load & re-save tokenizer (generates special_tokens_map) ──
    print("Loading tokenizer from export dir...")
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    tokenizer.model_max_length = 512
    tokenizer.save_pretrained(str(output_dir))
    print(f"  [OK] tokenizer.json")
    print(f"  [OK] tokenizer_config.json")
    print(f"  [OK] special_tokens_map.json")

    # ── Step 6: Verify ───────────────────────────────────────────────
    print("\nVerifying exported model...")
    loaded_model = AutoModelForSequenceClassification.from_pretrained(str(output_dir))
    loaded_tokenizer = AutoTokenizer.from_pretrained(str(output_dir))

    assert loaded_model.config.num_labels == 2, "num_labels != 2"
    assert loaded_model.config.id2label == {0: "HAM", 1: "SPAM"}, "id2label mismatch"
    assert loaded_tokenizer.pad_token == "[PAD]", "pad_token mismatch"
    assert loaded_tokenizer.model_max_length == 512, "model_max_length != 512"

    # Quick inference test
    test_text = "URGENT: You won 1 million dollars! Claim now."
    inputs = loaded_tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    print(f"  Test: \"{test_text}\"")
    print(f"    HAM={probs[0]:.4f}  SPAM={probs[1]:.4f}")

    # List files with sizes
    total_size = 0
    print(f"\nExported files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name:40s} {size_mb:8.1f} MB")
    print(f"  {'-' * 40}")
    print(f"  Total: {total_size:.1f} MB")

    print("\n[SUCCESS] Export complete!")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    export_hf_model(base, base / "hf_model")
