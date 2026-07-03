"""Verify HF export produces identical predictions to the old state_dict loading."""
import torch
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path


def load_old():
    config = AutoConfig.from_pretrained("microsoft/deberta-v3-base", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", config=config
    )
    sd = torch.load("model/transformer_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("model/transformer_tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_new():
    model = AutoModelForSequenceClassification.from_pretrained(
        "model/hf_model", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "model/hf_model", local_files_only=True
    )
    return model, tokenizer


def compare():
    print("Loading old model (base + state_dict)...")
    model_old, tok_old = load_old()
    print("Loading new model (hf_model directory)...")
    model_new, tok_new = load_new()

    texts = [
        "URGENT: You won 1 million dollars! Claim now.",
        "Hi John, can we reschedule our meeting to 3pm?",
        "FREE VIAGRA CHEAP CLICK HERE NOW!!!",
        "Please find attached the Q3 report for your review.",
        "You have been selected for a free iPhone! Click the link below.",
        "Dear customer, your account has been compromised. Verify now.",
        "Meeting agenda for Monday's sprint review.",
        "CONGRATULATIONS!!! YOU ARE THE LUCKY WINNER!!!",
        "Can you send me the latest TPS report?",
        "Limited time offer! 90% OFF! Buy now before it's too late!",
    ]

    print(f"\nComparing predictions on {len(texts)} texts:")
    print(f"{'Text':<55s} {'Old SPAM%':>11s} {'New SPAM%':>11s} {'Delta':>10s} {'Status':>8s}")
    print("-" * 95)

    max_delta = 0.0
    for text in texts:
        i1 = tok_old(text, return_tensors="pt", truncation=True, max_length=512)
        i2 = tok_new(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            p1 = torch.softmax(model_old(**i1).logits, dim=-1).cpu().numpy()
            p2 = torch.softmax(model_new(**i2).logits, dim=-1).cpu().numpy()
        spam1 = p1[0][1]
        spam2 = p2[0][1]
        delta = abs(spam1 - spam2)
        max_delta = max(max_delta, delta)
        status = "OK" if delta < 0.01 else "MISMATCH"
        print(f"{text[:52]:<55s} {spam1:10.6f} {spam2:10.6f} {delta:9.6f} {status:>8s}")

    print("-" * 95)
    print(f"Max delta: {max_delta:.6f}")

    # Also check that tokenized outputs match
    print("\nTokenization check:")
    t1 = tok_old(texts[0])["input_ids"]
    t2 = tok_new(texts[0])["input_ids"]
    print(f"  Old token ids (first 10): {t1[:10]}")
    print(f"  New token ids (first 10): {t2[:10]}")
    print(f"  Same: {t1 == t2}")

    # Check model dtype
    print(f"\nModel dtypes:")
    print(f"  Old model dtype: {next(model_old.parameters()).dtype}")
    print(f"  New model dtype: {next(model_new.parameters()).dtype}")

    if max_delta < 0.01:
        print("\n[PASS] Predictions match within tolerance.")
    else:
        print("\n[FAIL] Predictions diverge significantly.")


if __name__ == "__main__":
    compare()
