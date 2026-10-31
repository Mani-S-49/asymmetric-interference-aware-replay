
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .metrics import compute_metrics, forgetting
from .split_policy import build_hindi_split_provenance


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_hf_token():
    """Retrieve the Hugging Face token without displaying it."""

    token = os.environ.get("HF_TOKEN")

    if token:
        return token

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")

        if token:
            return token

    except Exception:
        pass

    # If no token is found, Transformers can use an existing
    # Hugging Face login or cached authentication.
    return None


def tokenize(ds, tokenizer, max_length):
    return ds.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )


def training_args(output_dir, cfg, epochs, seed):
    t = cfg["training"]

    cuda = torch.cuda.is_available()

    bf16 = bool(
        t.get("bf16", False)
        and cuda
        and torch.cuda.is_bf16_supported()
    )

    fp16 = bool(
        t.get("fp16", False)
        and cuda
        and not bf16
    )

    import inspect

    params = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    eval_key = (
        "eval_strategy"
        if "eval_strategy" in params
        else "evaluation_strategy"
    )

    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=float(t["learning_rate"]),
        per_device_train_batch_size=int(t["train_batch_size"]),
        per_device_eval_batch_size=int(t["eval_batch_size"]),
        gradient_accumulation_steps=int(
            t.get("gradient_accumulation_steps", 1)
        ),
        num_train_epochs=int(epochs),
        weight_decay=float(t.get("weight_decay", 0)),
        logging_steps=int(t.get("logging_steps", 10)),
        save_strategy=t.get("save_strategy", "epoch"),
        gradient_checkpointing=bool(
            t.get("gradient_checkpointing", False)
        ),
        max_grad_norm=float(t.get("max_grad_norm", 1)),
        warmup_steps=0,
        fp16=fp16,
        bf16=bf16,
        report_to=t.get("report_to", "none"),
        dataloader_num_workers=int(
            t.get("dataloader_num_workers", 0)
        ),
        seed=seed,
        data_seed=seed,
        save_total_limit=2,
        remove_unused_columns=True,
    )

    kwargs[eval_key] = t.get("eval_strategy", "epoch")

    return TrainingArguments(**kwargs)


def _json(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def run_experiment(
    cfg,
    en_train,
    en_test,
    hi_train,
    hi_test,
    output_dir,
    dry_run=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["phase"]["seed"])

    seed_everything(
        seed,
        cfg["reproducibility"].get("deterministic", True),
    )

    model_name = cfg["model"]["name"]

    hindi_split_provenance = build_hindi_split_provenance(
        cfg["datasets"]["hindi"]
    )

    # --------------------------------------------------
    # 1. Load tokenizer
    # --------------------------------------------------

    hf_token = get_hf_token()

    tokenizer_kwargs = {"use_fast": True}

    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **tokenizer_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    max_len = int(cfg["model"]["max_length"])

    # --------------------------------------------------
    # 2. Tokenize datasets
    # --------------------------------------------------

    en_train_t = tokenize(en_train, tokenizer, max_len)
    en_test_t = tokenize(en_test, tokenizer, max_len)

    hi_train_t = tokenize(hi_train, tokenizer, max_len)
    hi_test_t = tokenize(hi_test, tokenizer, max_len)

    # --------------------------------------------------
    # 3. Dry-run validation
    # --------------------------------------------------

    if dry_run:
        summary = {
            "dry_run": True,
            "model": model_name,
            "tokenizer_vocab_size": len(tokenizer),
            "hindi_split_provenance": hindi_split_provenance,
            "counts": {
                "english_train": len(en_train),
                "english_test": len(en_test),
                "hindi_train": len(hi_train),
                "hindi_test": len(hi_test),
            },
        }

        _json(
            output_dir / "dry_run_summary.json",
            summary,
        )

        return summary

    # --------------------------------------------------
    # 4. Load model
    # --------------------------------------------------

    t = cfg["training"]

    cuda = torch.cuda.is_available()

    bf16 = bool(
        t.get("bf16", False)
        and cuda
        and torch.cuda.is_bf16_supported()
    )

    fp16 = bool(
        t.get("fp16", False)
        and cuda
        and not bf16
    )

    model_kwargs = {
        "num_labels": int(cfg["model"]["num_labels"]),
    }

    if hf_token:
        model_kwargs["token"] = hf_token

    if fp16:
        model_kwargs["torch_dtype"] = torch.float16
    elif bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **model_kwargs,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # --------------------------------------------------
    # 5. Apply LoRA
    # --------------------------------------------------

    l = cfg["lora"]

    lora_config = LoraConfig(
        r=int(l["rank"]),
        lora_alpha=int(l["alpha"]),
        target_modules=list(l["target_modules"]),
        lora_dropout=float(l["dropout"]),
        bias=l.get("bias", "none"),
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(
        model,
        lora_config,
    )
    # Keep trainable parameters in FP32 for FP16 gradient scaling.
    # The frozen base-model parameters remain in FP16.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.print_trainable_parameters()

    if t.get("gradient_checkpointing", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # --------------------------------------------------
    # 6. Record experiment metadata
    # --------------------------------------------------

    run_meta = {
        "run_id": output_dir.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "model": model_name,
        "seed": seed,
        "lora": l,
        "hindi_split_provenance": hindi_split_provenance,
        "counts": {
            "english_train": len(en_train),
            "english_test": len(en_test),
            "hindi_train": len(hi_train),
            "hindi_test": len(hi_test),
        },
        "config": cfg,
    }

    _json(
        output_dir / "run_metadata.json",
        run_meta,
    )

    # --------------------------------------------------
    # 7. Create ONE Trainer
    # --------------------------------------------------

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args(
            output_dir / "english_training",
            cfg,
            cfg["training"]["epochs_english"],
            seed,
        ),
        train_dataset=en_train_t,
        eval_dataset=en_test_t,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    # --------------------------------------------------
    # 8. Stage 1: English training
    # --------------------------------------------------

    print("\n========== STAGE 1: ENGLISH TRAINING ==========")

    trainer.train()

    en_before = trainer.evaluate(
        eval_dataset=en_test_t,
        metric_key_prefix="english_after_english",
    )

    rows = [
        {
            "stage": "after_english_training",
            "english_accuracy": en_before[
                "english_after_english_accuracy"
            ],
            "english_f1": en_before[
                "english_after_english_f1"
            ],
            "english_loss": en_before[
                "english_after_english_loss"
            ],
            "hindi_accuracy": None,
            "hindi_f1": None,
            "hindi_loss": None,
        }
    ]

    english_checkpoint = output_dir / "model_after_english"

    trainer.save_model(
        str(english_checkpoint)
    )

    tokenizer.save_pretrained(
        english_checkpoint
    )

    # --------------------------------------------------
    # 9. Stage 2: Hindi training
    # --------------------------------------------------

    print("\n========== STAGE 2: HINDI TRAINING ==========")

    # Preserve the same Trainer and model.
    # Only the output directory, epoch count and datasets change.
    #
    # The English-trained model weights, optimizer and scheduler
    # remain in memory.

    trainer.args = training_args(
        output_dir / "hindi_training",
        cfg,
        cfg["training"]["epochs_hindi"],
        seed,
    )

    trainer.train_dataset = hi_train_t
    trainer.eval_dataset = hi_test_t

    trainer.train()

    # --------------------------------------------------
    # 10. Evaluate Hindi and English after Hindi training
    # --------------------------------------------------

    hi_after = trainer.evaluate(
        eval_dataset=hi_test_t,
        metric_key_prefix="hindi_after_hindi",
    )

    en_after = trainer.evaluate(
        eval_dataset=en_test_t,
        metric_key_prefix="english_after_hindi",
    )

    rows.append(
        {
            "stage": "after_hindi_training",
            "english_accuracy": en_after[
                "english_after_hindi_accuracy"
            ],
            "english_f1": en_after[
                "english_after_hindi_f1"
            ],
            "english_loss": en_after[
                "english_after_hindi_loss"
            ],
            "hindi_accuracy": hi_after[
                "hindi_after_hindi_accuracy"
            ],
            "hindi_f1": hi_after[
                "hindi_after_hindi_f1"
            ],
            "hindi_loss": hi_after[
                "hindi_after_hindi_loss"
            ],
        }
    )

    # --------------------------------------------------
    # 11. Calculate forgetting
    # --------------------------------------------------

    summary = {
        "english_accuracy_before_hindi": float(
            rows[0]["english_accuracy"]
        ),
        "english_accuracy_after_hindi": float(
            rows[1]["english_accuracy"]
        ),
        "english_forgetting": forgetting(
            float(rows[0]["english_accuracy"]),
            float(rows[1]["english_accuracy"]),
        ),
        "hindi_accuracy_after_hindi": float(
            rows[1]["hindi_accuracy"]
        ),
        "interpretation": (
            "Positive forgetting means English accuracy "
            "decreased after Hindi training."
        ),
    }

    # --------------------------------------------------
    # 12. Save metrics
    # --------------------------------------------------

    pd.DataFrame(rows).to_csv(
        output_dir / "stage_metrics.csv",
        index=False,
    )

    _json(
        output_dir / "summary_metrics.json",
        summary,
    )

    # --------------------------------------------------
    # 13. Generate forgetting curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        [
            r["stage"].replace("_", " ")
            for r in rows
        ],
        [
            r["english_accuracy"]
            for r in rows
        ],
        marker="o",
    )

    plt.xlabel("Training stage")
    plt.ylabel("English test accuracy")

    plt.title(
        "Phase 0: English performance after sequential training"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_dir / "english_forgetting_curve.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------
    # 14. Save final model
    # --------------------------------------------------

    final_checkpoint = output_dir / "final_model"

    trainer.save_model(
        str(final_checkpoint)
    )

    tokenizer.save_pretrained(
        final_checkpoint
    )

    print("\n========== PHASE 0 COMPLETED ==========")
    print(json.dumps(summary, indent=2))

    return summary
