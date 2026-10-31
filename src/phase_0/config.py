from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config

def validate_config(config: dict[str, Any]) -> None:
    required = {"phase", "model", "datasets", "lora", "training", "evaluation", "reproducibility"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    if int(config["lora"]["rank"]) < 1 or int(config["lora"]["alpha"]) < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if not config["lora"]["target_modules"]:
        raise ValueError("LoRA target_modules cannot be empty")
    for lang in ("english", "hindi"):
        ds = config["datasets"][lang]
        for key in ("name", "text_column", "label_column", "train_split", "test_split"):
            if key not in ds:
                raise ValueError(f"Missing datasets.{lang}.{key}")
    if int(config["model"]["max_length"]) < 1:
        raise ValueError("max_length must be positive")
