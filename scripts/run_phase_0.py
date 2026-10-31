
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase_0.config import load_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/phase_0.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--skip-dataset-download",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)

    if args.skip_dataset_download:
        print(
            json.dumps(
                {
                    "config_valid": True,
                    "phase": cfg["phase"]["id"],
                    "model": cfg["model"]["name"],
                },
                indent=2,
            )
        )
        return

    from src.phase_0.data import load_english, load_hindi
    from src.phase_0.experiment import run_experiment

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out = ROOT / cfg["phase"]["output_root"] / run_id

    print("Loading IMDb subset...")
    en_train, en_test = load_english(cfg)

    print(f"English train={len(en_train)} test={len(en_test)}")

    # Display the actual Hindi split policy configured for this experiment.
    hindi_cfg = cfg["datasets"]["hindi"]

    if hindi_cfg.get("swap_train_test", False):
        print(
            "Loading Hindi sentiment: "
            "training uses the original test split; "
            "evaluation uses the original train split..."
        )
    else:
        print(
            "Loading Hindi sentiment using the original "
            "train and test splits..."
        )

    hi_train, hi_test = load_hindi(cfg)

    print(f"Hindi train={len(hi_train)} test={len(hi_test)}")

    print("Output:", out)

    result = run_experiment(
        cfg,
        en_train,
        en_test,
        hi_train,
        hi_test,
        out,
        dry_run=args.dry_run,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
