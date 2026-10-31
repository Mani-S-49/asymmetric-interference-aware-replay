# Phase 0 implementation bundle

This bundle uses the uploaded `final_toy_continual_learning.py` as a reference and separates configuration, data loading, training, metrics, tests, and experiment records.

## Intentional corrections
- Research-plan LoRA settings are used: rank 16, alpha 32, target modules q/k/v/o projections. The reference script's r=2, alpha=4 differs from the plan.
- Original Hindi train/test splits are preserved by default. The reference script swaps them; this bundle does not.
- Hindi labels are normalized to 0=Negative and 1=Positive, with validation.
- Colab-only `!pip install` commands are removed from Python source.
- One explicit training configuration is used, with precision selected based on the available accelerator.
- Training is continued on the same model from English to Hindi, then English is evaluated again.

## Workflow
Develop and run lightweight tests in VS Code. Train in Kaggle, because the local Windows environment previously blocked PyTorch. The gated model `meta-llama/Llama-3.2-1B` requires Hugging Face access approval and a Kaggle secret named `HF_TOKEN`; never hard-code tokens.

In Kaggle, install dependencies with:
`pip install -U transformers datasets peft accelerate scikit-learn pandas matplotlib PyYAML safetensors`

Run:
`python scripts/run_phase_0.py --config configs/phase_0.yaml`

Run a configuration-only check:
`python scripts/run_phase_0.py --skip-dataset-download`

This is Phase 0 only; it does not implement the Phase 1 replay harness. Phase placeholders are included so each phase has its own experiment, results, and report folders. Do not start later phases until the previous checkpoint is reviewed.
