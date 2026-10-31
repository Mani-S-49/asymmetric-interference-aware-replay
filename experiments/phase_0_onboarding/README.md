# Phase 0 — Onboarding

Research question: does sequential Hindi sentiment training reduce English IMDb sentiment performance?

1. Fine-tune on a deterministic 2,000-example IMDb train subset.
2. Evaluate on 500 IMDb test examples.
3. Continue fine-tuning the same model on the original Hindi Indic Sentiment training split.
4. Evaluate Hindi, then reevaluate English.
5. Calculate English accuracy before minus English accuracy after Hindi and save a forgetting curve.

The reference script swaps the Hindi splits. This implementation preserves the published train/test assignment. Status: not run; checkpoint pending real Kaggle evidence.
