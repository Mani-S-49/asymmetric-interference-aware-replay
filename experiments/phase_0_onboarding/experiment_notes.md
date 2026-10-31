# Phase 0 experiment notes

- Status: not run
- Research question: does English performance drop after Hindi training?
- Stream: English IMDb → Hindi Indic Sentiment
- IMDb: 2,000 train / 500 test, seed 42
- Hindi: original published splits (verify counts at runtime)
- Model: meta-llama/Llama-3.2-1B
- LoRA: r=16, alpha=32, q/k/v/o projections, dropout 0.05
- Training parameters: configs/phase_0.yaml; pilot choices, not all mandated by research plan
- Seed: 42
- Results/artifacts: pending
- Limitations: small English subset, small Hindi train split, one order and one seed
- Decision: pending actual run and checkpoint review
