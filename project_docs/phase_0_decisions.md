# Phase 0 decisions

1. Use the research-plan model and LoRA settings.
2. IMDb source: stanfordnlp/imdb; 2,000 train and 500 test selected with seed 42.
3. Hindi source: mteb/indic_sentiment, config hi.
4. Preserve original Hindi train/test assignments; do not swap by default.
5. Normalize sentiment labels to 0=Negative, 1=Positive and validate.
6. Training settings are pilot choices and must be logged.
7. Train in Kaggle; local environment is for safe tests/configuration.
8. Do not claim checkpoint completion without actual results and artifact review.
