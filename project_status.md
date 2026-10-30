# Project Status

## Research plan

- Authoritative plan: `..\1 . Research_Plan\research_plan.md`
- Research title: Asymmetric Interference-Aware Replay for Language-Incremental Continual Learning

## Workspace setup

- [x] Create separate implementation folder
- [x] Initialize Git repository in the implementation folder
- [x] Configure repository-specific Git name and email
- [x] Verify the authoritative research-plan file
- [ ] Review initial documentation and Git ignore rules
- [ ] Create initial Git commit
- [ ] Create and connect the GitHub repository

## Research phases

| Phase | Status | Required checkpoint | Evidence |
|---|---|---|---|
| Phase 0 — Onboarding | Not started | Produce a two-task forgetting curve for English IMDB followed by Hindi sentiment data | Pending |
| Phase 1 — Harness construction | Not started | Three-language XNLI sequential fine-tuning reproduces the expected forgetting pattern | Pending |
| Phase 2 — Baselines | Not started | Five-language pilot baseline table with mean and standard deviation over five stream-order permutations | Pending |
| Phase 3 — Prior comparison | Not started | Evaluate candidate priors against observed forgetting and follow the plan's decision rule | Pending |
| Phase 4 — Main comparison | Not started | Evaluate M3 against all required baselines under the planned matched replay budget | Pending |
| Phase 5 — Ablations | Not started | Complete and tabulate all required ablations | Pending |
| Phase 6 — Scaling | Not started | Produce headline results at the planned model scales | Pending |
| Phase 7 — Learned prior | Not started | Evaluate M4 if the stretch phase proceeds | Pending |
| Phase 8 — Writing | Not started | Prepare an evidence-supported paper draft and submission materials | Pending |

## Experiment record

For every experiment, record:

- Experiment ID and phase
- Research question and hypothesis
- Configuration and stream order
- Dataset and split
- Model and training settings
- Random seed
- Software and hardware environment
- Metrics and evaluation results
- Logs, plots, and artifact locations
- Interpretation, limitations, and decision

## Checkpoint policy

A phase is not complete merely because its code runs. The required experiment must be executed, its results checked against the phase checkpoint, and the evidence documented before proceeding.
