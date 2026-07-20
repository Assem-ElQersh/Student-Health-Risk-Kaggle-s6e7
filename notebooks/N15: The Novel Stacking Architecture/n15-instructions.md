# N15v4 — Executed

## Status
**Already run and submitted.** N15v4 scored **Public LB 0.95027** (OOF 0.95019, L2 Meta blend of
RealMLP + TabPFN + GBDT). Do not re-upload.

## Result summary
- GBDT NumericTE ensemble OOF: 0.94996
- RealMLP prior-corr OOF: 0.94797 (~5.8h/5-fold — expensive, below GBDT)
- TabPFN prior-corr OOF: 0.93302 — **collapsed by the Double-Correction bug** (see below)
- L2 Meta selected: 0.95019 OOF -> 0.95027 LB

## Double-Correction finding
TabPFN combined `balance_probabilities=True` (native correction) with a second manual
`proba /= class_priors` division. This is the trap logged at N07: it over-corrects toward
minority classes and destroyed TabPFN's OOF (0.94872 raw in N14v3 -> 0.93302 here). See
Rules.md Rule 21 and `results/output.txt` for the full diagnosis.

## Superseded by
`notebooks/N17: Clean Ceiling Blend/` restores the clean (single-correction) recipe, drops
RealMLP, folds in N16's dilution test, and adds a searched blend weight + OOF-learned
per-class thresholds. N16 (`Solo TE Ablation and Safe TabPFN`) has been retired — its cheap
value was absorbed into N17's existing GBDT loop.
