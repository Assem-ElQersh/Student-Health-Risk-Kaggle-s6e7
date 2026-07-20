# N17: Clean Ceiling Blend

## Why this notebook exists
N15v4 ran to completion and scored **0.95027 Public LB** (OOF 0.95019, L2 Meta). Its TabPFN family
collapsed to **0.93302 OOF** because it combined `balance_probabilities=True` (already a native
prior correction) with an additional manual `proba /= class_priors` division. This is the exact
"Double Correction" trap logged at N07 — confirmed catastrophic here: TabPFN regressed by **-0.0157**
versus N14v3's raw **0.94872**, and the fusion search correctly zeroed its blend weight.

N14v3's original recipe (GBDT Numeric-TE + `balance_probabilities`-only TabPFN, fixed alpha=0.28)
remains the organic best: **OOF 0.95031, Public LB 0.95029**.

N17 restores that clean recipe and adds two cheap, previously untested levers, while folding in
N16's still-open question (retired notebook, logic absorbed here at zero extra compute):

1. **Searched blend weight** — grid scan alpha over [0.0, 0.6] instead of the fixed 0.28.
2. **OOF-learned per-class decision thresholds** — Nelder-Mead 3-vector multiplier on the blended
   probabilities, guardrailed to fall back to identity if it doesn't improve OOF (N07 got +0.00006
   on GBDT-solo; untested on the Era-2 blend).
3. **Folded-in dilution test (from N16)** — logs solo CatBoost/LightGBM/HGBC OOF (seed=42 only,
   captured for free inside the existing 3-seed GBDT loop) next to the 3x3 seed-ensemble OOF, to
   settle whether ensembling across seeds dilutes the Numeric-TE signal vs Era-1 `v0.7`'s 0.95020.

RealMLP is **dropped** (5.8h/5-fold in N15v4, scored 0.94797 — below GBDT's 0.94996, not worth the
compute for its marginal diversity).

## Kaggle setup
1. Attach competition data + model **`prior-labsai/tabpfn-3`**.
2. Accelerator: **GPU T4 x2**.
3. Factory reset, Run All.
4. Expect early print: `TabPFN checkpoint OK: .../tabpfn-v3-classifier-v3_default.ckpt`.
5. Expect the GBDT cell to also print three solo OOF lines (CatBoost-solo, LightGBM-solo, HGBC-solo)
   and a `Dilution check:` verdict line — this is the folded-in N16 test, not a bug.
6. TabPFN fold prints will show a single `BAcc=` value per fold (no more `raw=` / dual-number
   printout — the double-correction removal means there's only one probability vector now).

## Expected runtime
Roughly the same as N14v3 (~1.5-2h): GBDT 3x3 ensemble (~1h), TabPFN compact 100K/fold (~20-25min),
fusion + threshold search (seconds), pseudo-label retrain (~15min). No RealMLP, so much cheaper
than N15v4's ~7.5h.

## What to record
- GBDT 3x3 ensemble OOF, and each solo (CatBoost/LightGBM/HGBC) OOF + dilution verdict.
- TabPFN OOF (should land near 0.94872, confirming the double-correction fix).
- Searched blend alpha and its OOF (compare to N14v3's fixed alpha=0.28 / OOF 0.95031).
- Learned per-class thresholds, whether accepted or rejected, and resulting OOF.
- Final selected method + OOF, Public LB after submit.
- Append findings to README Provenance Ledger and `results/output.txt`.

## Honest expectation
The organic public ceiling sits at ~0.9503. N17's two new levers are each individually small
(historically +0.00000 to +0.0002 range for this kind of search on an already-tuned blend).
Realistic upside over N14v3's 0.95029 is **+0.0000 to +0.0004**. If N17 does not beat 0.95029,
N14v3 stands as the confirmed organic ceiling and the next mission should pivot to Private LB
robustness (variance reduction, seed-blending) rather than further public-score chasing.

## Reupload policy
Do **not** re-upload N13, N14v3, N14v4, N15v4, or N16 (retired/deleted). N17 is the only open
notebook.
