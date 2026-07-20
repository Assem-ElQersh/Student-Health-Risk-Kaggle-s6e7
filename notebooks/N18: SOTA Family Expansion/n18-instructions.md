# N18: SOTA Family Expansion (TabICLv2 + TabM)

## Why this notebook exists
N17 ran to completion and scored **Public LB 0.95034** (OOF 0.95038) — the new organic best,
confirming the double-correction fix and validating two new levers (searched blend weight,
OOF-learned per-class thresholds). It also **rejected** the N16 dilution hypothesis: the 3x3
GBDT seed-ensemble (0.94998) beat every solo learner, including HGBC-solo at 0.93491, which
also discredits Era-1's "v0.7 HGBC-solo 0.95020" as a trustworthy number on this feature matrix.

An exhaustive SOTA research sweep (TabArena NeurIPS 2025, TabICL ICML 2025 / v2 Feb 2026,
Chunked TabPFN, multiclass threshold-tuning literature) surfaced two never-tried, orthogonal
model families:

- **TabICLv2** — sees the **full 552K-row fold-train context**, not a 100K subsample. TabPFN's
  compact matrix (`X_tab`) is reused as-is.
- **TabM** — TabArena's #1-ranked model after post-hoc ensembling. Trained on the same
  Numeric-TE matrix (`X_full`) used by GBDT, via a custom k=32 parameter-efficient MLP
  ensemble (pure PyTorch, per Rule 10).

Both are added as **guarded** families: if either fails or underperforms, the N-way blend
search assigns it near-zero weight and the notebook falls back to N17's proven blend as a
guardrail floor. There is no downside beyond compute.

## Kaggle setup
1. Attach competition data + model **`prior-labsai/tabpfn-3`**.
2. Accelerator: **GPU T4 x2**.
3. **Internet: ON.** This is new vs N17 — `pip install tabicl tabm` and their checkpoint
   downloads require network access. If Internet is off, TabICL/TabM will fail their
   import check and the notebook will print `TabICL skipped` / `TabM skipped` and gracefully
   fall back to the N17 blend (GBDT + TabPFN only) — still a valid, if unchanged, run.
4. Factory reset, Run All.
5. Expect these prints near the top: `TabPFN checkpoint OK: ...`, `TabICL import OK`,
   `TabM import OK`. If any says "unavailable", check Internet is ON and re-run the pip
   install cell.

## What each family should print
- **Family A (GBDT)**: same as N17 — ensemble OOF, three solo OOFs, dilution verdict.
- **Family B (TabPFN)**: should land at **~0.94872** again (exact match expected).
- **Family D (TabICL)**: prints `context=<N>` per fold — should read ~441K-442K (80% of a
  552K fold-train set) if the full-context fit succeeds. If it falls back to 200K, that means
  the full-context fit hit an OOM/error; still a valid signal, just smaller than intended.
- **Family E (TabM)**: prints a `BAcc=` per fold; training may take several epochs each with
  early stopping (patience=8) — this is the slowest new family, budget ~45-90 min.
- **Fusion**: prints the N17 guardrail floor OOF first, then the N-way blend OOF, then whether
  the N-way blend was ACCEPTED or REJECTED relative to the floor.

## Expected runtime
GBDT (~1h) + TabPFN (~25min) + TabICL (~30-60min, full-context fit is slower than TabPFN's
100K subsample but should still beat TabPFN's per-row cost per the paper's 10x speed claim) +
TabM (~45-90min) + fusion/thresholds (seconds) + pseudo-label retrain (~15min) = roughly
**3-3.5h total**, well within Kaggle's 12h GPU budget.

## What to record
- Each family's OOF (GBDT ensemble + 3 solos, TabPFN, TabICL, TabM).
- N-way blend weights and whether it was accepted over the N17 guardrail floor.
- Learned per-class thresholds and whether accepted.
- Final selected method + OOF, Public LB after submit.
- Append findings to README Provenance Ledger and `results/output.txt`.

## Honest expectation
If TabICL and/or TabM land OOF >= 0.949, the N-way blend has a realistic shot at pushing past
N17's 0.95038 OOF toward 0.9505+. If both underperform, the guardrail floor holds and the
result matches N17 (0.95038 OOF / 0.95034 LB) — confirming the 0.9503-0.9504 plateau is likely
the true organic ceiling and justifying a pivot to Private LB robustness for the final week
before the July 31 deadline.

## Reupload policy
Do **not** re-upload N13, N14v3, N14v4, N15v4, N16 (retired), or N17 (all already scored).
N18 is the only open notebook.
