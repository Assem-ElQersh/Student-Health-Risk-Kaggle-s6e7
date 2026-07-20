# N18v2: SOTA Family Expansion (TabICLv2, disk-offload fix)

## Why v2 exists (N18v1 postmortem)
N18v1 ran to completion but produced **no improvement** — Public LB ~0.95027, below N17's 0.95034
— because both new families failed and the fusion fell back to N17's exact blend:

- **TabICL OOM'd and scored 0.** Error: `CPU memory allocation failed (CUDA out of memory) and
  disk offload is not available. Please specify disk_offload_dir ... (estimated: 18995MB)`. With
  `offload_mode="auto"` and no `disk_offload_dir`, the ~19GB column-wise embedding tensor for the
  full 552K context had nowhere to spill. The OOM was also raised during predict, outside the
  fit-only fallback, so the whole family was lost.
- **TabM collapsed to 0.84240.** It was fed the raw target-encoded matrix with no feature scaling;
  MLPs can't learn on unscaled TE columns.

The guardrail floor did its job (fell back to N17, no downside), but nothing new entered the blend.

## What changed in v2
- **TabM dropped entirely.** Even correctly scaled, it would very likely land below GBDT (0.94998)
  and add no blend value on this near-deterministic noised target (5th straight MLP-family
  underperformance in this project). Removed from pip install, imports, family cell, and fusion.
- **TabICL repaired** and kept as the one genuinely new full-context information source:
  - `disk_offload_dir="/kaggle/working/tabicl_offload"` set so the big embedding tensor spills to
    disk under `offload_mode="auto"`.
  - the per-fold `try/except` now wraps **both fit and predict**, so any OOM triggers a
    **150K-context fallback** (fits in memory, ~5GB, no disk needed) instead of losing the family.
  - query batch reduced to `bs=8192`.

## Kaggle setup
1. Attach competition data + model **`prior-labsai/tabpfn-3`**.
2. Accelerator: **GPU T4 x2**.
3. **Internet: ON** (required for `pip install tabicl` and its checkpoint download). If Internet is
   off, TabICL is skipped and the notebook falls back to the N17 blend (GBDT + TabPFN) — a valid
   but unimproved run.
4. Factory reset, Run All.
5. Expect near the top: `TabPFN checkpoint OK: ...` and `TabICL import OK`.

## What each family should print
- **Family A (GBDT)**: ensemble OOF ~0.94998 + three solo OOFs + dilution verdict (REJECTED again).
- **Family B (TabPFN)**: ~0.94872 again (deterministic given seeds).
- **Family D (TabICL)**: per fold prints `BAcc=... (context=N)`. `context` should read ~442K
  (80% of a 552K fold-train set) if the full-context fit succeeds. If it prints
  `full-context failed (...); retry 150K context...` then `context=150000`, the fallback engaged —
  still a valid signal, just a smaller context. If the whole family errors, it prints
  `TabICL FAILED completely` and is excluded (result then == N17).
- **Fusion**: prints the N17 guardrail floor OOF, then the N-way blend OOF, then ACCEPTED/REJECTED.

## Expected runtime
GBDT (~1.1h) + TabPFN (~25min) + TabICL (full-context fit + disk-offloaded predict can be slow;
budget ~1-2h, more if disk offload is heavily used) + fusion/thresholds (seconds) + PL retrain
(~15min). Total roughly **2.5-3.5h**, within Kaggle's 12h.

## Watch for
- If TabICL disk offload fills `/kaggle/working` (quota ~19-20GB), the full-context fit may still
  fail; the 150K fallback avoids disk entirely (~5GB in memory) and should always succeed.
- TabICL predict over a huge context is the slowest step; if a fold hangs far beyond the others,
  the fallback path (smaller context) is the intended safety valve.

## Honest expectation
If TabICL lands OOF >= 0.949, the N-way blend + thresholds has a real shot at beating N17's 0.95038
OOF. If it fails again or scores low, the result matches N17 (0.95034 LB) — no downside — and that
is strong evidence the ~0.9503-0.9504 plateau is the true organic ceiling, at which point the
sensible move before the July 31 deadline is to pivot to Private-LB robustness (seed-blending,
variance reduction) rather than chase further public-LB gains.

## Reupload policy
Do **not** re-upload N13, N14v3, N14v4, N15v4, N16 (retired), N17, or N18v1 (all scored/superseded).
N18v2 is the only open notebook.
