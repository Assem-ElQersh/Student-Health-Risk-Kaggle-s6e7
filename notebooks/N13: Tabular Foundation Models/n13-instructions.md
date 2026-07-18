# N13: Tabular Foundation Models (TabPFN Meta-Ensemble)

## Architecture Overview
Per Rule 17 (The Novelty Directive), the Epistemological Engine shifted from classical GBDTs to Tabular Foundation Models.

This directory contains `n13-tabpfn-ensemble.ipynb`.

### Historical Constraint (why this notebook bags)
Older TabPFN releases had a hard ~10k-row context limit. Our train set is 690,088 rows, so a single forward pass was not viable. N13 therefore implements a **Massive Bagged Meta-Ensemble**:

1. Categorical target-encoding + numeric median imputation, then `QuantileTransformer(output_distribution='normal')`.
2. 5-fold stratified CV.
3. Per fold: 10 non-overlapping bags of 5,000 rows.
4. Each bag fits a fresh `TabPFNClassifier` on that slice only.
5. Bag probabilities are averaged; fold test probabilities are averaged across folds.
6. Metric: balanced accuracy (competition metric).

### Ledger status (read before investing GPU hours)
README Provenance Ledger marks N13 as **obsolete for ceiling-breaking**: TabPFN-3 supports up to ~1M rows natively. Prefer **N14** / **N15** for full-data TabPFN-3 runs. Keep N13 only if you need an empirical bagged baseline vs full-context TabPFN.

---

## Mandatory Kaggle model mount (TabPFNLicenseError)

TabPFN-3 weights are gated. On Kaggle there is no interactive PriorLabs browser login. If the package cannot find a local checkpoint, `.fit()` raises `TabPFNLicenseError`.

### Required notebook input
1. Competition Models tab → **`prior-labsai/tabpfn-3`** (default variation).
2. Request access / accept Kaggle model license if prompted.
3. **Add the model as an input** to this notebook (not only to a different kernel).

Verified mount layout:

```text
/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1/tabpfn-v3-classifier-v3_default.ckpt
/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt
/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1/tabpfn-v3-classifier-v3_20260417_binary.ckpt
/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1/config.json
/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1/LICENSE
```

### How the notebook loads weights
Env-only `TABPFN_MODEL_CACHE_DIR` is **not sufficient** if `tabpfn` was imported before the env was set (settings are read at import time). N13 does both:

1. Sets `TABPFN_MODEL_CACHE_DIR` and `TABPFN_NO_BROWSER` **before** `import tabpfn`.
2. Passes an absolute checkpoint path into every constructor:

```python
TABPFN_CKPT = "/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1/tabpfn-v3-classifier-v3_default.ckpt"

clf = TabPFNClassifier(
    device=DEVICE,
    n_estimators=1,
    random_state=SEED + bag,
    model_path=TABPFN_CKPT,
)
```

Use the **default classifier** checkpoint for this competition (3 classes, ~690k rows). Do **not** use the `binary` or `<200k multiclass` specialized checkpoints unless running a deliberate ablation.

---

## API notes (current `tabpfn` package)

| Old (breaks) | Current |
|---|---|
| `N_ensemble_configurations=1` | `n_estimators=1` |
| Implicit HF download on Kaggle | Explicit `model_path=TABPFN_CKPT` + mounted model input |

`n_estimators=1` is intentional: N13 macro-ensembles across bags, so TabPFN’s internal ensemble is disabled.

---

## Handoff / execution checklist

1. Attach **`prior-labsai/tabpfn-3`** as a notebook input.
2. Upload / sync the **current** local `n13-tabpfn-ensemble.ipynb` (must contain `model_path=TABPFN_CKPT`).
3. **Factory reset** the Kaggle runtime (stale imports keep the license failure alive).
4. **Run All** from the top. Confirm early print:  
   `TabPFN checkpoint OK: .../tabpfn-v3-classifier-v3_default.ckpt`
5. Accelerator: GPU (T4 or better). Expect long runtime (many bag × fold inference passes over full val/test).
6. Record `Final OOF BAcc` and compare to organic ceiling **0.95011**.
7. Append the result to `README.md` Provenance Ledger and save raw output under `results/`.

### If `TabPFNLicenseError` still appears
- You are running an old cell without `model_path=TABPFN_CKPT`, **or**
- The model input is not attached to **this** notebook, **or**
- You re-ran only the training loop without Factory reset + Run All.

Quick probe (optional):

```python
import os
p = "/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1/tabpfn-v3-classifier-v3_default.ckpt"
print(os.path.isfile(p), p)
```

Fallback (only if mount is impossible): accept license at https://ux.priorlabs.ai, put API key in Kaggle Secret `TABPFN_TOKEN`. Prefer the mounted checkpoint path above.

---

## Success criterion
If bagged TabPFN organically exceeds **0.95011** OOF / Public LB without anchor CSVs, log it as a new organic baseline. If it fails, treat that as evidence that bagged short-context TabPFN does not unlock additional geometry beyond the established GBDT/RealMLP ceiling — then move execution budget to **N14 / N15** (native TabPFN-3 full-data path).
