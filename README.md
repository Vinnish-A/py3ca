# py3ca

Python reimplementation of the 3CA NMF pipeline (Gavish et al., 2023).

## What it does

- Per-sample NMF (K=4..9) after 3CA preprocessing.
- Robust program filtering (within-sample, across-sample, non-redundant).
- Greedy clustering into Meta-Programs (MPs).
- MP scoring per cell with one-sample t-test and FDR correction.

## Quick start

1) Install dependencies:

- See requirements.txt

2) Run:

- `python run.py --data-dir /mnt/sdb/xzh/Vproject/TCA/data --out-dir /mnt/sdb/xzh/Vproject/TCA/py3ca/out`

Outputs are written under the output directory.
