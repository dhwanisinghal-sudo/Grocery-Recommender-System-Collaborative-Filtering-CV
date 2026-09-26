# experiments/ — verification scripts for the deployed system

This folder contains reproducible scripts that verify the metrics reported
for the **current, deployed** hybrid CF system (`app.py`). Run any of them
directly:

```bash
python experiments/verify_ablation.py
python experiments/verify_k_sensitivity.py
python experiments/verify_worked_example.py
```

See **`VERIFICATION_README.md`** in this folder for what each script
reproduces and how it maps to the paper's sections/tables.

All three import shared hyperparameters from `config.py` at the repo
root — the same file `app.py` imports — so these numbers cannot drift out
of sync with the live app.

## Note on the old Instacart/MobileNetV2 notebook

This folder used to also document an early, unrelated exploration phase
(Instacart dataset + MobileNetV2 + `surprise`). That material has moved to
`notebooks/archived_phase1_instacart_mobilenet/` and is unrelated to the
scripts in this folder or to the deployed app. See `notebooks/README.md`
for details — it is kept only for historical reference and does not
describe the current system.
