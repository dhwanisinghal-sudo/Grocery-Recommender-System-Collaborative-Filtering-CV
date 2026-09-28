# experiments/

These scripts reproduce the metrics I report for the deployed system
(`app.py`). Run any of them from the repo root:

```bash
python experiments/verify_ablation.py
python experiments/verify_k_sensitivity.py
python experiments/verify_worked_example.py
python experiments/verify_grid_search.py
python experiments/verify_vision_accuracy.py
```

`VERIFICATION_README.md` in this folder says what each script reproduces and
which table or section of the paper it matches.

`verify_ablation.py`, `verify_k_sensitivity.py` and `verify_worked_example.py`
import their hyperparameters from `config.py` in the repo root. `app.py`,
`verify_grid_search.py` and `verify_vision_accuracy.py` don't. They use their
own copies of the same values, which I keep in sync by hand (the `config.py`
docstring explains this).

`vision_accuracy_colab.ipynb` is the Colab notebook I used for the second
vision run, the one with real Gemini and Hugging Face keys.

My first attempt (Instacart data, MobileNetV2, `surprise`) is not in this
folder. It is archived in `notebooks/` and has nothing to do with these
scripts or with the deployed app. See `notebooks/README.md`.
