# sbf-labeller

Tiny, very specific tool I use to **manually label data chunks derived from SBF (Septentrio Binary Format) logs**.

It is *not* a general package. It is just a helper I use together with my GNSS jamming generator:

- GNSS generator repo: <https://github.com/macaburguera/GNSS_generator>  
- Jamming classes: `NoJam`, `Chirp`, `NB`, `WB` (same taxonomy as in the generator).

The idea is simple:

1. I preprocess the raw SBF logs (outside this repo) into arrays / features.
2. I launch a small GUI (`label_gui.py`), look at each block, and assign one of those jamming classes.
3. The labels are saved so I can train and validate classifiers later.

> **Entry point:** the script you actually run is **`label_gui.py`**.  
> `feature_extractor.py` is a helper module kept here to support (future) pre-labelling with a pretrained model.

---

## Files

```text
.
├── label_gui.py           # main entry point – GUI to step through blocks and assign labels
├── feature_extractor.py   # helper for computing features for (future) pretrained-model pre-labelling
└── .gitignore
```

### `label_gui.py` (main script)

- This is the script you execute, e.g.:

  ```bash
  python label_gui.py
  ```

- It:
  - Loads the SBF-derived data / blocks I want to label.
  - Shows one block at a time (time series, spectra, or whatever I configured).
  - Lets me press a key / click a button to assign a label:
    - `NoJam`
    - `Chirp`
    - `NB`
    - `WB`
  - Stores the labels (e.g. CSV / NumPy file), which I then use in my ML repos.

In the future, the GUI can also call into `feature_extractor.py` and a pretrained model to **pre-label** samples
and present model suggestions that I can accept / correct.

### `feature_extractor.py` (helper for pre-labelling)

- This file is not meant to be run directly.
- It is kept here as a helper to:

  - compute the same kind of features I use in my classifiers (for example, from the GNSS jamming classifier repo),
  - feed those features into a **pretrained model** that could propose an initial label for each block.

The idea is:

1. `label_gui.py` calls feature-extraction functions from this module.
2. A pretrained model (loaded in `label_gui.py`) uses those features to propose a label.
3. The GUI shows both the raw data and the model’s proposed label, and I decide whether to keep or override it.

Right now this is mostly infrastructure for that future “model-assisted labelling” workflow, so the details may change.

---

## Notes

- This repository is intentionally small and opinionated.
- Many things are hard-coded (paths, class names, assumptions from Jammertest / GNSS_generator).
- If someone else wants to reuse it, expect to modify the scripts rather than treating them as a library.
