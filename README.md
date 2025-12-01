# sbf-labeller

Tiny, very specific tool I use to **manually label data chunks derived from SBF (Septentrio Binary Format) logs**.

It is *not* a general package. It is just a helper I use together with my GNSS jamming generator:

- GNSS generator repo: <https://github.com/macaburguera/GNSS_generator>  
- Jamming classes: `NoJam`, `Chirp`, `NB`, `WB` (same taxonomy as in the generator).

The idea is simple:

1. I preprocess the raw SBF logs (outside this repo) into arrays / features.
2. I open a small GUI, look at each block, and assign one of those jamming classes.
3. The labels are saved so I can train and validate classifiers later.

---

## Files

```text
.
├── feature_extractor.py   # quick-and-dirty feature / data preparation script
├── label_gui.py           # small GUI to step through blocks and assign labels
└── .gitignore
```

### `feature_extractor.py`

- Reads whatever preprocessed SBF data I have on disk.
- Computes the views I want to see while labelling (for example, spectra or time series).
- Writes them to a file that `label_gui.py` expects.

Everything here is tailored to my folder structure and file naming, so expect to edit it for your own use.

### `label_gui.py`

- Loads the data / features prepared by `feature_extractor.py`.
- Shows one block at a time.
- Lets me press a key / click a button to assign a label:
  - `NoJam`
  - `Chirp`
  - `NB`
  - `WB`
- Stores the labels (e.g. CSV / NumPy file), which I then use in my ML repos.

---


## Notes

- This repository is intentionally small and opinionated.
- Many things are hard-coded (paths, class names, assumptions from Jammertest / GNSS_generator).
- If someone else wants to reuse it, expect to modify the scripts rather than treating them as a library.
