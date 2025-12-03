# sbf-labeller

Tiny, very specific tool I use to **manually label BBSamples blocks from SBF (Septentrio Binary Format) logs**.

It is *not* a general package. It is just a helper I use together with my GNSS jamming generator:

- GNSS generator repo: <https://github.com/macaburguera/GNSS_generator>  
- Jamming-related classes (same taxonomy as in the generator, plus one extra):
  - `NoJam`
  - `Chirp`
  - `NB`
  - `WB`
  - `Interference` (for weird / non-jammer patterns)

The idea is:

1. I point the tool to an SBF file with `BBSamples` blocks.
2. I launch a small GUI (`label_gui.py`), step through candidate blocks, and assign one of the classes above.
3. The tool saves:
   - one **`.npz`** per labelled IQ snapshot, and  
   - one **`*_labels.csv`** with metadata and labels.  
4. I use those labels later to train and validate classifiers.

> **Entry point:** the script you actually run is **`label_gui.py`**.  
> `feature_extractor.py` is an optional helper module used only if you want model-assisted pre-labelling.

---

## Files

```text
.
├── label_gui.py           # main entry point – GUI to label / review SBF-derived IQ blocks
├── feature_extractor.py   # optional helper for model pre-labelling (extract_features)
└── .gitignore
```

---

## `label_gui.py` (main script)

### Running it

Basic usage:

```bash
python label_gui.py
```

On start, it shows a small menu:

- **Mode 1 – Label SBF file (new session or resume)**
- **Mode 2 – Review existing labelled dataset (CSV + NPZ)**

You normally live in Mode 1 while creating the dataset, and use Mode 2 later to clean up labels.

---

### Mode 1: Label SBF file (new / resume)

In this mode the script:

1. Asks for an **SBF file** (GUI dialog or path in the terminal).
2. Asks for a **sampling cadence**:
   - `Every block` (0 s → all `BBSamples` blocks)
   - `Every 10 seconds`
   - `Every 30 seconds`
   - `Custom (seconds)`
3. Scans the SBF and counts how many candidate blocks match that cadence.
4. Asks for an **output directory** (where NPZs + CSV will be saved).
5. Optionally loads a **pre-trained model** (if `feature_extractor.py` is available).
6. Opens the GUI and lets you label one sample at a time.

#### Classes / labels

Available labels:

- `NoJam`
- `Chirp`
- `NB`
- `WB`
- `Interference`  ← for suspicious / anomalous stuff that is clearly not clean GNSS but also not a classic jammer pattern.

#### What gets saved

For each labelled sample, the tool saves:

- A compressed **NPZ** file:
  - IQ data (complex64)
  - sampling frequency
  - basic time metadata (GPS week, TOW, UTC)
  - block index
  - SBF path
- A row in `<sbf_stem>_labels.csv` with:
  - `sample_id`
  - `label`
  - `iq_path`
  - `sbf_path`
  - `block_idx`
  - `gps_week`, `tow_s`, `utc_iso`
  - `fs_hz`

All labels for a given SBF live in a single `*_labels.csv` in the chosen output directory.

#### Session persistence (resume labelling)

If `<sbf_stem>_labels.csv` already exists in the output folder:

- The script **loads existing labels**.
- It scans the SBF again, builds the list of candidate blocks, and:
  - finds the **last candidate that already has a label**, and
  - **starts from the next one**.
- If you already labelled all candidates, it starts at the last one so you can still fix things with *Back*.

This lets you:

- Stop the session at any time (hit `q` in the GUI),
- Re-run `python label_gui.py` and select the same SBF + output directory,
- Continue exactly where you left off.

#### GUI controls (labelling mode)

Buttons and keys:

- **Class buttons / keys:**
  - `1`: `NoJam`
  - `2`: `Chirp`
  - `3`: `NB`
  - `4`: `WB`
  - `5`: `Interference`
- **Accept model prediction:**
  - Button: `Accept`
  - Keys: `a`, `Enter`, `Space`
- **Back (previous sample):**
  - Button: `Back`
  - Key: `b`
- **Skip (leave unlabeled / unchanged) for this pass:**
  - Button: `Skip`
  - Key: `s`
- **Quit session (save and exit):**
  - Button: `Quit`
  - Key: `q`

Display:

- **Upper panel**: spectrogram (STFT) of the IQ snapshot.
- **Lower panel**: I/Q waveforms vs time.
- Title line shows:
  - block index,
  - GPS week + TOW,
  - model prediction (if any),
  - **current saved label** (if this block was labeled before).

Colour hints:

- Button for the **model’s predicted class** is highlighted in **light green**.
- Button for the **currently saved label** (from the CSV) is highlighted in **orange**.

---

### Mode 2: Review existing labelled dataset (CSV + NPZ)

In this mode you:

1. Select an existing `*_labels.csv`.
2. The script loads all rows and corresponding NPZs (one per sample).
3. The same GUI opens, but now you are in **review / correction** mode.

You can:

- Step forward and backward through samples.
- Inspect spectrogram + I/Q.
- Change labels.

When you change the label:

- The NPZ file is **renamed** so the `sample_id` and filename reflect the new label.
- The row in the CSV is **updated**.
- The internal in-memory structure is kept in sync.

GUI controls (review mode):

- `1`–`5`: change label to that class (and rename NPZ + update CSV).
- `a` / `Enter` / `Space`: **keep** the current label and move on.
- `b`: go **back** to previous sample.
- `s`: **skip** (keep current label, move to next).
- `q`: **quit review**, saving all changes to the CSV.

---

## Model pre-labelling (`feature_extractor.py`)

`feature_extractor.py` is **optional**.

If it is present and defines:

- `extract_features(iq, fs)` → feature vector (1D array),

then `label_gui.py` can:

1. Ask if you want to enable **pre-labelling**.
2. Let you select a model file:
   - `*.joblib` / `*.pkl` (scikit-learn / XGBoost style),
   - `*.pt` / `*.pth` (PyTorch).
3. For each IQ sample:
   - compute features via `extract_features`,
   - run the model,
   - show the predicted class + probability.

From there you can:

- hit **Accept** to use the model’s label, or
- choose another class manually.

If `feature_extractor.py` or the model is missing, the script simply runs without pre-labelling.

---

## Notes

- This repository is intentionally small and opinionated.
- Many things are hard-coded (paths, class names, assumptions from Jammertest / GNSS_generator).
- It’s meant as a **tool**, not a library. If you want to adapt it, expect to edit the scripts.
