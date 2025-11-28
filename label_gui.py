#!/usr/bin/env python
"""
Interactive SBF labelling tool with optional model pre-labelling.

- Loads BBSamples from an SBF file (via sbf_parser.SbfParser).
- Lets you choose sampling cadence:
    * every block
    * every 10 seconds
    * every 30 seconds
    * custom (seconds)
- First pass: counts how many candidate samples that cadence would produce.
  Asks for confirmation; allows you to change cadence until you’re happy.
- Lets you choose an output directory (GUI dialog, with CLI fallback).
- Optional pre-labelling:
    * Load a model (.joblib/.pkl or .pt/.pth).
    * Uses feature_extractor.extract_features(iq, fs).
    * For joblib: expects sklearn/XGBoost-like API (predict, predict_proba).
    * For PyTorch: expects a model that maps [1, D] -> [1, num_classes].
- GUI:
    * Upper panel: spectrogram (STFT) – visually matched to validation.py.
    * Lower panel: I/Q waveforms.
    * Shows model prediction + probability.
    * Buttons: NoJam, Chirp, NB, WB, Accept, Skip, Quit.
    * Keyboard:
        - 1: NoJam
        - 2: Chirp
        - 3: NB
        - 4: WB
        - a / Enter / Space: Accept model prediction
        - s: Skip
        - q: Quit
"""

from __future__ import annotations
import os
import sys
import csv
from pathlib import Path
from typing import Optional, Iterable, Tuple, List, Dict, Callable

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import stft

from datetime import datetime, timedelta, timezone

# --- Match look & feel of validation.py ---
matplotlib.rcParams["image.cmap"] = "viridis"
matplotlib.rcParams["figure.dpi"] = 140
plt.style.use("default")

# SBF parser
from sbf_parser import SbfParser

# Feature extractor (if present)
try:
    from feature_extractor import extract_features, FEATURE_NAMES
    HAS_FEATURE_EXTRACTOR = True
except ImportError:
    extract_features = None
    FEATURE_NAMES = []
    HAS_FEATURE_EXTRACTOR = False

# ============================ CONFIG ============================

CLASS_LABELS = ["NoJam", "Chirp", "NB", "WB"]

# Spectrogram params (same as validation.py)
NPERSEG = 64
NOVERLAP = 56
REMOVE_DC = True
VMIN_DB = -80
VMAX_DB = -20

# GPS <-> UTC
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0  # seconds (valid for Jammertest)

EPS = 1e-20

# ============================ TIME HELPERS ============================

def gps_week_tow_to_utc(wn: int, tow_s: float) -> datetime:
    dt_gps = GPS_EPOCH + timedelta(weeks=int(wn), seconds=float(tow_s))
    return dt_gps - timedelta(seconds=GPS_MINUS_UTC)


def extract_time_labels(infos: Dict) -> Tuple[int, float, str, str, str, datetime]:
    wnc = int(infos.get("WNc", -1))
    tow_raw = float(infos.get("TOW", 0.0))
    # In some SBFs TOW is in ms
    tow_s = tow_raw / 1000.0 if tow_raw > 604800.0 else tow_raw

    tow_h = int(tow_s // 3600)
    tow_m = int((tow_s % 3600) // 60)
    tow_sec = tow_s - tow_h * 3600 - tow_m * 60
    tow_hms = f"{tow_h:02d}:{tow_m:02d}:{tow_sec:06.3f}"

    utc_dt = gps_week_tow_to_utc(wnc, tow_s)
    utc_hms = utc_dt.strftime("%H:%M:%S.%f")[:-3]
    utc_iso = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    return wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt

# ============================ SBF / IQ HELPERS ============================

def decode_bbsamples_iq(infos: Dict) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Decode BBSamples -> complex IQ float32, same as in validation.py."""
    if "Samples" not in infos or "N" not in infos:
        return None, None

    buf = infos["Samples"]
    N = int(infos["N"])

    arr = np.frombuffer(buf, dtype=np.int8)
    if arr.size != 2 * N:
        return None, None

    I = arr[0::2].astype(np.float32) / 128.0
    Q = arr[1::2].astype(np.float32) / 128.0
    x = I + 1j * Q
    fs = float(infos.get("SampleFreq", 1.0))
    return x, fs


def iter_candidate_blocks(
    sbf_path: Path,
    cadence_sec: float,
    decim: int = 1
) -> Iterable[Tuple[int, np.ndarray, float, Dict]]:
    """
    Yield (block_idx, iq, fs, meta_dict) for BBSamples matching the cadence.

    Cadence logic matches validation.py:
    - cadence_sec == 0  → keep every BBSamples block
    - cadence_sec > 0   → align to fixed UTC boundaries of 'cadence_sec' and
                          keep the first block at/after each boundary.
    """
    parser = SbfParser()
    block_idx = -1
    next_save_time: Optional[datetime] = None

    with open(sbf_path, "rb") as f:
        while True:
            chunk = f.read(1_000_000)
            if not chunk:
                break

            for blk, infos in parser.parse(chunk):
                if blk != "BBSamples":
                    continue

                block_idx += 1

                iq, fs = decode_bbsamples_iq(infos)
                if iq is None:
                    continue

                if decim and decim > 1:
                    iq = iq[::decim]
                    fs = fs / decim

                gps_week, tow_s, tow_hms, utc_hms, utc_iso, utc_dt = extract_time_labels(infos)
                meta = dict(
                    gps_week=gps_week,
                    tow_s=tow_s,
                    tow_hms=tow_hms,
                    utc_hms=utc_hms,
                    utc_iso=utc_iso,
                    utc_dt=utc_dt,
                )

                # --- No cadence → every block ---
                if cadence_sec <= 0:
                    yield block_idx, iq.astype(np.complex64), fs, meta
                    continue

                # --- Same gating as validation.py ---
                stride = int(cadence_sec)

                if next_save_time is None:
                    # Anchor to nearest lower multiple of 'stride' seconds,
                    # then move to the next boundary.
                    floor = utc_dt.replace(
                        second=(utc_dt.second // stride) * stride,
                        microsecond=0,
                    )
                    if floor > utc_dt:
                        floor -= timedelta(seconds=stride)
                    next_save_time = floor + timedelta(seconds=stride)

                # Skip until we reach the next boundary
                if utc_dt < next_save_time:
                    continue

                # If we jumped over more than one boundary, catch up
                while utc_dt >= next_save_time + timedelta(seconds=stride):
                    next_save_time += timedelta(seconds=stride)

                # Now utc_dt is in [next_save_time, next_save_time + stride)
                # → keep this block
                yield block_idx, iq.astype(np.complex64), fs, meta

                # Move boundary forward for the next sample
                next_save_time = next_save_time + timedelta(seconds=stride)


def count_candidates(sbf_path: Path, cadence_sec: float) -> int:
    return sum(1 for _ in iter_candidate_blocks(sbf_path, cadence_sec))

# ============================ GUI HELPERS (Tk) ============================

def _tk_choose_file(
    title: str,
    filetypes=None,
    initialdir: Optional[Path] = None
) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        root.attributes("-topmost", True)

        fname = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes if filetypes is not None else [("All files", "*.*")],
            initialdir=str(initialdir) if initialdir is not None else None,
        )
        root.destroy()
        if not fname:
            return None
        return Path(fname).expanduser().resolve()
    except Exception:
        return None


def _tk_choose_dir(
    title: str,
    initialdir: Optional[Path] = None
) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        root.attributes("-topmost", True)

        dname = filedialog.askdirectory(
            title=title,
            initialdir=str(initialdir) if initialdir is not None else None,
        )
        root.destroy()
        if not dname:
            return None
        p = Path(dname).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return None


def choose_sbf_file_from_gui() -> Optional[Path]:
    print(">>> Select SBF file in the file dialog (or cancel to type the path).")
    return _tk_choose_file(
        title="Select SBF file",
        filetypes=[("SBF files", "*.sbf"), ("All files", "*.*")]
    )


def choose_output_dir(initial: Optional[Path] = None) -> Path:
    """
    Prefer GUI directory selection; fallback to CLI path.
    """
    print("\n>>> Select OUTPUT directory in the dialog (or cancel to type it).")
    p = _tk_choose_dir(
        title="Select output directory for labelled dataset",
        initialdir=initial
    )
    if p is not None:
        print(f"Output directory: {p}")
        return p

    while True:
        if initial is not None:
            txt = input(
                f"Output directory (ENTER for default = '{initial}'): "
            ).strip()
            if not txt:
                p = initial
            else:
                p = Path(txt).expanduser()
        else:
            txt = input("Output directory (will be created if missing): ").strip()
            if not txt:
                print("Please enter a non-empty path.")
                continue
            p = Path(txt).expanduser()

        p = p.resolve()
        p.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {p}")
        return p


def choose_model_file(initial: Optional[Path] = None) -> Optional[Path]:
    """
    Choose model file (.joblib/.pkl/.pt/.pth) via GUI, fallback to CLI.
    """
    print("\n>>> Select MODEL file in the dialog (or cancel to type the path).")
    p = _tk_choose_file(
        title="Select pre-trained model (.joblib/.pkl/.pt/.pth)",
        filetypes=[
            ("Model files", "*.joblib *.pkl *.pt *.pth"),
            ("All files", "*.*")
        ],
        initialdir=initial,
    )
    if p is not None:
        return p

    txt = input("Enter model path (.joblib/.pkl/.pt/.pth): ").strip()
    if not txt:
        return None
    p = Path(txt).expanduser().resolve()
    if not p.exists():
        print(f"Model file not found: {p}")
        return None
    return p

# ============================ GUI LABELLING ============================

class LabelFigure:
    """
    Matplotlib-based GUI for interactive labelling.

    - Upper panel: spectrogram (same STFT & scaling as validation.py)
    - Lower panel: I/Q waveform
    """

    def __init__(self, class_labels: List[str]):
        self.class_labels = class_labels
        self.choice: Optional[str] = None

        self.fig, (self.ax_spec, self.ax_wave) = plt.subplots(2, 1, figsize=(10, 7))
        plt.subplots_adjust(bottom=0.22)

        self.btn_class: List[Button] = []
        self.btn_accept: Optional[Button] = None
        self.btn_skip: Optional[Button] = None
        self.btn_quit: Optional[Button] = None
        self.cbar_spec = None

        self._build_buttons()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        plt.ion()
        plt.show()

    # ----- buttons & events -----

    def _build_buttons(self):
        n_cls = len(self.class_labels)
        total_buttons = n_cls + 3  # + Accept, Skip, Quit
        pad = 0.015
        width = (0.92 - pad * (total_buttons + 1)) / total_buttons
        left = 0.04
        y0 = 0.08
        h = 0.07

        for i, label in enumerate(self.class_labels):
            axb = self.fig.add_axes([left + i * (width + pad), y0, width, h])
            btn = Button(axb, f"{i+1}: {label}")
            btn.on_clicked(self._make_button_handler(label))
            self.btn_class.append(btn)

        ax_accept = self.fig.add_axes([left + n_cls * (width + pad), y0, width, h])
        self.btn_accept = Button(ax_accept, "a / ↲ : Accept")
        self.btn_accept.on_clicked(self._make_button_handler("__accept__"))

        ax_skip = self.fig.add_axes([left + (n_cls + 1) * (width + pad), y0, width, h])
        self.btn_skip = Button(ax_skip, "s: Skip")
        self.btn_skip.on_clicked(self._make_button_handler("__skip__"))

        ax_quit = self.fig.add_axes([left + (n_cls + 2) * (width + pad), y0, width, h])
        self.btn_quit = Button(ax_quit, "q: Quit")
        self.btn_quit.on_clicked(self._make_button_handler("__quit__"))

    def _make_button_handler(self, action: str):
        def _handler(event):
            self.choice = action
        return _handler

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in ("1", "2", "3", "4"):
            idx = int(key) - 1
            if 0 <= idx < len(self.class_labels):
                self.choice = self.class_labels[idx]
        elif key in ("a", "enter", "return", " "):
            self.choice = "__accept__"
        elif key == "s":
            self.choice = "__skip__"
        elif key == "q":
            self.choice = "__quit__"

    def reset_button_colors(self):
        for btn in self.btn_class:
            btn.ax.set_facecolor("0.85")
        if self.btn_accept is not None:
            self.btn_accept.ax.set_facecolor("0.85")
        if self.btn_skip is not None:
            self.btn_skip.ax.set_facecolor("0.85")
        if self.btn_quit is not None:
            self.btn_quit.ax.set_facecolor("0.85")

    # ----- plotting -----

    def show_sample(
        self,
        iq: np.ndarray,
        fs: float,
        meta: Dict,
        block_idx: int,
        total_count: int,
        sample_idx: int,
        pred_label: Optional[str] = None,
        pred_proba: Optional[Dict[str, float]] = None,
    ):
        """
        Update figure with new IQ sample.
        STFT / spectrogram logic is copied from validation.py::plot_and_save.
        """
        self.choice = None

        # --------- Signal preprocessing (same as validation.py) ---------
        xx = iq.astype(np.complex64, copy=False)
        if REMOVE_DC and xx.size > 0:
            xx = xx - np.mean(xx)

        if xx.size > 0:
            nperseg_eff = min(int(NPERSEG), len(xx))
            noverlap_eff = min(int(NOVERLAP), max(0, nperseg_eff - 1))

            f, t_stft, Z = stft(
                xx, fs=fs, window="hann",
                nperseg=nperseg_eff, noverlap=noverlap_eff,
                return_onesided=False, boundary=None, padded=False
            )
            if t_stft.size < 2:
                nperseg_eff = max(16, min(len(xx) // 4, nperseg_eff))
                noverlap_eff = int(0.9 * nperseg_eff)
                f, t_stft, Z = stft(
                    xx, fs=fs, window="hann",
                    nperseg=nperseg_eff, noverlap=noverlap_eff,
                    return_onesided=False, boundary=None, padded=False
                )

            Z = np.fft.fftshift(Z, axes=0)
            f = np.fft.fftshift(f)
            S_dB = 20.0 * np.log10(np.abs(Z) + EPS)
        else:
            f = np.array([0.0, 1.0], dtype=float)
            t_stft = np.array([0.0, 1.0], dtype=float)
            S_dB = np.zeros((2, 2), dtype=float)
            nperseg_eff = NPERSEG

        # --------- Spectrogram axes (mirror validation.py) ---------
        self.ax_spec.cla()

        if t_stft.size >= 2:
            mappable = self.ax_spec.pcolormesh(
                t_stft, f, S_dB,
                shading="auto", vmin=VMIN_DB, vmax=VMAX_DB
            )
        else:
            mappable = self.ax_spec.imshow(
                S_dB, aspect="auto", origin="lower",
                extent=[0.0, max(1.0 / fs, nperseg_eff / fs), f[0], f[-1]],
                vmin=VMIN_DB, vmax=VMAX_DB
            )

        if self.cbar_spec is None:
            self.cbar_spec = self.fig.colorbar(mappable, ax=self.ax_spec, label="dB")
        else:
            try:
                self.cbar_spec.update_normal(mappable)
            except Exception:
                try:
                    self.cbar_spec.remove()
                except Exception:
                    pass
                self.cbar_spec = self.fig.colorbar(mappable, ax=self.ax_spec, label="dB")

        self.ax_spec.set_ylabel("Frequency [Hz]")

        # --------- Waveform axes (same as validation.py) ---------
        self.ax_wave.cla()
        if xx.size > 0:
            tt = np.arange(len(xx), dtype=np.float32) / fs
            I = np.real(xx)
            Q = np.imag(xx)
            self.ax_wave.plot(tt, I, linewidth=0.7, label="I")
            self.ax_wave.plot(tt, Q, linewidth=0.7, alpha=0.85, label="Q")
        self.ax_wave.set_xlabel("Time [s]")
        self.ax_wave.set_ylabel("Amplitude (norm.)")
        self.ax_wave.legend(loc="upper right")

        utc_iso = meta.get("utc_iso", "")
        gps_week = meta.get("gps_week", -1)
        tow_s = meta.get("tow_s", 0.0)
        tow_hms = meta.get("tow_hms", "")

        # Prediction text, like "Pred: Chirp (0.95)" in validation
        if pred_label is not None:
            if isinstance(pred_proba, dict) and pred_label in pred_proba:
                p = pred_proba[pred_label]
                pred_txt = f"Pred: {pred_label} ({p:.2f})"
            else:
                pred_txt = f"Pred: {pred_label}"
        else:
            pred_txt = "Pred: (none)"

        title = (
            f"Spectrogram (BBSamples #{block_idx})  |  GPS week {gps_week}  |  "
            f"TOW {tow_s:.3f}s  |  {pred_txt}\n"
            f"Sample {sample_idx}/{total_count}   nperseg={nperseg_eff}, noverlap={NOVERLAP}"
        )
        self.ax_spec.set_title(title)

        self.ax_wave.text(
            0.01, 0.02, utc_iso,
            transform=self.ax_wave.transAxes,
            fontsize=8, ha="left", va="bottom"
        )

        # Button highlights
        self.reset_button_colors()
        if pred_label in self.class_labels:
            idx = self.class_labels.index(pred_label)
            self.btn_class[idx].ax.set_facecolor("lightgreen")

        self.fig.canvas.draw_idle()

    def wait_for_choice(self) -> str:
        while self.choice is None:
            plt.pause(0.1)
        return self.choice

# ============================ IO HELPERS ============================

def open_label_csv(out_dir: Path, sbf_path: Path) -> Tuple[csv.DictWriter, object]:
    base = sbf_path.stem
    csv_path = out_dir / f"{base}_labels.csv"
    file_exists = csv_path.exists()

    fh = open(csv_path, "a", newline="", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "label",
        "iq_path",
        "sbf_path",
        "block_idx",
        "gps_week",
        "tow_s",
        "utc_iso",
        "fs_hz",
    ]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    return writer, fh

# ============================ PRE-LABEL MODEL ============================

def setup_prelabel_model() -> Optional[Callable[[np.ndarray], Tuple[Optional[str], Dict[str, float]]]]:
    ans = input("\nDo you want to pre-label using a trained model? [y/N]: ").strip().lower()
    if ans != "y":
        print("Pre-labelling disabled.")
        return None

    if not HAS_FEATURE_EXTRACTOR:
        print("Pre-labelling requested, but feature_extractor.py not found.")
        print("Put your extract_features(...) function in feature_extractor.py and try again.")
        return None

    model_path = choose_model_file()
    if model_path is None or not model_path.exists():
        print("Model file not selected or not found. Pre-labelling disabled.")
        return None

    ext = model_path.suffix.lower()
    if ext in (".joblib", ".pkl"):
        try:
            from joblib import load as joblib_load
        except ImportError:
            print("joblib is not installed. Install it to use .joblib/.pkl models.")
            return None

        print(f"Loading joblib model: {model_path}")
        model = joblib_load(model_path)

        def make_joblib_predict_fn(m):
            classes_attr = getattr(m, "classes_", None)

            if classes_attr is not None:
                arr = np.array(classes_attr)
                if np.issubdtype(arr.dtype, np.integer):
                    idx_to_name = {
                        int(i): CLASS_LABELS[int(i)]
                        for i in range(min(len(CLASS_LABELS), len(arr)))
                    }

                    def to_name(y):
                        y = int(y)
                        return idx_to_name.get(y, str(y))

                    raw_classes = arr
                else:
                    def to_name(y):
                        return str(y)

                    raw_classes = arr
            else:
                raw_classes = np.arange(len(CLASS_LABELS))

                def to_name(y):
                    if isinstance(y, (int, np.integer)) and 0 <= int(y) < len(CLASS_LABELS):
                        return CLASS_LABELS[int(y)]
                    return str(y)

            def predict_fn(feat_vec: np.ndarray):
                try:
                    yhat = m.predict([feat_vec])[0]
                    label = to_name(yhat)
                    proba_dict: Dict[str, float] = {}
                    if hasattr(m, "predict_proba"):
                        probs = m.predict_proba([feat_vec])[0]
                        if classes_attr is not None:
                            names = [to_name(c) for c in raw_classes]
                        else:
                            names = CLASS_LABELS[:len(probs)]
                        proba_dict = {n: float(p) for n, p in zip(names, probs)}
                    return label, proba_dict
                except Exception:
                    return None, {}

            return predict_fn

        return make_joblib_predict_fn(model)

    elif ext in (".pt", ".pth"):
        try:
            import torch
        except ImportError:
            print("torch is not installed. Install it to use .pt/.pth models.")
            return None

        print(f"Loading PyTorch model: {model_path}")
        model = torch.load(model_path, map_location="cpu")
        model.eval()

        def make_torch_predict_fn(m):
            def predict_fn(feat_vec: np.ndarray):
                import torch
                x = torch.from_numpy(np.asarray(feat_vec, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    out = m(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                out = out.squeeze(0)
                probs = torch.softmax(out, dim=-1).cpu().numpy()
                idx = int(np.argmax(probs)) if probs.size > 0 else -1
                if 0 <= idx < len(CLASS_LABELS):
                    label = CLASS_LABELS[idx]
                else:
                    label = str(idx)
                proba_dict = {
                    CLASS_LABELS[i]: float(probs[i])
                    for i in range(min(len(CLASS_LABELS), len(probs)))
                }
                return label, proba_dict

            return make_torch_predict_fn(model)

        return make_torch_predict_fn(model)

    else:
        print(f"Unsupported model extension: {ext}")
        print("Supported: .joblib, .pkl, .pt, .pth")
        return None

# ============================ MAIN FLOW ============================

def prompt_cadence() -> float:
    while True:
        print("\nSelect sampling cadence:")
        print("  1) Every block")
        print("  2) Every 10 seconds")
        print("  3) Every 30 seconds")
        print("  4) Custom (seconds)")
        choice = input("Your choice [1-4]: ").strip()

        if choice == "1":
            return 0.0
        elif choice == "2":
            return 10.0
        elif choice == "3":
            return 30.0
        elif choice == "4":
            txt = input("Enter cadence in seconds (e.g. 5, 10, 30): ").strip()
            try:
                val = float(txt)
                if val <= 0:
                    print("Cadence must be positive. Try again.")
                    continue
                return val
            except ValueError:
                print("Could not parse number. Try again.")
        else:
            print("Invalid choice. Please select 1–4.")


def main():
    # ----- SBF PATH (GUI preferred) -----
    if len(sys.argv) > 1:
        sbf_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        sbf_path = choose_sbf_file_from_gui()
        if sbf_path is None:
            txt = input("Enter SBF file path: ").strip()
            sbf_path = Path(txt).expanduser().resolve()

    if not sbf_path.exists():
        print(f"ERROR: SBF file not found: {sbf_path}")
        sys.exit(1)

    print(f"\nSBF file: {sbf_path}")

    # ----- Cadence + counting -----
    while True:
        cadence_sec = prompt_cadence()
        desc = "every BBSamples block" if cadence_sec <= 0 else f"every {cadence_sec:.1f} seconds"
        print(f"\nCounting candidate samples ({desc})...")
        num_candidates = count_candidates(sbf_path, cadence_sec)
        print(f"  → This SBF will produce {num_candidates} samples.")

        if num_candidates == 0:
            print("No samples found with that cadence. Please choose again.")
            continue

        ans = input("Proceed with this cadence? [y/N]: ").strip().lower()
        if ans == "y":
            break

    # ----- Output directory -----
    out_dir = choose_output_dir(initial=sbf_path.parent)
    writer, fh_csv = open_label_csv(out_dir, sbf_path)

    # ----- Pre-labelling -----
    prelabel_predict = setup_prelabel_model()

    # ----- GUI -----
    fig_tool = LabelFigure(CLASS_LABELS)

    print("\nStarting labelling session.")
    print("Keyboard shortcuts:")
    for i, lbl in enumerate(CLASS_LABELS, start=1):
        print(f"  {i}: {lbl}")
    print("  a / Enter / Space: Accept model prediction")
    print("  s: Skip sample")
    print("  q: Quit session\n")

    sample_idx = 0

    for block_idx, iq, fs, meta in iter_candidate_blocks(sbf_path, cadence_sec):
        sample_idx += 1

        pred_label = None
        pred_proba: Dict[str, float] = {}
        if prelabel_predict is not None and extract_features is not None:
            try:
                feat_vec = extract_features(iq, fs)
                pred_label, pred_proba = prelabel_predict(feat_vec)
            except Exception as e:
                print(f"[WARN] Pre-labelling failed on block {block_idx}: {e}")
                pred_label, pred_proba = None, {}

        fig_tool.show_sample(
            iq, fs, meta,
            block_idx=block_idx,
            total_count=num_candidates,
            sample_idx=sample_idx,
            pred_label=pred_label,
            pred_proba=pred_proba,
        )

        while True:
            choice = fig_tool.wait_for_choice()

            if choice == "__quit__":
                print("Quitting labelling session.")
                fh_csv.close()
                plt.close("all")
                print(f"\nLabelling finished. Labels saved in: {out_dir}")
                return

            if choice == "__skip__":
                print(f"Sample {sample_idx}/{num_candidates} (block {block_idx}) skipped.")
                label = None
                break

            if choice == "__accept__":
                if pred_label is None:
                    print("No model prediction available for this sample. Please choose a class (1–4).")
                    continue
                label = pred_label
                break

            if choice in CLASS_LABELS:
                label = choice
                break

            print("Unknown choice, waiting again...")

        if label is None:
            continue  # skipped

        # ----- Save labelled sample -----
        base = sbf_path.stem
        sample_id = f"{base}_blk{block_idx:06d}_{label}"
        iq_path = out_dir / f"{sample_id}.npz"

        np.savez_compressed(
            iq_path,
            iq=iq.astype(np.complex64),
            fs_hz=float(fs),
            gps_week=int(meta["gps_week"]),
            tow_s=float(meta["tow_s"]),
            utc_iso=str(meta["utc_iso"]),
            block_idx=int(block_idx),
            sbf_path=str(sbf_path),
        )

        writer.writerow({
            "sample_id": sample_id,
            "label": label,
            "iq_path": str(iq_path),
            "sbf_path": str(sbf_path),
            "block_idx": block_idx,
            "gps_week": meta["gps_week"],
            "tow_s": meta["tow_s"],
            "utc_iso": meta["utc_iso"],
            "fs_hz": fs,
        })
        fh_csv.flush()
        print(f"Labeled sample {sample_idx}/{num_candidates} (block {block_idx}) as {label}.")

    fh_csv.close()
    plt.close("all")
    print(f"\nLabelling finished. Labels saved in: {out_dir}")


if __name__ == "__main__":
    main()
