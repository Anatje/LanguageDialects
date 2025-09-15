"""
Streamlit app for browsing a language/dialect speech dataset.

Features
- Manifest loader with Arrow-friendly normalization.
- Language/dialect filters and row selection.
- Audio preview (waveform) and mel-spectrogram rendering.
- Safe, cached data/audio/spectrogram loaders.

Assumptions
- Manifest CSV contains at least a 'language' column; 'dialect' is optional and
  defaults to 'language' if missing.
- Audio files are mono or will be converted to mono at load time.
- Paths in the manifest are valid on the local filesystem.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import librosa
import librosa.display  # axes helpers for spectrograms
import soundfile as sf  # ensure soundfile backend is available for librosa


st.set_page_config(page_title="Dialect Browser", layout="wide")
st.title("Language & Dialect Dataset Browser")


# -----------------------------
# Arrow-friendly DataFrame utils
# -----------------------------
def make_arrow_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dtypes to be stable for Arrow/Streamlit (LLM suggestion, streamlit ran with normal pandas frame):
    - Known string-like columns -> pandas 'string' dtype
    - Known numeric columns -> numeric with coercion
    - Remaining object columns -> stringify (and decode bytes if present)
    """
    df = df.copy()

    # Known string-like columns
    for col in ["path", "language", "dialect", "speaker_id", "split", "sentence", "accent_raw"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Known numeric columns
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").astype("float32")

    # Any remaining object columns: ensure no bytes + make uniform strings
    for col in df.columns:
        if df[col].dtype == "object":
            has_bytes = df[col].map(lambda x: isinstance(x, (bytes, bytearray))).any()
            if has_bytes:
                df[col] = df[col].map(
                    lambda v: v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                ).astype("string")
            else:
                # Mixed types? stringify for Arrow stability
                df[col] = df[col].map(lambda v: v if isinstance(v, (int, float)) else str(v)).astype("string")
    return df


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_manifest(path: str) -> pd.DataFrame:
    """
    Read a manifest CSV and enforce Arrow-friendly dtypes.
    If 'dialect' is missing, copy from 'language'.
    """
    df = pd.read_csv(path)
    if "dialect" not in df.columns:
        df["dialect"] = df["language"]
    df = make_arrow_friendly(df)
    return df


@st.cache_data
def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load audio with native sampling rate; force mono for visualization consistency.
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


@st.cache_data
def compute_mel_db(path: str, n_mels: int = 128, hop: int = 512) -> tuple[np.ndarray, int]:
    """
    Compute mel-spectrogram in decibels for a given audio file.
    Returns (S_db, sr).
    """
    y, sr = load_audio(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop, fmax=sr // 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db, sr


# -----------------------------
# UI
# -----------------------------
manifest_path = st.text_input(
    "Manifest CSV path",
    "./data/manifests/manifest_balanced.csv",
    help="You can also point to manifest_raw.csv / manifest_processed.csv / manifest_split.csv",
)
if not manifest_path:
    st.stop()

# Load manifest
try:
    manifest_df = load_manifest(manifest_path)
except Exception as e:
    st.error(f"Failed to read manifest: {e}")
    st.stop()

with st.expander("Data preview (first 100 rows)"):
    st.dataframe(manifest_df.head(100), use_container_width=True)

# Filters
langs = sorted(manifest_df["language"].dropna().unique().tolist())
if not langs:
    st.warning("No languages found in manifest.")
    st.stop()
lang = st.selectbox("Language", langs)

dialects = sorted(manifest_df.loc[manifest_df["language"] == lang, "dialect"].dropna().unique().tolist())
if not dialects:
    st.warning("No dialects for this language.")
    st.stop()
dialect = st.selectbox("Dialect", dialects)

subset = manifest_df[(manifest_df["language"] == lang) & (manifest_df["dialect"] == dialect)].reset_index(drop=True)
st.write(f"Samples in selection: **{len(subset)}**")
if subset.empty:
    st.stop()

# Choose a row
idx = st.number_input(
    "Row index",
    min_value=0,
    max_value=max(0, len(subset) - 1),
    value=0,
    step=1
)
sample = subset.iloc[int(idx)]

# Show metadata
cols = st.columns(2)
with cols[0]:
    show_cols = [c for c in ["language", "dialect", "speaker_id", "duration", "split"] if c in subset.columns]
    st.write(sample[show_cols])
with cols[1]:
    st.write("Path:", sample["path"])

# Audio preview + plots
audio_path = str(sample["path"])
if not Path(audio_path).exists():
    st.error("File does not exist on disk. If a conversion step is required, run it first.")
    st.stop()

# Plot toggles / settings
c1, c2, c3 = st.columns(3)
with c1:
    show_wave = st.checkbox("Show waveform", value=True)
with c2:
    show_spec = st.checkbox("Show mel spectrogram", value=True)
with c3:
    n_mels = st.slider("Mel bins", min_value=64, max_value=256, value=128, step=16)

# Waveform
if show_wave:
    try:
        y, sr = load_audio(audio_path)
        t = np.linspace(0, len(y) / sr, num=len(y))
        fig, ax = plt.subplots(figsize=(12, 2.5), dpi=180)
        ax.plot(t, y, linewidth=0.8)
        ax.set(title="Waveform", xlabel="Time [s]", ylabel="Amplitude")
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render waveform: {e}")

# Mel spectrogram (with colorbar)
if show_spec:
    try:
        S_db, sr = compute_mel_db(audio_path, n_mels=n_mels, hop=512)
        fig, ax = plt.subplots(figsize=(12, 3.5), dpi=180)
        img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax)
        ax.set(title=f"Mel Spectrogram (dB) — sr={sr}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute spectrogram: {e}")

# Audio player
try:
    with open(audio_path, "rb") as f:
        st.audio(f.read(), format="audio/wav")
except Exception as e:
    st.warning(f"Could not load audio: {e}")
