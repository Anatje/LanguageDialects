
# --- (Optional) make Hugging Face Datasets think torch is not installed ONLY for `download`
# Not strictly needed anymore since I am not usingg `datasets` in download, but harmless to keep.
# Torch did not work on my laptop therefore the entire code relies on pydub

import sys, importlib.util as _iu
if len(sys.argv) > 1 and sys.argv[1] == "download":
    _real_find_spec = _iu.find_spec
    def _find_spec_no_torch(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            return None  # pretend torch isn't installed
        return _real_find_spec(name, *a, **k)
    _iu.find_spec = _find_spec_no_torch
# ------------------------------------------------------------------------

# Make pydub find ffmpeg in any terminal (silences the warning)
import shutil
from pathlib import Path
from pydub import AudioSegment
if not shutil.which("ffmpeg"):
    home = Path.home()
    # WinGet default Links path under the current user's profile
    AudioSegment.converter = str(home / r"AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe")
    AudioSegment.ffprobe   = str(home / r"AppData\Local\Microsoft\WinGet\Links\ffprobe.exe")

import argparse, yaml, random
from pathlib import Path as _Path  # avoid shadowing above import
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local modules (none of these should import torch, else it throws an error)
from .manifest import normalize_dialect, filter_by_duration, drop_invalid, make_manifest
from .audio_utils import convert_to_wav16k_mono, recompute_duration_seconds
from .split_balance import stratified_group_split, balance_by_minutes
from .qc import stats_and_checks, duration_histogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("cmd", choices=[
        "download", "manifest", "convert", "split", "balance", "qc", "augment", "all"
    ])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    work = _Path(args.workdir or cfg.get("work_dir", "./data"))
    work.mkdir(exist_ok=True, parents=True)

    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    raw_dir = work / "raw"
    proc_dir = work / "processed"
    man_dir = work / "manifests"
    fig_dir = proc_dir / "figures"

    # ---------------------- DOWNLOAD ----------------------
    if args.cmd in ["download", "all"]:
        # Use the fast snapshot-based fetcher (no datasets, no torch)
        print("USING SNAPSHOT FETCHER (no datasets, no torch)")
        from .tools.cv17_fetch_min import fetch_common_voice_subset

        for ds in cfg["datasets"]:
            if ds.get("name") != "common_voice":
                continue

            df = fetch_common_voice_subset(
                locales=ds["locales"],                          # e.g. ["es","pt","ru","tr","sw"]
                version=ds["version"],                          # e.g. "17_0"
                minutes_per_locale=cfg.get("target_minutes_per_dialect", 5.0),
                min_duration_sec=ds.get("min_duration_sec", 0.0),
                max_duration_sec=ds.get("max_duration_sec", 1e9),
                splits=("test", "validation", "train"),         # smallest segment first, train is fallback option
                out_root=str(raw_dir / "cv17"),                 # audio saved under data/raw/cv17/<locale>/
            )

            out_csv = man_dir / "common_voice_raw.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            print(f"Wrote {out_csv} with {len(df)} rows")

    # ---------------------- MANIFEST ----------------------
    if args.cmd in ["manifest", "all"]:
        raw_csv = man_dir / "common_voice_raw.csv"
        if not raw_csv.exists():
            print(f"Missing {raw_csv}; run 'download' first.")
            return

        raw = pd.read_csv(raw_csv)
        # find the cv config
        ds_cfg = next((d for d in cfg["datasets"] if d.get("name") == "common_voice"), {})
        raw = drop_invalid(raw)

        if "min_duration_sec" in ds_cfg and "max_duration_sec" in ds_cfg:
            raw = raw.dropna(subset=["duration"])
            raw = filter_by_duration(raw, ds_cfg["min_duration_sec"], ds_cfg["max_duration_sec"])

        raw = normalize_dialect(raw, cfg.get("dialect_normalization", {}))
        # Keep only rows with known dialects
        manifest = raw.dropna(subset=["dialect"]).reset_index(drop=True)

        man_dir.mkdir(parents=True, exist_ok=True)
        make_manifest(manifest, man_dir / "manifest_raw.csv")
        print("Manifest rows:", len(manifest))

    # ---------------------- CONVERT -----------------------
    if args.cmd in ["convert", "all"]:
        in_manifest = man_dir / "manifest_raw.csv"
        if not in_manifest.exists():
            print(f"Missing {in_manifest}; run 'manifest' first.")
            return

        m = pd.read_csv(in_manifest)
        new_paths, new_durations = [], []
        for _, r in tqdm(m.iterrows(), total=len(m), desc="Converting"):
            in_path = r["path"]
            out_path = proc_dir / "audio" / r["language"] / r["dialect"] / _Path(in_path).name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            convert_to_wav16k_mono(in_path, str(out_path), sample_rate=cfg["audio"]["sample_rate"])
            new_paths.append(str(out_path))
            new_durations.append(recompute_duration_seconds(str(out_path)))

        m["path"] = new_paths
        m["duration"] = new_durations
        out_manifest = man_dir / "manifest_processed.csv"
        m.to_csv(out_manifest, index=False)
        print("Converted and updated durations ->", out_manifest)

    # ---------------------- SPLIT -------------------------
    if args.cmd in ["split", "all"]:
        in_manifest = man_dir / "manifest_processed.csv"
        if not in_manifest.exists():
            print(f"Missing {in_manifest}; run 'convert' first.")
            return

        m = pd.read_csv(in_manifest)
        m = stratified_group_split(m, **cfg["split"])
        out_manifest = man_dir / "manifest_split.csv"
        m.to_csv(out_manifest, index=False)
        print(m["split"].value_counts())
        print("Wrote", out_manifest)

    # ---------------------- BALANCE -----------------------
    if args.cmd in ["balance", "all"]:
        in_manifest = man_dir / "manifest_split.csv"
        if not in_manifest.exists():
            print(f"Missing {in_manifest}; run 'split' first.")
            return

        m = pd.read_csv(in_manifest)
        balanced = balance_by_minutes(m[m["split"].eq("train")], cfg["target_minutes_per_dialect"])
        # Keep val/test as-is; only balance train for demonstration
        out = pd.concat([balanced.assign(split="train"), m[m["split"].isin(["val", "test"])]], ignore_index=True)
        out_manifest = man_dir / "manifest_balanced.csv"
        out.to_csv(out_manifest, index=False)
        print("Balanced train rows:", len(balanced), "->", out_manifest)

    # ---------------------- QC ----------------------------
    if args.cmd in ["qc", "all"]:
        src = man_dir / "manifest_balanced.csv"
        if not src.exists():
            src = man_dir / "manifest_split.csv"
        if not src.exists():
            print("Missing manifests; run 'split' or 'balance' first.")
            return

        m = pd.read_csv(src)
        fig_dir.mkdir(parents=True, exist_ok=True)
        stats_and_checks(m)
        out_fig = fig_dir / "duration_hist.png"
        duration_histogram(m, str(out_fig))
        print("QC figure at", out_fig)

    # ---------------------- AUGMENT -----------------------
    if args.cmd in ["augment", "all"]:
        # Import here so 'download' never sees torchaudio/torch
        from .augment import augment_file

        if not cfg.get("augmentation", {}).get("enabled", False):
            print("Augmentation disabled in config.")
            return

        in_manifest = man_dir / "manifest_balanced.csv"
        if not in_manifest.exists():
            print(f"Missing {in_manifest}; run 'balance' first.")
            return

        m = pd.read_csv(in_manifest)
        speed_factors = cfg["augmentation"]["speed_factors"]
        snr_lo, snr_hi = cfg["augmentation"]["noise_snr_db"]

        train_rows = m[m["split"].eq("train")]
        aug_rows = []
        for _, r in tqdm(train_rows.iterrows(), total=len(train_rows), desc="Augmenting"):
            speed = random.choice(speed_factors)
            snr = random.uniform(snr_lo, snr_hi)
            in_path = r["path"]
            p = _Path(in_path)
            out_path = p.with_name(p.stem + f"_sp{speed:.2f}_snr{snr:.0f}.wav")
            augment_file(in_path, str(out_path), speed, snr)

            new_r = r.copy()
            new_r["path"] = str(out_path)
            # approximate new duration after speed perturb
            new_r["duration"] = float(new_r["duration"]) / float(speed)
            aug_rows.append(new_r)

        if aug_rows:
            aug_df = pd.DataFrame(aug_rows)
            out = pd.concat([m, aug_df], ignore_index=True)
            out_manifest = man_dir / "manifest_augmented.csv"
            out.to_csv(out_manifest, index=False)
            print("Augmented rows:", len(aug_df), "->", out_manifest)


if __name__ == "__main__":
    main()
