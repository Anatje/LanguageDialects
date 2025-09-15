# src/tools/cv17_fetch_min.py
from __future__ import annotations
from pathlib import Path
import os, sys, csv, tarfile, shutil, random, subprocess
from typing import List, Dict, Tuple, Optional
import pandas as pd
from huggingface_hub import snapshot_download

# --- CSV can contain very long "sentence" fields; raise the limit
try:
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
except Exception:
    pass

SplitOrder = Tuple[str, ...]
AUDIO_EXTS = (".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a", ".webm")


# ---------------- ffprobe helpers ----------------
def _ffprobe_path() -> Optional[str]:
    # 1) honor explicit env var if you set it in PowerShell
    p = os.environ.get("FFPROBE_PATH")
    if p and Path(p).exists():
        return p
    # 2) try PATH
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    # 3) WinGet shim (common on Windows)
    candidate = Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffprobe.exe"
    return str(candidate) if candidate.exists() else None


def _ffprobe_duration_seconds(path: Path) -> Optional[float]:
    exe = _ffprobe_path()
    if not exe or not path.exists():
        return None
    try:
        out = subprocess.check_output(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stderr=subprocess.STDOUT,
        )
        return float(out.decode().strip())
    except Exception:
        return None


# ---------------- TSV helpers ----------------
def _read_manifest_rows(tsv_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not tsv_path or not tsv_path.exists():
        return rows
    # csv module recommends newline=""
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            base = Path(r["path"]).name  # normalize to filename
            dur = r.get("duration")
            if dur is None or dur == "":
                dms = r.get("duration_ms")
                dur = float(dms) / 1000.0 if dms not in (None, "") else 0.0
            else:
                dur = float(dur)
            rows.append({
                "filename": base,
                "duration": float(dur),
                "accent_raw": (r.get("accent") or "").strip().lower(),
                "speaker_id": str(r.get("client_id") or ""),
            })
    return rows


def _build_meta_index(tsv_paths: list[Path]) -> dict[str, dict]:
    """filename -> {duration, accent_raw, speaker_id}"""
    idx: dict[str, dict] = {}
    for tsv in tsv_paths:
        for r in _read_manifest_rows(tsv):
            idx[r["filename"]] = {
                "duration": r["duration"],
                "accent_raw": r["accent_raw"],
                "speaker_id": r["speaker_id"],
            }
    return idx


# ---------------- download helpers ----------------
def _download_only_tsv(*, repo_id: str, loc: str, split_name: str, cache_key: str) -> tuple[str, Path]:
    """Fetch just transcript/<loc>/<split>.tsv; return (repo_dir, tsv_path)."""
    repo_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"transcript/{loc}/{split_name}.tsv"],
        local_dir=str(Path(".hf_cache") / cache_key),
        etag_timeout=60,
        token=True,
        max_workers=8,
    )
    tsv = Path(repo_dir) / "transcript" / loc / f"{split_name}.tsv"
    return repo_dir, tsv


def _download_split_tar(*, repo_id: str, repo_dir: str, loc: str, split_name: str, tar_index: int) -> Path | None:
    """Fetch a single tar for test/validated: e.g. pt_validated_0.tar"""
    tar_name = f"{loc}_{split_name}_{tar_index}.tar"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"audio/{loc}/{split_name}/{tar_name}"],
        local_dir=repo_dir,
        etag_timeout=60,
        token=True,
        max_workers=8,
    )
    p = Path(repo_dir) / "audio" / loc / split_name / tar_name
    return p if p.exists() else None


def _download_train_tar(*, repo_id: str, repo_dir: str, loc: str, tar_index: int) -> Path | None:
    """Fetch a single train tar: e.g. es_train_0.tar"""
    tar_name = f"{loc}_train_{tar_index}.tar"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"audio/{loc}/train/{tar_name}"],
        local_dir=repo_dir,
        etag_timeout=60,
        token=True,
        max_workers=8,
    )
    p = Path(repo_dir) / "audio" / loc / "train" / tar_name
    return p if p.exists() else None


# ---------------- extraction helpers ----------------
def _index_tar_members(tar_path: Path) -> Dict[str, tarfile.TarInfo]:
    """Map base filename -> TarInfo, for audio files only."""
    idx: Dict[str, tarfile.TarInfo] = {}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if m.isfile() and Path(m.name).suffix.lower() in AUDIO_EXTS:
                idx[Path(m.name).name] = m
    return idx


def _extract_from_tars_scanning(
    *,
    tars: list[Path],
    out_dir: Path,
    target_remaining: float,
    already: set[str],
    meta_idx: dict[str, dict],
    min_sec: float,
    max_sec: float,
) -> tuple[list[dict], float]:
    """Scan tar contents; prefer TSV duration; else ffprobe; keep 1..15s; stop at target."""
    kept: list[dict] = []
    added = 0.0
    for tar_path in tars:
        if added >= target_remaining:
            break
        idx = _index_tar_members(tar_path)
        with tarfile.open(tar_path, "r") as tf:
            for base, m in idx.items():
                if added >= target_remaining:
                    break
                if base in already:
                    continue

                # extract ONLY this member
                tf.extract(m, path=out_dir)
                extracted = out_dir / m.name
                final_path = out_dir / base
                try:
                    if extracted != final_path:
                        extracted.replace(final_path)
                except Exception:
                    shutil.copy2(extracted, final_path)
                    try:
                        extracted.unlink()
                    except Exception:
                        pass

                # duration: TSV -> ffprobe -> last-resort 2.0s
                meta = meta_idx.get(base)
                dur = float(meta["duration"]) if (meta and meta.get("duration", 0) > 0) else (_ffprobe_duration_seconds(final_path) or 0.0)
                if not dur:
                    dur = 2.0  # last resort to make progress; adjust/disable later if you prefer

                if dur < min_sec or dur > max_sec:
                    # drop and continue
                    try:
                        final_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue

                kept.append({
                    "path": str(final_path),
                    "duration": float(dur),
                    "accent_raw": (meta.get("accent_raw") if meta else ""),
                    "speaker_id": (meta.get("speaker_id") if meta else ""),
                })
                already.add(base)
                added += float(dur)
    return kept, added


def _scan_folder_to_rows(loc: str, loc_out: Path, meta_idx: dict, min_sec: float, max_sec: float) -> list[dict]:
    """Return rows for EVERYTHING already extracted in loc_out (existing + new)."""
    rows = []
    for p in loc_out.glob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        base = p.name
        dur = meta_idx.get(base, {}).get("duration")
        if not dur:
            dur = _ffprobe_duration_seconds(p) or 0.0
        if dur < min_sec or dur > max_sec:
            continue
        rows.append({
            "path": str(p),
            "language": loc,
            "accent_raw": meta_idx.get(base, {}).get("accent_raw", ""),
            "speaker_id": meta_idx.get(base, {}).get("speaker_id", ""),
            "duration": float(dur),
        })
    return rows


# ---------------- main entry ----------------
def fetch_common_voice_subset(
    *,
    locales: List[str],
    version: str,
    minutes_per_locale: float = 5.0,
    min_duration_sec: float = 1.0,
    max_duration_sec: float = 15.0,
    splits: SplitOrder = ("test", "validated", "train"),
    out_root: str = "data/cv17",
) -> pd.DataFrame:
    """
    Torch-free CV fetcher that:
      - reads TSVs from transcript/<loc>/<split>.tsv
      - counts what's already extracted in data/cv17/<loc>
      - uses test -> validated; train (one tar at a time) only if still short
      - scans tar contents; durations via TSV or ffprobe; last-resort 2.0s
      - returns ALL files present (existing + new) and de-dupes by path
    """
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    repo_id = f"mozilla-foundation/common_voice_{version}"
    target_sec = minutes_per_locale * 60.0

    rows_all: list[dict] = []

    for loc in locales:
        print(f"\n=== {loc} ===")
        loc_out = out / loc
        loc_out.mkdir(parents=True, exist_ok=True)

        # TSVs (small)
        repo_test, tsv_test = _download_only_tsv(repo_id=repo_id, loc=loc, split_name="test",      cache_key=f"cv17_{loc}_test")
        repo_val,  tsv_val  = _download_only_tsv(repo_id=repo_id, loc=loc, split_name="validated", cache_key=f"cv17_{loc}_validated")
        repo_trn,  tsv_trn  = _download_only_tsv(repo_id=repo_id, loc=loc, split_name="train",     cache_key=f"cv17_{loc}_train")

        meta_idx = _build_meta_index([tsv_test, tsv_val, tsv_trn])

        # Count existing files
        already: set[str] = set()
        acc_sec = 0.0
        for p in loc_out.glob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                base = p.name
                already.add(base)
                dur = (meta_idx.get(base, {}).get("duration") or _ffprobe_duration_seconds(p) or 0.0)
                acc_sec += float(dur)
        print(f"   already on disk: ~{int(acc_sec)}s across {len(already)} files")

        # 1) TEST (one tar at a time)
        if "test" in splits and acc_sec < target_sec and tsv_test.exists():
            for i in range(0, 3):  # try a few shards
                if acc_sec >= target_sec:
                    break
                tar = _download_split_tar(repo_id=repo_id, repo_dir=repo_test, loc=loc, split_name="test", tar_index=i)
                if not tar:
                    if i == 0:
                        print("   ! No test tar found.")
                    break
                keep, gained = _extract_from_tars_scanning(
                    tars=[tar], out_dir=loc_out, target_remaining=target_sec - acc_sec,
                    already=already, meta_idx=meta_idx, min_sec=min_duration_sec, max_sec=max_duration_sec
                )
                rows_all.extend([{**r, "language": loc} for r in keep])
                acc_sec += gained
            print(f"   test: +{int(max(0, acc_sec))}s (total ~{int(acc_sec)}s)")

        # 2) VALIDATED (one tar at a time)
        if "validated" in splits and acc_sec < target_sec and tsv_val.exists():
            for i in range(0, 6):  # validated can have several shards
                if acc_sec >= target_sec:
                    break
                tar = _download_split_tar(repo_id=repo_id, repo_dir=repo_val, loc=loc, split_name="validated", tar_index=i)
                if not tar:
                    if i == 0:
                        print("   ! No validated tar found.")
                    break
                keep, gained = _extract_from_tars_scanning(
                    tars=[tar], out_dir=loc_out, target_remaining=target_sec - acc_sec,
                    already=already, meta_idx=meta_idx, min_sec=min_duration_sec, max_sec=max_duration_sec
                )
                rows_all.extend([{**r, "language": loc} for r in keep])
                acc_sec += gained
            print(f"   validated: total ~{int(acc_sec)}s")

        # 3) TRAIN (one tar at a time)
        if "train" in splits and acc_sec < target_sec and tsv_trn.exists():
            for i in range(0, 6):  # try a handful; stop as soon as we hit target
                if acc_sec >= target_sec:
                    break
                tar = _download_train_tar(repo_id=repo_id, repo_dir=repo_trn, loc=loc, tar_index=i)
                if not tar:
                    if i == 0:
                        print("   ! No train tar found.")
                    break
                keep, gained = _extract_from_tars_scanning(
                    tars=[tar], out_dir=loc_out, target_remaining=target_sec - acc_sec,
                    already=already, meta_idx=meta_idx, min_sec=min_duration_sec, max_sec=max_duration_sec
                )
                rows_all.extend([{**r, "language": loc} for r in keep])
                acc_sec += gained
            print(f"   train: total ~{int(acc_sec)}s")

        if acc_sec < target_sec:
            print(f"   ⚠ Collected only ~{int(acc_sec)}s for {loc} (target {int(target_sec)}s).")

        # Always include EVERYTHING on disk for this locale (existing + new)
        rows_all.extend(_scan_folder_to_rows(
            loc=loc, loc_out=loc_out, meta_idx=meta_idx,
            min_sec=min_duration_sec, max_sec=max_duration_sec
        ))

    # De-dupe by path and return
    df = pd.DataFrame(rows_all)
    if not df.empty:
        df = df.drop_duplicates(subset=["path"])
    return df
