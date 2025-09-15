"""
Utilities for audio conversion and duration probing using FFmpeg/ffprobe.

- Conversion: arbitrary input -> mono, 16-bit PCM WAV at a target sample rate.
- Duration: fast probe via ffprobe (format duration first, then stream fallback).

Environment
- Optionally set FFMPEG_PATH and FFPROBE_PATH to absolute executables.
- On Windows, WinGet-based fallbacks are attempted if not on PATH.
"""

from __future__ import annotations
import os, shutil, subprocess
from pathlib import Path
from typing import Optional

AUDIO_SAMPLE_WIDTH = 2  # 16-bit PCM

# ---------- resolve ffmpeg/ffprobe ----------
def _win_ffmpeg_candidates() -> list[str]:
    """Common WinGet install locations for ffmpeg.exe (best-effort heuristics)."""
    home = Path.home()
    base = home / r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    return [
        str(base / r"ffmpeg-8.0-full_build\bin\ffmpeg.exe"),
        str(home / r"AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"),
    ]

def _win_ffprobe_candidates() -> list[str]:
    """Common WinGet install locations for ffprobe.exe (best-effort heuristics)."""
    home = Path.home()
    base = home / r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    return [
        str(base / r"ffmpeg-8.0-full_build\bin\ffprobe.exe"),
        str(home / r"AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"),
    ]

def _find_exe(name: str, env_key: str, fallbacks: list[str]) -> Optional[str]:
    """
    Resolve an executable path by:
    1) environment variable (env_key), 2) PATH via shutil.which, 3) known fallbacks.
    Returns the first existing path or None.
    """
    p = os.environ.get(env_key)
    if p and Path(p).exists():
        return p
    p = shutil.which(name)
    if p:
        return p
    for cand in fallbacks:
        if Path(cand).exists():
            return cand
    return None

def _resolve_ff_tools() -> tuple[str, str]:
    """Return (ffmpeg, ffprobe) absolute paths or raise if not found."""
    ffmpeg = _find_exe("ffmpeg", "FFMPEG_PATH", _win_ffmpeg_candidates())
    ffprobe = _find_exe("ffprobe", "FFPROBE_PATH", _win_ffprobe_candidates())
    if not ffmpeg or not ffprobe:
        raise RuntimeError(
            "FFmpeg/ffprobe not found. Set FFMPEG_PATH and FFPROBE_PATH env vars to the .exe files."
        )
    return ffmpeg, ffprobe

def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and raise with a helpful tail of STDERR on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n" + " ".join(cmd) + "\n\nSTDERR (tail):\n" + p.stderr[-2000:]
        )
    return p

# ---------- public API ----------
def convert_to_wav16k_mono(in_path: str, out_path: str, sample_rate: int | None = None, target_sr: int | None = None) -> None:
    """
    Convert input audio to mono 16-bit PCM WAV at the requested sample rate (default 16000),
    using FFmpeg directly (no pydub/torchaudio). Creates parent directory for out_path.
    """
    sr = sample_rate or target_sr or 16000
    ffmpeg, _ = _resolve_ff_tools()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", in_path,
        "-vn",          # strip any video
        "-ac", "1",     # mono
        "-ar", str(sr), # sample rate
        "-sample_fmt", "s16",  # 16-bit PCM
        out_path,
    ]
    _run(cmd)

def recompute_duration_seconds(path: str) -> float:
    """
    Return duration (seconds) via ffprobe.
    - Tries container/format duration first; if unavailable, falls back to first audio stream.
    - Returns 0.0 on failure. Values are rounded to milliseconds (3 decimals).
    """
    _, ffprobe = _resolve_ff_tools()
    # Try container duration first
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path,
    ]
    try:
        p = _run(cmd)
        d = float(p.stdout.strip())
        if d > 0:
            return round(d, 3)
    except Exception:
        pass
    # Fallback to stream duration
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path,
    ]
    try:
        p = _run(cmd)
        d = float(p.stdout.strip())
        return round(max(d, 0.0), 3)
    except Exception:
        return 0.0
