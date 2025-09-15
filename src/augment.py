from pathlib import Path
import os
import random
import shutil

# Resolve ffmpeg/ffprobe first and expose them via PATH/env 
from .audio_utils import _resolve_ff_tools
_FFMPEG, _FFPROBE = _resolve_ff_tools()
_ffdir = os.path.dirname(_FFMPEG)

# Ensure old pydub can find ffprobe via PATH 
os.environ["PATH"] = _ffdir + os.pathsep + os.environ.get("PATH", "")

os.environ["FFMPEG_BINARY"]  = _FFMPEG
os.environ["FFPROBE_BINARY"] = _FFPROBE

# Import pydub only after PATH/env are set (prevents warnings & WinError 2
# (LLM assisted here, as setting the path the usual way threw winerror multiple times))
from pydub import AudioSegment
from pydub.generators import WhiteNoise

AudioSegment.converter = _FFMPEG
# Older pydub may not have AudioSegment.ffprobe; that's OK
try:
    AudioSegment.ffprobe = _FFPROBE
except Exception:
    pass

# Try torch/torchaudio, but never crash if they fail to import or load
_HAVE_TORCHAUDIO = True
_torch_import_error = None
try:
    import torch
    import torchaudio as ta
except Exception as e:
    _HAVE_TORCHAUDIO = False
    _torch_import_error = e  # for optional logging

# ---------------------------
# Torch / torchaudio pipeline
# ---------------------------

def _add_noise_torch(wav: "torch.Tensor", snr_db: float):
    # wav: [C, N], float or float-like tensor
    sig_power = wav.pow(2).mean()
    # Guard against complete silence
    if float(sig_power) <= 0.0:
        return wav
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise

def _speed_perturb_torch(wav: "torch.Tensor", sr: int, factor: float):
    new_sr = max(8000, int(sr * factor))
    wav2 = ta.functional.resample(wav, sr, new_sr)
    return wav2, new_sr

def _augment_with_torchaudio(in_path: str, out_path: str, speed_factor: float, snr_db: float):
    wav, sr = ta.load(in_path)  # [C, N]
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    wav, sr2 = _speed_perturb_torch(wav, sr, speed_factor)
    wav = _add_noise_torch(wav, snr_db)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ta.save(out_path, wav, sr2)

# ---------------------------
# pydub / ffmpeg fallback
# ---------------------------

def _speed_perturb_pydub(seg: AudioSegment, factor: float) -> AudioSegment:
    # Frame-rate trick to change speed, then restore nominal rate
    factor = max(0.5, min(2.0, float(factor)))
    changed = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * factor)})
    return changed.set_frame_rate(seg.frame_rate)

def _add_noise_pydub(seg: AudioSegment, snr_db: float) -> AudioSegment:
    # Target noise loudness so that SNR_dB = seg_dBFS - noise_dBFS
    # If seg.dBFS is None (e.g., silence), default to -30 dBFS baseline
    sig_dbfs = seg.dBFS if seg.dBFS is not None and seg.dBFS != float("-inf") else -30.0
    noise_target_dbfs = sig_dbfs - float(snr_db)

    noise = WhiteNoise().to_audio_segment(duration=len(seg), volume=noise_target_dbfs)
    # Match format to input
    noise = noise.set_frame_rate(seg.frame_rate).set_channels(seg.channels).set_sample_width(seg.sample_width)

    # Overlay adds signals; no clipping protection—keep SNR conservative
    return seg.overlay(noise)

def _augment_with_pydub(in_path: str, out_path: str, speed_factor: float, snr_db: float):
    seg = AudioSegment.from_file(in_path)
    seg = seg.set_channels(1)  # mono to mirror torch path
    seg = _speed_perturb_pydub(seg, speed_factor)
    seg = _add_noise_pydub(seg, snr_db)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(out_path).suffix.lstrip(".").lower() or "wav"
    seg.export(out_path, format=ext)

# ---------------------------
# Public API
# ---------------------------

def augment_file(in_path: str, out_path: str, speed_factor: float, snr_db: float):
    """
    Augments an audio file with speed perturbation and additive white noise.

    Behavior:
      - If torchaudio loads and works, use it (best quality resampling).
      - If torchaudio fails to import or errors at runtime, fall back to pydub+ffmpeg.
    """
    if _HAVE_TORCHAUDIO:
        try:
            _augment_with_torchaudio(in_path, out_path, speed_factor, snr_db)
            return out_path
        except Exception:
            _augment_with_pydub(in_path, out_path, speed_factor, snr_db)
            return out_path
    else:
        _augment_with_pydub(in_path, out_path, speed_factor, snr_db)
        return out_path
