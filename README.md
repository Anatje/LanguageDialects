# LanguageDialects

Small pipeline to pull cca 5 minutes of speech per locale from Common Voice v17, build a clean manifest, convert audio to WAV/16k/mono, split and balance it, run QC, and (optionally) augment. No PyTorch required for the data steps.

---

## What this repo does

- Fetches just-enough audio per locale (for example ~5 minutes) from Common Voice 17.0 using a minimal snapshot fetcher (no `datasets`, no `torch`).
- Writes a raw manifest with path, language, accent_raw, speaker_id, duration.
- Converts audio to WAV/16k/mono via ffmpeg.
- Splits with group awareness (each speaker appears in exactly one split).
- Balances training minutes per (language, dialect).
- Runs lightweight QC (stats + duration histogram).
- Optional augmentation (speed + noise).

---

## Requirements

- OS: Windows 10/11. (Linux/macOS also fine, but paths below are Win).
- Python: 3.10–3.12 (tested on 3.11).
- ffmpeg + ffprobe installed and reachable.
- Hugging Face account with a Read token (for authenticated dataset access).

No GPU / CUDA / torch needed for data pipeline steps.

---

## Quickstart (Windows PowerShell shown)

```powershell
# 0) Create and activate a virtual environment
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip

# 1) Install dependencies
pip install -r requirements.txt
#pip install -r requirements-augment.txt if you enable augmentation in config and wish to run torch. 

# 2) Hugging Face auth (use a Read token)
python - << 'PY'
from huggingface_hub import login
import getpass
print("Paste your HF token (Read):")
login(token=getpass.getpass(''), add_to_git_credential=True)
PY

# 3) Ensure ffmpeg/ffprobe are reachable (set env vars if needed)
$env:FFMPEG_PATH  = "C:\Users\YOURUSER\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
$env:FFPROBE_PATH = "C:\Users\YOURUSER\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffprobe.exe"
# Optional persistence for future shells:
# setx FFMPEG_PATH  "<path to ffmpeg.exe>"
# setx FFPROBE_PATH "<path to ffprobe.exe>"

# 4) Adjust config/config.yaml (example below), then run:
python -m src.cli download --config config/config.yaml
python -m src.cli manifest --config config/config.yaml
python -m src.cli convert  --config config/config.yaml
python -m src.cli split    --config config/config.yaml
python -m src.cli balance  --config config/config.yaml
python -m src.cli qc       --config config/config.yaml
# optional, requires setting augmentation enabled to True, default is False :
# python -m src.cli augment --config config/config.yaml

#to see the app, run: 
streamlit run apps/dialect_browser.py 
```

Linux/macOS users can export the same env vars and run the identical commands.

---

Notes:
- `target_minutes_per_dialect` steers the balance step for training rows.
- The fetcher tops up across splits (test → validation → train) until the target minutes are met or data runs out in bounds.
- Durations are filtered to `[min_duration_sec, max_duration_sec]` when building manifests.

---

## Commands and outputs

| Command  | What it does | Primary output(s) |
|----------|---------------|-------------------|
| download | Pulls TSVs and only the minimal audio tar(s) needed to reach target minutes per locale. | data/raw/cv17/<locale>/*.mp3, data/manifests/common_voice_raw.csv |
| manifest | Normalizes dialect labels (or defaults to language), filters durations, writes a clean manifest. | data/manifests/manifest_raw.csv |
| convert  | Converts to WAV/16k/mono and recomputes durations. | data/processed/audio/... + manifest_processed.csv |
| split    | Group-aware, stratified split; each speaker in one split. | manifest_split.csv |
| balance  | Downsamples train to ~target minutes per (language, dialect). | manifest_balanced.csv |
| qc       | Prints stats; writes a duration histogram. | processed/figures/duration_hist.png |
| augment  | (Optional) Adds speed/noise variants for train and appends to manifest. | manifest_augmented.csv |

---

## Sanity checks

PowerShell one-liners tht can run immediately:

```powershell
# Minutes per language/dialect/split in the balanced manifest
@'
import pandas as pd
m = pd.read_csv("data/manifests/manifest_balanced.csv")
print(m.groupby(["language","dialect","split"])["duration"].sum().div(60).round(2).to_string())
'@ | python -

# Spot-check existence and audio integrity (first 5 files)
@'
import pandas as pd, os, soundfile as sf
m = pd.read_csv("data/manifests/manifest_processed.csv").head(5)
for p in m["path"]:
    assert os.path.exists(p), f"Missing: {p}"
    y, sr = sf.read(p)
    assert y.ndim == 1 and sr == 16000, (p, y.shape, sr)
print("OK: files exist and are mono 16k.")
'@ | python -

# Verify each speaker_id appears in exactly one split
@'
import pandas as pd
m = pd.read_csv("data/manifests/manifest_split.csv")
dup = m.groupby("speaker_id")["split"].nunique().gt(1).sum()
print("Speakers in multiple splits:", dup)
'@ | python -
```

---

## How the fast fetch works

- Uses `huggingface_hub.snapshot_download` to pull only:
  - the `transcript/<locale>/<split>.tsv` files to decide what’s eligible, and
  - the smallest `audio/<locale>/<split>/...` tar(s) needed to reach your target minutes.
- Extracts only the MP3 files referenced in TSVs; no decoding occurs during fetch.
- Resumable: reruns pick up where they left off.
- Cache: default Hugging Face cache under `~/.cache/huggingface`; extracted audio goes to `data/raw/cv17`.

---

## Dialects, accents, and labels

- Common Voice often lacks accent labels. Default behavior sets `dialect = language` to avoid dropping rows.
- If you need sub-dialects, add keyword mappings under `dialect_normalization`. Rows matching those keywords are labeled; others remain at the language level.

Why underrepresented locales are suitable here:
- Big-language shards are large; fetching a few minutes can still pull multi-GB tar files. This pipeline emphasizes reproducible data handling over heavy downloads.
- Smaller locales let me hit targets quickly and focus on manifests, consistent audio format, group-aware splitting, balancing, QC, and inspection.
- The pipeline is config-driven: you can switch locales or add dialect rules and just rerun.

If you want big-language dialects:
1. Prioritize smaller splits (validation/test) before train.
2. Tighten duration bounds (for example, 2–10 s).
3. Add accent keyword mappings to label first; top up with unlabeled rows only if needed.
4. Rely on cache and resume to avoid refetching.

---

## Troubleshooting

Pydub warns “Couldn’t find ffmpeg/ffprobe”  
- Set `FFMPEG_PATH` and `FFPROBE_PATH` as shown above. The pipeline calls ffmpeg directly, so the warning is harmless.

ffprobe or ffmpeg cannot launch  
- Usually a wrong path. Confirm both print version banners.

Hugging Face unauthorized  
- Log in once with a Read token or set `HF_TOKEN` for headless runs.

Slow downloads or timeouts  
- Network-bound. Reruns resume; cache prevents refetching already-downloaded files.

Datasets or torch issues  
- Not used by the fast path; the snapshot fetcher avoids those dependencies.

Windows HF cache symlink warnings  
- Harmless; it falls back automatically.

---

## Tips

- To try new locales: edit `config.yaml` and rerun from `download`. Existing files are reused.
- To target more minutes: increase `target_minutes_per_dialect` and rerun `download → manifest → convert → split → balance → qc`.
- To start clean for a single locale: remove that locale’s folders under `data/raw/cv17/<locale>` and `data/processed/audio/<locale>`.

---

## License and attribution

- Respect the license terms of the Common Voice dataset and each language pack.
- Data is fetched via Hugging Face; ensure you comply with dataset licenses and attribution requirements.

---

## FAQ

**Why these specific languages and not major languages and their dialects?** 
I intentionally chose smaller locales (e.g., bas, fy-NL, kmr, kk, ba, gl, ug, ha, gn, hsb) instead of major languages (like English/Spanish) because:
•	Bandwidth & time realism: Big-language shards (e.g., Spanish train) are multi-GB and slow to fetch even if you only need a few minutes. For a demo-scale deliverable, that’s a lot of time spent waiting on downloads instead of building a reliable, reproducible pipeline. Furthermore, frequent timeouts (even with changing the retry and timeout parameters).
•	Balanced coverage quickly: Smaller locales allow to the 5-minute target per “dialect” fast, effort can be spent in the things the brief emphasizes: clean manifests, consistent audio format, group-aware splitting, per-dialect balancing, QC, and a Streamlit browser. 
•	Reusability > Volume: If you want classic dialect families (e.g., Spanish Andean vs. Rioplatense), you can switch the config.yaml locales/dialect rules and re-run. The pipeline already handles top-ups, duration filters, format conversion, stratified splits, and balancing. 

**Why dialect defaults to language?** 
•	Common Voice accent sparsity: In many locales, the accent field is empty. Rather than drop those rows, I use a safe fallback: dialect = language when accent is missing. That keeps coverage stable and meets the “clearly labeled with language and dialect” requirement—with a transparent rule that avoids fabricating labels. (When accent text exists, it can be normalized via dialect_normalization.) 
•	Small duration window by design: Clips are filtered to 1–15 s to keep examples consistent and useful for classification. If a shard has lots of ultra-short or too-long clips, they’re excluded by rule. That can make a locale run short in a split; the fetcher tops up across splits (validated/test/train) until ~5 minutes are reached, or reports it’s short if there simply isn’t enough eligible audio.
•	Exactly-once extraction: The snapshot fetcher maps each transcript entry to a filename and extracts only those MP3s needed from the tar, then de-duplicates by path. That can look like “+0s” for some tar files if none of the remaining wanted files are inside that particular tar. This behaviour is expected.


**Why not use `datasets.load_dataset`?**  
It often pulls large shards and may implicate torch on streaming. The snapshot approach is lighter for small, targeted pulls.

**Why WAV/16k/mono?**  
A common baseline for ASR/SSL models; keeps processing consistent across sources.

**Do I need augmentation?**  
Not for a quick baseline. If enabled, simple speed/noise variants are added to training rows and appended to the manifest.


**Llm disclosure statement**  
I used an LLM to help draft/iterate on:
•	ffmpeg-based conversion utilities and environment hooks in src/audio_utils.py and src/augment.py as explicitly setting path code did not work as intended and created many win2 errors.
•	The Streamlit browser (apps/dialect_browser.py) as I am not familiar with Streamlit and
•	 organizing documentation (this README).
All code was run locally, debugged, and adjusted to my machine (Windows + PowerShell).
•	Streamlit app: Reads manifest, filters by language/dialect, plays audio, and renders waveform + mel spectrogram without torch/torchaudio. Mel spectrogram uses mel scale which is logarithmic and mimics the way human ear perceives sound. Implemented through librosa. Also added the waveform plot, as average users are more familiar with that signal representation. 


**Known limitations.**

•	Accent labels are often missing; for this submission, dialect == language for most rows.

•	Some locales may not have enough clips in the 1–15 s window; the fetcher reports shortfalls.

•	Network speed limits shard fetches; even “minimal” pulls can be hundreds of MB.

•	Windows-specific paths are included; Linux/macOS users should set FFMPEG_PATH/FFPROBE_PATH accordingly.
