# src/manifest.py
import re
import pandas as pd
from typing import Dict, List, Any

def normalize_dialect(df: pd.DataFrame, norm_cfg: Dict[str, Dict[str, List[str] | str]]) -> pd.DataFrame:
    """
    Map free-text 'accent_raw' to a canonical 'dialect' per language using the provided mapping.

    Behavior:
    - Coerces 'accent_raw' to clean lowercase strings (handles NaN/None/float).
    - Matches exact tokens or token-boundary occurrences (spaces, hyphens, underscores, slashes).
    - If no mapping is found, sets 'dialect' to the row's 'language'.

    norm_cfg format example:
        {
          "pt": {"br": ["brasil", "brasileiro"], "pt": ["portugal", "europeu"]},
          "es": {"mx": ["mexico", "mexicano"], "es": ["españa", "castellano"]}
        }
    """
    out = df.copy()

    # Ensure 'accent_raw' exists and is a clean lowercase string
    if "accent_raw" not in out.columns:
        out["accent_raw"] = ""
    out["accent_raw"] = (
        out["accent_raw"]
        .astype(str)               # turns NaN/None/floats into strings
        .replace({"nan": "", "None": ""})
        .fillna("")
        .str.strip()
        .str.lower()
    )

    # Build a normalized mapping (lowercased variants)
    norm_cfg_norm: Dict[str, Dict[str, List[str]]] = {}
    for lang, rules in (norm_cfg or {}).items():
        lang_rules: Dict[str, List[str]] = {}
        for canon, variants in (rules or {}).items():
            if isinstance(variants, str):
                variants = [variants]
            lang_rules[str(canon)] = [(v or "").strip().lower() for v in variants if v]
        norm_cfg_norm[str(lang)] = lang_rules

    # Token boundary pattern: start/end or separators: space, hyphen, underscore, slash
    def matches_variant(text: str, v: str) -> bool:
        if not v:
            return False
        if text == v:
            return True
        return re.search(rf"(^|[\s_\-\/]){re.escape(v)}($|[\s_\-\/])", text) is not None

    def pick_dialect(lang: str, accent_raw: str):
        rules = norm_cfg_norm.get(lang, {})
        if not rules or not accent_raw:
            return None
        for canon, variants in rules.items():
            for v in variants:
                if matches_variant(accent_raw, v):
                    return canon
        return None

    out["dialect"] = out.apply(
        lambda r: pick_dialect(str(r.get("language", "")), str(r.get("accent_raw", ""))),
        axis=1,
    )
    # Fallback: if we couldn't map, use the language as the dialect
    out["dialect"] = out["dialect"].where(out["dialect"].notna(), out["language"])
    return out

def filter_by_duration(df: pd.DataFrame, min_s: float, max_s: float) -> pd.DataFrame:
    """Return only rows where 'duration' is within [min_s, max_s], inclusive."""
    return df[df["duration"].between(min_s, max_s, inclusive="both")].copy()

def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing mandatory fields 'path' or 'language'."""
    return df.dropna(subset=["path", "language"]).reset_index(drop=True)

def make_manifest(df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    """
    Export a minimal manifest CSV with required columns.

    Expects the following columns to exist in df:
    - path, language, dialect, speaker_id, duration
    """
    cols = ["path", "language", "dialect", "speaker_id", "duration"]
    m = df[cols].dropna(subset=["path", "language"]).copy()
    m.to_csv(output_csv, index=False)
    return m

