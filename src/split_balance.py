from __future__ import annotations
import random
import pandas as pd
from typing import List, Optional, Sequence, Dict, Tuple

def _as_tuple_label(row, cols: Sequence[str]) -> Tuple:
    """Return a hashable tuple label from selected columns."""
    return tuple(row[c] for c in cols)

def stratified_group_split(
    df: pd.DataFrame,
    *,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratify_on: Optional[List[str]] = None,   # e.g. ["language"] or ["language","dialect"]
    group_by: str = "speaker_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Group-aware, approximately stratified split.

    Behavior
    - Each group (e.g., speaker_id) is assigned to exactly one split.
    - Approximate stratification over labels in `stratify_on`.
    - Fractions are renormalized if they do not sum to 1.0.
    """
    assert 0 < train_size < 1 and 0 <= val_size < 1 and 0 <= test_size < 1
    total = train_size + val_size + test_size
    if round(total, 6) != 1.0:
        train_size, val_size, test_size = [x / total for x in (train_size, val_size, test_size)]

    if not stratify_on:
        stratify_on = ["language"]

    out = df.copy()

    # Ensure a usable group key: fill missing groups with unique ids
    if group_by not in out.columns:
        out[group_by] = [f"grp_{i}" for i in range(len(out))]
    else:
        # Fill NaNs deterministically with a unique id per row
        if out[group_by].isna().any():
            out[group_by] = out[group_by].astype(object)
            out.loc[out[group_by].isna(), group_by] = [
                f"grp_{i}" for i in out.index[out[group_by].isna()]
            ]

    # Build one-row-per-group frame:
    grp = out.groupby(group_by, dropna=False)
    # first() for strat labels
    g_first = grp[stratify_on].first()
    # size() for row counts
    g = g_first.copy()
    g["__rows__"] = grp.size()
    g = g.reset_index()
    g["__label__"] = g.apply(lambda r: _as_tuple_label(r, stratify_on), axis=1)

    rnd = random.Random(random_state)
    assign: Dict[str, str] = {}

    for _, sub in g.groupby("__label__"):
        grp_ids = list(sub[group_by])
        rnd.shuffle(grp_ids)
        n = len(grp_ids)
        n_test = round(n * test_size)
        n_val  = round(n * val_size)
        # Don’t exceed n
        if n_test + n_val > n:
            over = n_test + n_val - n
            n_val = max(0, n_val - over)

        ids_test  = grp_ids[:n_test]
        ids_val   = grp_ids[n_test:n_test + n_val]
        ids_train = grp_ids[n_test + n_val:]

        for gid in ids_test:  assign[gid] = "test"
        for gid in ids_val:   assign[gid] = "val"
        for gid in ids_train: assign[gid] = "train"

    out["split"] = out[group_by].map(assign).fillna("train")
    return out

def balance_by_minutes(
    df_train: pd.DataFrame,
    target_minutes_per_dialect: float = 5.0,
    *,
    label_cols: Sequence[str] = ("language","dialect"),
    duration_col: str = "duration",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Subsample rows per (language, dialect) until approximately
    `target_minutes_per_dialect` of audio is accumulated.

    Note: sampling order is randomized per label (seeded).
    """
    import random as _random
    rnd = _random.Random(random_state)
    kept = []
    for label, sub in df_train.groupby(list(label_cols)):
        sub = sub.sample(frac=1.0, random_state=rnd.randint(0, 10_000))
        target_sec = target_minutes_per_dialect * 60.0
        acc = 0.0
        for _, r in sub.iterrows():
            kept.append(r)
            acc += float(r.get(duration_col, 0.0) or 0.0)
            if acc >= target_sec:
                break
    return pd.DataFrame(kept).reset_index(drop=True) if kept else df_train.iloc[0:0].copy()
