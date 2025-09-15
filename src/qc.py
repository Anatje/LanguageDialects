import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def stats_and_checks(manifest: pd.DataFrame):
    """
    Print basic dataset stats and run sanity checks.

    - Prints total samples, counts by dialect, and average duration by dialect.
    - Asserts that 'language' is non-null for all rows.
    - If 'split' exists, asserts it contains only {'train','val','test'} (no NaNs).
    """
    print("Total samples:", len(manifest))
    print("By dialect:")
    print(manifest["dialect"].value_counts(dropna=False))
    print("\nAvg duration (s) by dialect:")
    print(manifest.groupby("dialect")["duration"].mean().round(2))

    # Mandatory label present
    assert manifest["language"].notna().all()

    # No NaNs in split for final manifest; must be one of the expected values
    if "split" in manifest.columns:
        assert manifest["split"].isin(["train", "val", "test"]).all()

def duration_histogram(manifest: pd.DataFrame, out_png: str):
    """
    Save a histogram of 'duration' to out_png (30 bins).
    Creates parent directories as needed.
    """
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    manifest["duration"].plot(kind="hist", bins=30)
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Clip Duration Histogram")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
