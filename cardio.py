"""Cardiovascular disease prediction from routine examination data.

Dataset: cardio.csv, 70,000 patients, 11 features collected at a medical examination and a binary
target (cardio: presence of cardiovascular disease).

This module holds everything the report (report.ipynb) relies on, so the notebook and a plain run of
this file always use the same code:

    python cardio.py clean      data-quality audit and the three cleaning strategies, side by side

Cleaning works one row at a time with fixed rules (no statistic is learned from the data), so it can
be applied before or after the train/test split with identical results.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cardio.csv"
OUTPUTS = ROOT / "outputs"
SEED = 42
TEST_SIZE = 0.2
TARGET = "cardio"

# Plausibility bounds for adults at a routine examination. They are deliberately wide: they remove
# values that cannot be measurements of a living adult, not patients who are merely unusual.
BOUNDS = {
    "ap_hi": (70, 250),   # systolic blood pressure, mmHg
    "ap_lo": (40, 150),   # diastolic blood pressure, mmHg
    "height": (120, 220),  # cm
    "weight": (30, 200),   # kg
}
BMI_BOUNDS = (12, 70)

STRATEGIES = ("raw", "drop", "repair")
STRATEGY_TEXT = {
    "raw": "no cleaning: exact duplicates removed, every value kept as recorded",
    "drop": "remove every record with an implausible value",
    "repair": "fix blood pressures with a recognisable entry error, then remove what is still implausible",
}

# Features as the models will see them, after prepare() and add_features().
BASE_FEATURES = ["age_years", "male", "height", "weight", "ap_hi", "ap_lo",
                 "cholesterol", "gluc", "smoke", "alco", "active"]
DERIVED_FEATURES = ["bmi", "pulse_pressure", "mean_arterial_pressure", "bp_stage"]
FEATURES = BASE_FEATURES + DERIVED_FEATURES

# ACC/AHA 2017 blood-pressure categories (0 normal, 1 elevated, 2 stage 1, 3 stage 2)
BP_STAGE_NAMES = ["normal", "elevated", "stage 1", "stage 2"]


# ----------------------------------------------------------------------------------------- loading

def load(path: Path | str = DATA) -> pd.DataFrame:
    """The dataset exactly as distributed."""
    return pd.read_csv(path)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate records (identical in every column except id), keeping the first.

    A duplicate pair split across train and test would let a model be tested on a patient it was
    trained on. Records with identical features but different diagnoses are kept: they are
    contradictory, not duplicated, and removing them would make the task look easier than it is.
    """
    return df[~df.drop(columns="id").duplicated()].reset_index(drop=True)


# ----------------------------------------------------------------------------------- data quality

def bmi(df: pd.DataFrame) -> pd.Series:
    return df["weight"] / (df["height"] / 100) ** 2


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per plausibility check, True where the record fails it."""
    checks = {f"{col} out of range": ~df[col].between(lo, hi) for col, (lo, hi) in BOUNDS.items()}
    checks["diastolic >= systolic"] = df["ap_lo"] >= df["ap_hi"]
    checks["BMI out of range"] = ~bmi(df).between(*BMI_BOUNDS)
    return pd.DataFrame(checks, index=df.index)


def repair_pressure(value: float, lo: float, hi: float) -> tuple[float, str | None]:
    """The reading a recorded blood pressure most plausibly was, and the rule that recovered it.

    Returns (nan, None) when no rule gives a plausible value. Rules, in order:
      sign          a negative reading is the same reading with a stray minus (-120 -> 120)
      cmHg          recorded in cmHg instead of mmHg (12 -> 120), only if that lands in range
      extra digits  trailing digits typed by mistake (1000 -> 100, 14020 -> 140, 906 -> 90)
    """
    if pd.isna(value):
        return math.nan, None
    rule = None
    if value < 0:
        value, rule = -value, "sign"
    if lo <= value <= hi:
        return value, rule
    if 0 < value < lo:
        return (value * 10, "cmHg") if lo <= value * 10 <= hi else (math.nan, None)
    if value > hi:
        value = int(value)
        while value > hi:
            value //= 10
        return (value, "extra digits") if value >= lo else (math.nan, None)
    return math.nan, None  # zero


def repair(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blood pressures with a recognisable entry error, corrected; plus a log of every change.

    Height and weight are not repaired: their errors show no pattern that identifies the true value.
    A value no rule can recover becomes NaN, so the record fails audit() and is dropped afterwards.
    """
    out = df.copy()
    out[["ap_hi", "ap_lo"]] = out[["ap_hi", "ap_lo"]].astype(float)  # room for NaN
    log = []
    for col in ("ap_hi", "ap_lo"):
        lo, hi = BOUNDS[col]
        bad = ~out[col].between(lo, hi)
        for idx in out.index[bad]:
            before = out.at[idx, col]
            after, rule = repair_pressure(before, lo, hi)
            out.at[idx, col] = after
            log.append({"id": out.at[idx, "id"], "field": col, "before": f"{before:g}",
                        "after": "" if math.isnan(after) else f"{after:g}",
                        "rule": rule or "unrecoverable"})
    # systolic and diastolic typed into each other's field
    swapped = (out["ap_lo"] > out["ap_hi"]) & out["ap_hi"].between(*BOUNDS["ap_lo"]) \
        & out["ap_lo"].between(*BOUNDS["ap_hi"])
    for idx in out.index[swapped]:
        hi_, lo_ = out.at[idx, "ap_hi"], out.at[idx, "ap_lo"]
        log.append({"id": out.at[idx, "id"], "field": "ap_hi/ap_lo", "before": f"{hi_:g}/{lo_:g}",
                    "after": f"{lo_:g}/{hi_:g}", "rule": "swapped"})
    out.loc[swapped, ["ap_hi", "ap_lo"]] = out.loc[swapped, ["ap_lo", "ap_hi"]].to_numpy()
    return out, pd.DataFrame(log, columns=["id", "field", "before", "after", "rule"])


def clean(df: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicated records cleaned with one strategy, and the repair log (empty unless 'repair').

    'raw' keeps every record as recorded (after deduplication), to measure what cleaning is worth.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
    df = deduplicate(df)
    log = pd.DataFrame(columns=["id", "field", "before", "after", "rule"])
    if strategy == "raw":
        return df, log
    if strategy == "repair":
        df, log = repair(df)
    keep = ~audit(df).any(axis=1)
    out = df[keep].reset_index(drop=True)
    out[["ap_hi", "ap_lo"]] = out[["ap_hi", "ap_lo"]].astype(int)
    return out, log


def cleaning_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Records kept, repaired and removed under each strategy, starting from the raw file."""
    deduped = deduplicate(df)
    rows = []
    for s in STRATEGIES:
        out, log = clean(df, s)
        repaired = log.loc[log["rule"] != "unrecoverable", "id"]
        rows.append({
            "strategy": s,
            "records": len(out),
            "duplicates removed": len(df) - len(deduped),
            # repaired and then kept: a record with a second, unrecoverable error is still removed
            "records repaired": repaired[repaired.isin(out["id"])].nunique(),
            "implausible removed": len(deduped) - len(out),
            "share kept": len(out) / len(df),
            "description": STRATEGY_TEXT[s],
        })
    return pd.DataFrame(rows).set_index("strategy")


# --------------------------------------------------------------------------------------- features

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Readable units and codes: age in years, sex as male = 1.

    The documentation calls gender 'a categorical code'. Code 2 is male: its mean height is 8.5 cm
    above code 1 (169.9 vs 161.4 cm), the usual adult sex difference.
    """
    out = df.copy()
    out["age_years"] = out["age"] / 365.25
    out["male"] = (out["gender"] == 2).astype(int)
    return out.drop(columns=["age", "gender"])


def bp_stage(systolic: pd.Series, diastolic: pd.Series) -> pd.Series:
    """ACC/AHA 2017 category: the higher of the systolic and diastolic categories."""
    sys_stage = np.select([systolic >= 140, systolic >= 130, systolic >= 120], [3, 2, 1], 0)
    dia_stage = np.select([diastolic >= 90, diastolic >= 80], [3, 2], 0)
    return pd.Series(np.maximum(sys_stage, dia_stage), index=systolic.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clinically standard quantities derived from the recorded values."""
    out = df.copy()
    out["bmi"] = bmi(out)
    out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
    out["mean_arterial_pressure"] = (out["ap_hi"] + 2 * out["ap_lo"]) / 3
    out["bp_stage"] = bp_stage(out["ap_hi"], out["ap_lo"])
    return out


def dataset(strategy: str = "drop", path: Path | str = DATA) -> pd.DataFrame:
    """Load, clean, prepare and derive features: the table every model is trained on."""
    cleaned, _ = clean(load(path), strategy)
    return add_features(prepare(cleaned))


def split_ids(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED) -> tuple[set, set]:
    """Train and test patient ids, drawn once from the deduplicated raw data, stratified by diagnosis.

    Every cleaning strategy is split by these same ids, so all of them are scored on the same patients.
    """
    from sklearn.model_selection import train_test_split

    d = deduplicate(df)
    train, test = train_test_split(d["id"], test_size=test_size, stratify=d[TARGET], random_state=seed)
    return set(train), set(test)


def train_test(strategy: str = "drop", path: Path | str = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The modelling table for one cleaning strategy, divided by the fixed train/test patient ids."""
    train_ids, test_ids = split_ids(load(path))
    data = dataset(strategy, path)
    return (data[data["id"].isin(train_ids)].reset_index(drop=True),
            data[data["id"].isin(test_ids)].reset_index(drop=True))


# --------------------------------------------------------------------------------------------- CLI

def _cmd_clean(args: argparse.Namespace) -> None:
    raw = load(args.data)
    print(f"raw records: {len(raw):,}\n")
    print("== records failing each plausibility check (after deduplication)")
    print(audit(deduplicate(raw)).sum().to_string(), "\n")
    summary = cleaning_summary(raw)
    print("== cleaning strategies")
    print(summary.drop(columns="description").to_string(float_format=lambda x: f"{x:.3f}"), "\n")
    _, log = clean(raw, "repair")
    print("== repairs by rule")
    print(log.groupby(["field", "rule"]).size().to_string(), "\n")
    print("== correlation of systolic pressure with the diagnosis")
    for s in STRATEGIES:
        d, _ = clean(raw, s)
        print(f"   {s:<7} {d['ap_hi'].corr(d[TARGET]):.3f}")
    OUTPUTS.mkdir(exist_ok=True)
    summary.to_csv(OUTPUTS / "cleaning_summary.csv")
    log.to_csv(OUTPUTS / "repair_log.csv", index=False)
    print(f"\nwritten: {OUTPUTS / 'cleaning_summary.csv'}, {OUTPUTS / 'repair_log.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=DATA, help="path to cardio.csv")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("clean", help="data-quality audit and cleaning strategies").set_defaults(func=_cmd_clean)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
