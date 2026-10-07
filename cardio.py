"""Cardiovascular disease prediction from routine examination data.

Dataset: cardio.csv, 70,000 patients, 11 features collected at a medical examination and a binary
target (cardio: presence of cardiovascular disease).

This module holds everything the report (report.ipynb) relies on, so the notebook and a plain run of
this file always use the same code:

    python cardio.py clean      data-quality audit and the three cleaning strategies, side by side
    python cardio.py compare    cross-validated comparison of the clinical baselines and the models

Cleaning works one row at a time with fixed rules (no statistic is learned from the data), so it can
be applied before or after the train/test split with identical results.
"""

from __future__ import annotations

import argparse
import math
import os
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


# ---------------------------------------------------------------------------- clinical baselines

# Framingham general cardiovascular risk, office-based model (BMI instead of cholesterol), for
# patients not treated for hypertension (D'Agostino et al., Circulation 2008):
#     10-year risk = 1 - s0 ** exp(sum(coefficient * term) - mean)
FRAMINGHAM = {
    0: {"ln_age": 2.72107, "ln_bmi": 0.51125, "ln_sbp": 2.81291, "smoke": 0.61868, "diabetes": 0.77763,
        "s0": 0.94833, "mean": 26.0145},   # women
    1: {"ln_age": 3.11296, "ln_bmi": 0.79277, "ln_sbp": 1.85508, "smoke": 0.70953, "diabetes": 0.53160,
        "s0": 0.88431, "mean": 23.9388},   # men
}
SBP_RULE = 130  # mmHg: stage 1 hypertension (ACC/AHA 2017)


def diabetes_proxy(df: pd.DataFrame) -> pd.Series:
    """The dataset has no diabetes field; glucose 'well above normal' is the closest stand-in."""
    return (df["gluc"] == 3).astype(int)


def framingham_terms(df: pd.DataFrame) -> np.ndarray:
    """ln age, ln BMI, ln systolic pressure, smoking and diabetes: the Framingham office model's inputs."""
    return np.column_stack([np.log(df["age_years"]), np.log(df["bmi"]), np.log(df["ap_hi"]),
                            df["smoke"], diabetes_proxy(df)])


def framingham_risk(df: pd.DataFrame) -> pd.Series:
    """Published Framingham 10-year CVD risk, as a clinician would compute it, with nothing refitted."""
    terms = framingham_terms(df)
    risk = np.empty(len(df))
    for male, c in FRAMINGHAM.items():
        rows = df["male"].to_numpy() == male
        beta = np.array([c["ln_age"], c["ln_bmi"], c["ln_sbp"], c["smoke"], c["diabetes"]])
        risk[rows] = 1 - c["s0"] ** np.exp(terms[rows] @ beta - c["mean"])
    return pd.Series(risk, index=df.index)


def framingham_design(df: pd.DataFrame) -> np.ndarray:
    """Framingham's terms once for women and once for men, plus a sex indicator, so that a logistic
    regression refits the published model's two equations on this population."""
    male = df["male"].to_numpy()[:, None]
    terms = framingham_terms(df)
    return np.column_stack([terms * (1 - male), terms * male, male])


# ------------------------------------------------------------------------------------------ models

N_JOBS = int(os.environ.get("CARDIO_JOBS", "4"))
CV_FOLDS = 5
# Simplest first: when several models are statistically tied, the earliest in this order is chosen
SIMPLICITY = ["Framingham form (refit)", "Logistic regression", "Gradient boosting", "Random forest",
              "Neural network (MLP)"]


def cv_splitter(seed: int = SEED):
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)


def model_specs() -> dict[str, tuple]:
    """Each model with the hyperparameter grid that cross-validation searches."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    def scaled(name, model):
        return Pipeline([("scale", StandardScaler()), (name, model)])

    return {
        "Framingham form (refit)": (
            Pipeline([("terms", FunctionTransformer(framingham_design)), ("scale", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=5000))]),
            {"lr__C": [1.0]}),
        "Logistic regression": (
            scaled("lr", LogisticRegression(max_iter=5000)),
            {"lr__C": [0.01, 0.1, 1.0, 10.0]}),
        "Random forest": (
            RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
            {"min_samples_leaf": [10, 30, 100], "max_features": ["sqrt", 0.5]}),
        "Gradient boosting": (
            # early stopping on 10% of the training data (sklearn's default above 10,000 rows) ends
            # the boosting long before max_iter at the larger learning rates
            HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, random_state=SEED),
            {"learning_rate": [0.01, 0.03, 0.1], "max_leaf_nodes": [7, 15, 31],
             "min_samples_leaf": [20, 50, 200]}),
        "Neural network (MLP)": (
            scaled("mlp", MLPClassifier(early_stopping=True, max_iter=300, random_state=SEED)),
            {"mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "mlp__alpha": [1e-5, 1e-4, 1e-2]}),
    }


def classification_metrics(y, score, threshold: float) -> dict:
    """AUC from the score; accuracy, sensitivity and specificity from score >= threshold."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    pred = np.asarray(score) >= threshold
    return {"auc": roc_auc_score(y, score), "accuracy": (pred == y).mean(),
            "sensitivity": pred[y == 1].mean(), "specificity": (~pred[y == 0]).mean()}


def accuracy_threshold(y, score) -> float:
    """The cut-off on a score that classifies the given (training) patients most accurately."""
    y, score = np.asarray(y), np.asarray(score)
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197)))
    return candidates[np.argmax([((score >= t) == y).mean() for t in candidates])]


def compare_models(train: pd.DataFrame, features: list[str] = FEATURES, specs: dict | None = None,
                   n_jobs: int = N_JOBS) -> tuple[pd.DataFrame, dict]:
    """Cross-validated comparison of the clinical baselines and the machine-learning models.

    Everything is scored on the same five validation folds of the training patients. The baselines
    fit nothing; where a baseline needs a cut-off (systolic pressure alone, published Framingham risk),
    it is chosen on the other four folds. Each model's grid is searched by AUC, and its row reports
    the best setting. Returns the table and the fitted searches (refitted on all training patients).
    """
    from sklearn.metrics import make_scorer, recall_score
    from sklearn.model_selection import GridSearchCV

    X, y = train[features], train[TARGET]
    cv = cv_splitter()
    rows = []
    baselines = [
        (f"Rule: systolic ≥ {SBP_RULE} mmHg", (train["ap_hi"] >= SBP_RULE).astype(float), 0.5),
        ("Systolic pressure alone", train["ap_hi"].astype(float), None),
        ("Framingham (published)", framingham_risk(train), None),
    ]
    for name, score, fixed in baselines:
        folds = []
        for tr, va in cv.split(X, y):
            cut = fixed if fixed is not None else accuracy_threshold(y.iloc[tr], score.iloc[tr])
            folds.append(classification_metrics(y.iloc[va], score.iloc[va], cut))
        f = pd.DataFrame(folds)
        rows.append({"model": name, "kind": "clinical baseline", "auc": f["auc"].mean(), "auc_sd": f["auc"].std(),
                     **f[["accuracy", "sensitivity", "specificity"]].mean().to_dict(),
                     "best setting": "-" if fixed is not None else "cut-off chosen on training folds",
                     "fit seconds": 0.0})

    scoring = {"auc": "roc_auc", "accuracy": "accuracy", "sensitivity": make_scorer(recall_score),
               "specificity": make_scorer(recall_score, pos_label=0)}
    searches = {}
    for name, (estimator, grid) in (specs or model_specs()).items():
        search = GridSearchCV(estimator, grid, scoring=scoring, refit="auc", cv=cv, n_jobs=n_jobs)
        search.fit(X, y)
        r, i = search.cv_results_, search.best_index_
        rows.append({"model": name, "kind": "machine learning", "auc": r["mean_test_auc"][i],
                     "auc_sd": r["std_test_auc"][i],
                     **{m: r[f"mean_test_{m}"][i] for m in ("accuracy", "sensitivity", "specificity")},
                     "best setting": ", ".join(f"{k.split('__')[-1]}={v}" for k, v in search.best_params_.items()),
                     "fit seconds": r["mean_fit_time"][i]})
        searches[name] = search
    return pd.DataFrame(rows).set_index("model"), searches


def choose_model(table: pd.DataFrame) -> str:
    """The simplest machine-learning model whose CV AUC is within one standard error of the best.

    Differences smaller than the fold-to-fold noise are not evidence that one model is better, so they
    should not buy complexity (the 'one-standard-error rule').
    """
    ml = table[table["kind"] == "machine learning"]
    best = ml["auc"].idxmax()
    margin = ml.loc[best, "auc_sd"] / math.sqrt(CV_FOLDS)
    tied = ml.index[ml["auc"] >= ml.loc[best, "auc"] - margin]
    return min(tied, key=SIMPLICITY.index)


def feature_set_comparison(train: pd.DataFrame, searches: dict, names: list[str],
                           n_jobs: int = N_JOBS) -> pd.DataFrame:
    """CV AUC of the chosen settings with the recorded features only, and with the derived ones too."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    rows = []
    for name in names:
        for label, cols in (("recorded only", BASE_FEATURES), ("recorded + derived", FEATURES)):
            s = cross_val_score(clone(searches[name].best_estimator_), train[cols], train[TARGET],
                                cv=cv_splitter(), scoring="roc_auc", n_jobs=n_jobs)
            rows.append({"model": name, "features": label, "auc": s.mean(), "auc_sd": s.std(ddof=1)})
    return pd.DataFrame(rows).pivot(index="model", columns="features", values="auc")


def learning_curve_auc(train: pd.DataFrame, estimators: dict,
                       sizes: tuple = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
                       n_jobs: int = N_JOBS) -> pd.DataFrame:
    """Validation AUC as the number of training patients grows (sizes are shares of a training fold):
    does the model run out of data, or of information in the features?"""
    from sklearn.base import clone
    from sklearn.model_selection import learning_curve

    rows = []
    for name, est in estimators.items():
        n, _, val = learning_curve(clone(est), train[FEATURES], train[TARGET], train_sizes=list(sizes),
                                   cv=cv_splitter(), scoring="roc_auc", n_jobs=n_jobs, shuffle=True,
                                   random_state=SEED)
        rows += [{"model": name, "patients": k, "auc": v.mean(), "auc_sd": v.std(ddof=1)} for k, v in zip(n, val)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- evaluation

def bootstrap(y, scores: dict, statistic, n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Percentile bootstrap of statistic(y, score) for several scores, resampling the same patients
    for all of them, so differences between scores can be bootstrapped too. One row per resample."""
    y = np.asarray(y)
    scores = {k: np.asarray(v) for k, v in scores.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        rows.append({k: statistic(y[idx], s[idx]) for k, s in scores.items()})
    return pd.DataFrame(rows)


def interval(samples: pd.Series, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(samples.quantile(a)), float(samples.quantile(1 - a))


def out_of_fold(estimator, train: pd.DataFrame, features: list[str] = FEATURES,
                n_jobs: int = N_JOBS) -> np.ndarray:
    """Each training patient's predicted risk from a model that did not see that patient."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    return cross_val_predict(clone(estimator), train[features], train[TARGET], cv=cv_splitter(),
                             method="predict_proba", n_jobs=n_jobs)[:, 1]


def screening_threshold(y, risk, sensitivity: float = 0.80) -> float:
    """The highest cut-off that still detects at least `sensitivity` of the diseased patients."""
    y, risk = np.asarray(y), np.asarray(risk)
    sick = np.sort(risk[y == 1])
    return float(sick[int(math.floor((1 - sensitivity) * len(sick)))])


# ---------------------------------------------------------------------------------- points score

POINTS_BANDS = {
    "age_years": ([0, 45, 50, 55, 60, 200], ["<45", "45-49", "50-54", "55-59", "60+"]),
    "ap_hi": ([0, 120, 130, 140, 160, 400], ["<120", "120-129", "130-139", "140-159", "160+"]),
    "bmi": ([0, 25, 30, 35, 200], ["<25", "25-29.9", "30-34.9", "35+"]),
}
POINTS_LEVELS = {"cholesterol": {1: "normal", 2: "above normal", 3: "well above normal"},
                 "gluc": {1: "normal", 2: "above normal", 3: "well above normal"},
                 "smoke": {0: "no", 1: "yes"}, "alco": {0: "no", 1: "yes"}, "active": {0: "no", 1: "yes"}}
POINTS_LABELS = {"age_years": "Age (years)", "ap_hi": "Systolic pressure (mmHg)", "bmi": "BMI (kg/m²)",
                 "cholesterol": "Cholesterol", "gluc": "Glucose", "smoke": "Smokes", "alco": "Drinks alcohol",
                 "active": "Physically active"}


class PointsScore:
    """A paper scorecard: each factor's band earns whole points, and the total maps to a risk.

    Fitted as a logistic regression on banded factors (lowest band as reference). One point is worth
    `unit` in log-odds, so every coefficient is rounded to whole points. The total is then mapped back
    to a risk with a one-variable logistic regression on the training patients.
    """

    def __init__(self, unit: float = 0.1):
        self.unit = unit

    def _bands(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for col, (edges, labels) in POINTS_BANDS.items():
            cols[col] = pd.Categorical(pd.cut(df[col], edges, labels=labels, right=False), labels)
        for col, levels in POINTS_LEVELS.items():
            cols[col] = pd.Categorical(df[col].map(levels), list(levels.values()))
        return pd.DataFrame(cols, index=df.index)

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(self._bands(df), drop_first=True, dtype=float)

    def fit(self, df: pd.DataFrame, y):
        from sklearn.linear_model import LogisticRegression

        X = self._design(df)
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
        self.points_ = pd.Series(np.round(lr.coef_[0] / self.unit).astype(int), index=X.columns)
        self.calibration_ = LogisticRegression(C=1e6, max_iter=5000).fit(self.total(df).to_frame(), y)
        return self

    def total(self, df: pd.DataFrame) -> pd.Series:
        return (self._design(df)[self.points_.index] @ self.points_).rename("points")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibration_.predict_proba(self.total(df).to_frame())[:, 1]

    def table(self) -> pd.DataFrame:
        """The scorecard: every band of every factor with its points (reference bands score 0)."""
        rows = []
        for col in [*POINTS_BANDS, *POINTS_LEVELS]:
            labels = POINTS_BANDS[col][1] if col in POINTS_BANDS else list(POINTS_LEVELS[col].values())
            for label in labels:
                rows.append({"factor": POINTS_LABELS[col], "band": label,
                             "points": int(self.points_.get(f"{col}_{label}", 0))})
        return pd.DataFrame(rows)

    def risk_by_total(self, totals) -> pd.Series:
        totals = np.asarray(list(totals))
        return pd.Series(self.calibration_.predict_proba(pd.DataFrame({"points": totals}))[:, 1], index=totals)


# ---------------------------------------------------------------------------------- explanation

# Features that carry the same information are permuted together: permuting systolic pressure alone
# would leave mean arterial pressure, pulse pressure and the stage to stand in for it.
FEATURE_GROUPS = {
    "blood pressure": ["ap_hi", "ap_lo", "pulse_pressure", "mean_arterial_pressure", "bp_stage"],
    "age": ["age_years"],
    "body size (height, weight, BMI)": ["height", "weight", "bmi"],
    "cholesterol": ["cholesterol"],
    "glucose": ["gluc"],
    "physical activity": ["active"],
    "smoking": ["smoke"],
    "alcohol": ["alco"],
    "sex": ["male"],
}


def grouped_permutation_importance(model, df: pd.DataFrame, groups: dict = FEATURE_GROUPS,
                                   n_repeats: int = 10, seed: int = SEED) -> pd.DataFrame:
    """Drop in AUC when each group of features is shuffled across patients (the same shuffle for every
    column of the group, so the group stays internally consistent but loses its link to the diagnosis)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[cols] = X[cols].to_numpy()[order]
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        rows.append({"features": name, "auc drop": np.mean(drops), "sd": np.std(drops, ddof=1)})
    return pd.DataFrame(rows).set_index("features").sort_values("auc drop", ascending=False)


def partial_dependence_curve(model, df: pd.DataFrame, feature: str, values) -> pd.Series:
    """Average predicted risk when every patient's `feature` is set to each value in turn, with the
    features derived from it recomputed, so the model never sees an impossible combination."""
    out = {}
    for v in values:
        d = df.copy()
        d[feature] = v
        if feature == "ap_hi":  # keep every diastolic reading below the systolic one it is paired with
            d["ap_lo"] = np.minimum(d["ap_lo"], v - 10)
        d = add_features(d.drop(columns=DERIVED_FEATURES))
        out[v] = model.predict_proba(d[FEATURES])[:, 1].mean()
    return pd.Series(out, name=feature)


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


def _cmd_compare(args: argparse.Namespace) -> None:
    train, _ = train_test(args.strategy, args.data)
    print(f"{len(train):,} training patients ({args.strategy}), {CV_FOLDS}-fold cross-validation\n")
    table, _ = compare_models(train)
    print(table.drop(columns="best setting").to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nchosen (one-standard-error rule): {choose_model(table)}")
    OUTPUTS.mkdir(exist_ok=True)
    table.to_csv(OUTPUTS / "model_comparison.csv")
    print(f"written: {OUTPUTS / 'model_comparison.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=DATA, help="path to cardio.csv")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("clean", help="data-quality audit and cleaning strategies").set_defaults(func=_cmd_clean)
    compare = sub.add_parser("compare", help="cross-validated comparison of baselines and models")
    compare.add_argument("--strategy", default="drop", choices=STRATEGIES)
    compare.set_defaults(func=_cmd_compare)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
