"""Cardiovascular disease prediction from routine examination data.

Dataset: cardio.csv, 70,000 patients, 11 features collected at a medical examination and a binary
target (cardio: presence of cardiovascular disease).

This module holds everything the report (report.ipynb) relies on, so the notebook and a plain run of
this file always use the same code:

    python cardio.py clean      data-quality audit and the three cleaning strategies, side by side
    python cardio.py compare    cross-validated comparison of the clinical baselines and the models
    python cardio.py train      fit the final model and save it to outputs/model.joblib
    python cardio.py predict --age 55 --sex male --height 175 --weight 90 --systolic 150 --diastolic 95
                                assess one patient with the saved model

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


def split_ids(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED) -> tuple[set, set]:
    """Train and test patient ids, drawn once from the deduplicated raw data, stratified by diagnosis.

    Every cleaning strategy is split by these same ids, so all of them are scored on the same patients.
    """
    from sklearn.model_selection import train_test_split

    d = deduplicate(df)
    train, test = train_test_split(d["id"], test_size=test_size, stratify=d[TARGET], random_state=seed)
    return set(train), set(test)


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

    Exactly one rule may explain a reading; a value that would need two guesses (a sign flip and then
    a unit fix, say) is not a recognisable error, and (nan, None) is returned. The rules:
      sign          a negative reading is the same reading with a stray minus (-120 -> 120)
      cmHg          recorded in cmHg instead of mmHg (12 -> 120), only if that lands in range
      extra digits  trailing digits typed by mistake (1000 -> 100, 14020 -> 140, 906 -> 90)
    """
    if pd.isna(value) or not math.isfinite(value):
        return math.nan, None
    if lo <= value <= hi:
        return value, None
    if value < 0:
        return (-value, "sign") if lo <= -value <= hi else (math.nan, None)
    if 0 < value < lo:
        return (value * 10, "cmHg") if lo <= value * 10 <= hi else (math.nan, None)
    if value > hi and float(value).is_integer():
        digits = int(value)
        while digits > hi:
            digits //= 10
        return (digits, "extra digits") if digits >= lo else (math.nan, None)
    return math.nan, None  # zero, or a fractional reading above range


def repair(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blood pressures with a recognisable entry error, corrected; plus a log of every change.

    One error per record pair: readings typed into each other's fields are swapped first, on the
    values as recorded, and only when the swapped pair is plausible; a record that is swapped gets no
    other rule. Each remaining reading may then be fixed by one rule of repair_pressure(). Nothing is
    swapped after another rule has fired, so no record carries two stacked guesses.

    Height and weight are not repaired: their errors show no pattern that identifies the true value.
    A value no rule can recover becomes NaN, so the record fails audit() and is dropped afterwards.
    """
    out = df.copy()
    out[["ap_hi", "ap_lo"]] = out[["ap_hi", "ap_lo"]].astype(float)  # room for NaN
    log = []
    # 1. systolic and diastolic typed into each other's field
    swapped = (out["ap_lo"] > out["ap_hi"]) & out["ap_hi"].between(*BOUNDS["ap_lo"]) \
        & out["ap_lo"].between(*BOUNDS["ap_hi"])
    for idx in out.index[swapped]:
        hi_, lo_ = out.at[idx, "ap_hi"], out.at[idx, "ap_lo"]
        log.append({"id": out.at[idx, "id"], "field": "ap_hi/ap_lo", "before": f"{hi_:g}/{lo_:g}",
                    "after": f"{lo_:g}/{hi_:g}", "rule": "swapped"})
    out.loc[swapped, ["ap_hi", "ap_lo"]] = out.loc[swapped, ["ap_lo", "ap_hi"]].to_numpy()
    # 2. one rule per remaining implausible reading (a swapped pair is already in range)
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
    risk = np.full(len(df), np.nan)
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
# Simplest first: when several models are statistically tied, the earliest in this order is chosen.
# Fixed before any result: the two linear models (a few coefficients that can be read directly), then
# the tree ensembles from smaller to larger (gradient boosting builds small trees, at most 31 leaves
# each; random forest grows 200 deep ones), then the neural network, the hardest to inspect.
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
            {"min_samples_leaf": [10, 30, 100], "max_features": [2, "sqrt", 0.5]}),
        "Gradient boosting": (
            # early stopping on 10% of the training data (sklearn's default above 10,000 rows) ends
            # the boosting long before max_iter at the larger learning rates
            HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, random_state=SEED),
            {"learning_rate": [0.01, 0.03, 0.1], "max_leaf_nodes": [7, 15, 31],
             "min_samples_leaf": [20, 50, 200]}),
        "Neural network (MLP)": (
            scaled("mlp", MLPClassifier(early_stopping=True, max_iter=300, random_state=SEED)),
            {"mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (256, 128)],
             "mlp__alpha": [1e-4, 1e-2, 1e-1, 1.0]}),
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
    the best setting. `auc_sd` is the sample standard deviation over the five folds for every row.
    Returns the table and the fitted searches (refitted on all training patients).
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
    for name, (estimator, grid) in (specs if specs is not None else model_specs()).items():
        search = GridSearchCV(estimator, grid, scoring=scoring, refit="auc", cv=cv, n_jobs=n_jobs)
        search.fit(X, y)
        r, i = search.cv_results_, search.best_index_
        fold_aucs = [r[f"split{k}_test_auc"][i] for k in range(CV_FOLDS)]
        rows.append({"model": name, "kind": "machine learning", "auc": r["mean_test_auc"][i],
                     "auc_sd": float(np.std(fold_aucs, ddof=1)),
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
    caught = math.ceil(sensitivity * len(sick) - 1e-9)   # patients the cut-off must catch
    return float(sick[len(sick) - caught])


def threshold_at_specificity(y, risk, specificity: float) -> float:
    """The lowest cut-off that still clears at least `specificity` of the healthy patients, i.e. the
    most sensitive operating point at that specificity. Patients at or above the cut-off are flagged."""
    y, risk = np.asarray(y), np.asarray(risk)
    healthy = np.sort(risk[y == 0])
    cleared = math.ceil(specificity * len(healthy) - 1e-9)  # healthy patients that must fall below it
    if cleared == 0:
        return -math.inf
    return float(np.nextafter(healthy[cleared - 1], math.inf))


# ---------------------------------------------------------------------------------- points score

POINTS_BANDS = {
    "age_years": ([0, 45, 50, 55, 60, 200], ["<45", "45-49", "50-54", "55-59", "60+"]),
    "ap_hi": ([0, 120, 130, 140, 160, 400], ["<120", "120-129", "130-139", "140-159", "160+"]),
    "bmi": ([0, 25, 30, 35, 200], ["<25", "25-29.9", "30-34.9", "35+"]),
}
POINTS_LEVELS = {"cholesterol": {1: "normal", 2: "above normal", 3: "well above normal"},
                 "gluc": {1: "normal", 2: "above normal", 3: "well above normal"},
                 "smoke": {0: "no", 1: "yes"}, "alco": {0: "no", 1: "yes"}, "active": {0: "no", 1: "yes"}}
# The clinical scorecard (Section 8.5): measured values only, and only those whose training-set points
# go the way medicine expects. The self-reported answers are left out whatever their sign (reverse
# causation makes them unreliable here), and glucose because its training-set points are negative.
CLINICAL_FACTORS = ["age_years", "ap_hi", "bmi", "cholesterol"]
POINTS_LABELS = {"age_years": "Age (years)", "ap_hi": "Systolic pressure (mmHg)", "bmi": "BMI (kg/m²)",
                 "cholesterol": "Cholesterol", "gluc": "Glucose", "smoke": "Smokes", "alco": "Drinks alcohol",
                 "active": "Physically active"}


class PointsScore:
    """A paper scorecard: each factor's band earns whole points, and the total maps to a risk.

    Fitted as a logistic regression on banded factors (lowest band as reference). One point is worth
    `unit` in log-odds, so every coefficient is rounded to whole points. The total is then mapped back
    to a risk by isotonic regression on the training patients: the share with CVD at each total,
    forced to rise with the total. (A logistic curve in the total is too rigid here: the large step in
    points between systolic bands makes the totals cluster, and the curve misstates both ends.)
    """

    def __init__(self, unit: float = 0.1, factors: list[str] | None = None):
        self.unit = unit
        self.factors = factors or [*POINTS_BANDS, *POINTS_LEVELS]

    def _bands(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for col in self.factors:
            if col in POINTS_BANDS:
                edges, labels = POINTS_BANDS[col]
                cols[col] = pd.Categorical(pd.cut(df[col], edges, labels=labels, right=False), labels)
            else:
                levels = POINTS_LEVELS[col]
                cols[col] = pd.Categorical(df[col].map(levels), list(levels.values()))
        bands = pd.DataFrame(cols, index=df.index)
        outside = [c for c in bands if bands[c].isna().any()]
        if outside:  # never score a value the card has no band for as if it were the reference band
            raise ValueError("values outside the scorecard's bands: " + ", ".join(outside))
        return bands

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(self._bands(df), drop_first=True, dtype=float)

    def fit(self, df: pd.DataFrame, y):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        X = self._design(df)
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
        self.points_ = pd.Series(np.round(lr.coef_[0] / self.unit).astype(int), index=X.columns)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(self.total(df), np.asarray(y))
        self.risk_points_ = np.asarray(iso.X_thresholds_, dtype=float)
        self.risk_values_ = np.asarray(iso.y_thresholds_, dtype=float)
        return self

    def total(self, df: pd.DataFrame) -> pd.Series:
        return (self._design(df)[self.points_.index] @ self.points_).rename("points")

    def _risk(self, totals) -> np.ndarray:
        totals = np.asarray(totals, dtype=float)
        risk = np.interp(totals, self.risk_points_, self.risk_values_)
        # isotonic regression can give neighbouring totals the same risk; a millionth-sized nudge keeps
        # more points always meaning more risk, so ranking by risk is ranking by points
        span = max(self.risk_points_[-1] - self.risk_points_[0], 1.0)
        return risk * (1 - 1e-6) + 1e-6 * np.clip((totals - self.risk_points_[0]) / span, 0, 1)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._risk(self.total(df))

    def to_dict(self) -> dict:
        """Plain numbers only, so a saved scorecard loads in any program, with no pickled class."""
        return {"unit": self.unit, "factors": list(self.factors),
                "points": {k: int(v) for k, v in self.points_.items()},
                "risk_points": self.risk_points_.tolist(), "risk_values": self.risk_values_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "PointsScore":
        card = cls(d["unit"], d["factors"])
        card.points_ = pd.Series(d["points"], dtype=int)
        card.risk_points_ = np.asarray(d["risk_points"], dtype=float)
        card.risk_values_ = np.asarray(d["risk_values"], dtype=float)
        return card

    def table(self) -> pd.DataFrame:
        """The scorecard: every band of every factor with its points (reference bands score 0)."""
        rows = []
        for col in self.factors:
            labels = POINTS_BANDS[col][1] if col in POINTS_BANDS else list(POINTS_LEVELS[col].values())
            for label in labels:
                rows.append({"factor": POINTS_LABELS[col], "band": label,
                             "points": int(self.points_.get(f"{col}_{label}", 0))})
        return pd.DataFrame(rows)

    def risk_by_total(self, totals) -> pd.Series:
        totals = np.asarray(list(totals))
        return pd.Series(self._risk(totals), index=totals)


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
    features derived from it recomputed, so the model never sees an impossible combination.

    For systolic pressure, a diastolic reading that would reach the new systolic value is capped 10 mmHg
    below it; at the low end of the curve this also lowers some diastolic readings (about 7% of patients
    at 100 mmHg, under 1% from 120 mmHg up), so that end is not a pure systolic effect."""
    out = {}
    for v in values:
        d = df.copy()
        d[feature] = v
        if feature == "ap_hi":  # keep every diastolic reading below the systolic one it is paired with
            d["ap_lo"] = np.minimum(d["ap_lo"], v - 10)
        d = add_features(d.drop(columns=DERIVED_FEATURES))
        out[v] = model.predict_proba(d[FEATURES])[:, 1].mean()
    return pd.Series(out, name=feature)


def drop_group_importance(estimator, train: pd.DataFrame, groups: dict = FEATURE_GROUPS,
                          n_jobs: int = N_JOBS) -> pd.DataFrame:
    """Cross-validated AUC (training patients) of the same model retrained without each group of
    features. Unlike permutation importance, this asks what the data can do without a group, not how
    much one fitted model leans on it."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    def cv_auc(cols):
        return cross_val_score(clone(estimator), train[cols], train[TARGET], cv=cv_splitter(),
                               scoring="roc_auc", n_jobs=n_jobs).mean()

    full = cv_auc(FEATURES)
    rows = [{"features left out": "none (all 15)", "CV AUC": full, "AUC lost": 0.0}]
    for name, cols in groups.items():
        auc = cv_auc([f for f in FEATURES if f not in cols])
        rows.append({"features left out": name, "CV AUC": auc, "AUC lost": full - auc})
    return pd.DataFrame(rows).set_index("features left out")


# ------------------------------------------------------------------------------------ reliability

def strategy_comparison(estimators: dict, path: Path | str = DATA, n_boot: int = 2000) -> pd.DataFrame:
    """Each estimator trained on each cleaning strategy's training patients and scored on the common
    test patients (those whose records pass every check), so only the training data differs. The AUC
    change against 'drop' gets a 95% paired bootstrap interval over the test patients."""
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    common = train_test("drop", path)[1].set_index("id")
    y = common[TARGET]
    rows, risks = [], {}
    for strategy in STRATEGIES:
        tr, te = train_test(strategy, path)
        te = te.set_index("id").loc[common.index]          # the same patients, in the same order
        for name, est in estimators.items():
            model = clone(est).fit(tr[FEATURES], tr[TARGET])
            risk = model.predict_proba(te[FEATURES])[:, 1]
            risks[(name, strategy)] = risk
            rows.append({"model": name, "strategy": strategy, "training patients": len(tr),
                         "test AUC": roc_auc_score(y, risk), "test accuracy": float(((risk >= 0.5) == y).mean())})
    table = pd.DataFrame(rows)
    table["AUC change vs drop"] = 0.0
    table["95% CI"] = "-"
    for name in estimators:
        boot = bootstrap(y, {s: risks[(name, s)] for s in STRATEGIES}, roc_auc_score, n=n_boot)
        base = table.loc[(table["model"] == name) & (table["strategy"] == "drop"), "test AUC"].iloc[0]
        for s in STRATEGIES:
            row = (table["model"] == name) & (table["strategy"] == s)
            table.loc[row, "AUC change vs drop"] = table.loc[row, "test AUC"].iloc[0] - base
            if s != "drop":
                lo, hi = interval(boot[s] - boot["drop"])
                table.loc[row, "95% CI"] = f"{lo:+.4f} to {hi:+.4f}"
    return table


def excluded_test_patients(model, path: Path | str = DATA) -> pd.DataFrame:
    """The test patients that 'drop' removes as implausible: how many, how often they have CVD, how many
    'repair' recovers, and how the final model (trained on 'drop') scores the recovered ones."""
    from sklearn.metrics import roc_auc_score

    raw = load(path)
    _, test_ids = split_ids(raw)
    raw_test = deduplicate(raw)
    raw_test = raw_test[raw_test["id"].isin(test_ids)]
    valid = train_test("drop", path)[1]
    repaired = train_test("repair", path)[1]
    removed = raw_test[~raw_test["id"].isin(valid["id"])]
    recovered = repaired[repaired["id"].isin(removed["id"])]

    def scored(label, d):
        risk = model.predict_proba(d[FEATURES])[:, 1]
        return {"test patients": label, "patients": len(d), "share with CVD": d[TARGET].mean(),
                "AUC": roc_auc_score(d[TARGET], risk), "accuracy": float(((risk >= 0.5) == d[TARGET]).mean())}

    rows = [scored("valid records (the test set of Section 8)", valid),
            {"test patients": "removed as implausible by 'drop'", "patients": len(removed),
             "share with CVD": removed[TARGET].mean(), "AUC": np.nan, "accuracy": np.nan},
            scored("  of which recovered by 'repair', scored after repair", recovered),
            scored("valid + recovered", pd.concat([valid, recovered]))]
    return pd.DataFrame(rows).set_index("test patients")


def smote(X: np.ndarray, y: np.ndarray, n_new: int, k: int = 5, seed: int = SEED):
    """SMOTE: new patients interpolated between a patient and one of its k nearest neighbours of the
    same class, n_new per class. Written out here because the leakage experiment needs nothing more."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    new_X, new_y = [], []
    for c in np.unique(y):
        Xc = X[y == c]
        _, nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Xc).kneighbors(Xc)
        i = rng.integers(0, len(Xc), n_new)
        j = nbrs[i, rng.integers(1, k + 1, n_new)]
        lam = rng.random((n_new, 1))
        new_X.append(Xc[i] + lam * (Xc[j] - Xc[i]))
        new_y.append(np.full(n_new, c))
    return np.vstack(new_X), np.concatenate(new_y)


def leakage_experiments(train: pd.DataFrame, test: pd.DataFrame, correct_accuracy: float,
                        seed: int = SEED) -> pd.DataFrame:
    """Accuracy 'reported' by pipelines that make mistakes common in published work on this kind of
    data, next to the correct pipeline. Each mistake lets the evaluation see what it should not."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [{"pipeline": "Correct: model chosen by cross-validation, scored once on unseen patients",
             "model": "gradient boosting", "reported accuracy": correct_accuracy}]

    # 1. scoring the model on the patients it was trained on
    # 50 fully grown trees: enough to show the effect, and they stay well inside the 4 GB container
    forest = RandomForestClassifier(n_estimators=50, min_samples_leaf=1, random_state=seed, n_jobs=N_JOBS)
    forest.fit(train[FEATURES], train[TARGET])
    rows.append({"pipeline": "Mistake 1: accuracy measured on the training patients",
                 "model": "random forest (fully grown trees)",
                 "reported accuracy": forest.score(train[FEATURES], train[TARGET])})
    rows.append({"pipeline": "  (the same forest on unseen patients)", "model": "random forest (fully grown trees)",
                 "reported accuracy": forest.score(test[FEATURES], test[TARGET])})

    # 2. oversampling (SMOTE) the whole dataset, then splitting it. Each model is shown three ways: with
    #    no oversampling and with SMOTE done correctly (training patients only), both scored on unseen
    #    patients, and with SMOTE done before the split
    everything = pd.concat([train, test])
    X, y = everything[FEATURES].to_numpy(float), everything[TARGET].to_numpy()
    sx, sy = smote(X, y, n_new=len(everything), seed=seed)      # triples the data, as some papers do
    Xa, ya = np.vstack([X, sx]), np.concatenate([y, sy])
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=TEST_SIZE, stratify=ya, random_state=seed)
    tx, ty = train[FEATURES].to_numpy(float), train[TARGET].to_numpy()
    vx, vy = test[FEATURES].to_numpy(float), test[TARGET].to_numpy()
    tsx, tsy = smote(tx, ty, n_new=len(train), seed=seed)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    for label, model in [("k-nearest neighbours (k = 1)", knn), ("random forest (fully grown trees)", forest)]:
        model.fit(tx, ty)
        rows.append({"pipeline": "  (no oversampling, scored on unseen patients)", "model": label,
                     "reported accuracy": model.score(vx, vy)})
        model.fit(np.vstack([tx, tsx]), np.concatenate([ty, tsy]))
        rows.append({"pipeline": "  (SMOTE on the training patients only, scored on unseen patients)",
                     "model": label, "reported accuracy": model.score(vx, vy)})
        model.fit(Xtr, ytr)
        rows.append({"pipeline": "Mistake 2: SMOTE oversampling before the train/test split", "model": label,
                     "reported accuracy": model.score(Xte, yte)})

    # 3. trying many random splits and reporting the luckiest
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    accs = []
    for s in range(50):
        a, b = train_test_split(everything, test_size=TEST_SIZE, stratify=everything[TARGET], random_state=s)
        accs.append(lr.fit(a[FEATURES], a[TARGET]).score(b[FEATURES], b[TARGET]))
    rows.append({"pipeline": "Mistake 3: best of 50 random train/test splits", "model": "logistic regression",
                 "reported accuracy": max(accs)})
    rows.append({"pipeline": "  (the median split)", "model": "logistic regression",
                 "reported accuracy": float(np.median(accs))})
    return pd.DataFrame(rows)


def subgroup_table(df: pd.DataFrame, risk, threshold: float = 0.5, n_boot: int = 1000) -> pd.DataFrame:
    """By sex and by age band: the model's AUC with a 95% bootstrap interval, the AUC of systolic
    pressure alone in the same group (to separate the model from the group), sensitivity and
    specificity at the cut-off, and the mean predicted risk against the observed share
    (calibration-in-the-large). AUCs of groups that differ in spread or prevalence are not like for
    like: within a narrow age band, for instance, age itself no longer separates anyone."""
    from sklearn.metrics import roc_auc_score

    d = df.assign(risk=np.asarray(risk))
    groups = {
        "women": d["male"] == 0, "men": d["male"] == 1,
        "age < 50": d["age_years"] < 50, "age 50-59": d["age_years"].between(50, 60, inclusive="left"),
        "age 60+": d["age_years"] >= 60, "everyone": pd.Series(True, index=d.index),
    }
    rows = []
    for name, mask in groups.items():
        g = d[mask]
        m = classification_metrics(g[TARGET], g["risk"], threshold)
        lo, hi = interval(bootstrap(g[TARGET], {"auc": g["risk"]}, roc_auc_score, n=n_boot)["auc"])
        rows.append({"group": name, "patients": len(g), "share with CVD": g[TARGET].mean(),
                     "mean predicted risk": g["risk"].mean(), "AUC": m["auc"], "AUC 95% CI": f"{lo:.3f}-{hi:.3f}",
                     "systolic alone AUC": roc_auc_score(g[TARGET], g["ap_hi"]),
                     "sensitivity": m["sensitivity"], "specificity": m["specificity"]})
    return pd.DataFrame(rows).set_index("group")


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Share of flagged patients who really have CVD, in a population with the given prevalence."""
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return tp / (tp + fp)


# ------------------------------------------------------------------------------- using the model

MODEL_FILE = OUTPUTS / "model.joblib"
# The setting Section 7 selected for gradient boosting by cross-validation (the report checks that
# its own search still agrees with this before saving the model)
FINAL_PARAMS = {"learning_rate": 0.01, "max_leaf_nodes": 15, "min_samples_leaf": 50}
SCREENING_SENSITIVITY = 0.80
AGE_RANGE = (30, 65)   # years: the ages the model was trained on


def final_estimator():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, random_state=SEED, **FINAL_PARAMS)


def build_final(train: pd.DataFrame, model=None) -> dict:
    """Everything needed to assess a new patient: the fitted model, the two cut-offs and the scorecard."""
    model = model if model is not None else final_estimator().fit(train[FEATURES], train[TARGET])
    oof = out_of_fold(model, train)
    return {"model": model, "accuracy_cutoff": 0.5,
            "screening_cutoff": screening_threshold(train[TARGET], oof, SCREENING_SENSITIVITY),
            "points": PointsScore(factors=CLINICAL_FACTORS).fit(train, train[TARGET]), "features": FEATURES}


def save_final(bundle: dict, path: Path | str = MODEL_FILE) -> Path:
    """Save the bundle. The scorecard is stored as plain numbers (PointsScore.to_dict), so the file
    loads in any program that has this code, not only in the one that created it."""
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**bundle, "points": bundle["points"].to_dict()}, path)
    return path


def load_final(path: Path | str = MODEL_FILE) -> dict:
    import joblib
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found: fit and save the final model first (build_final, save_final)")
    bundle = joblib.load(path)
    return {**bundle, "points": PointsScore.from_dict(bundle["points"])}


def patient(age: float, sex: str, height: float, weight: float, systolic: float, diastolic: float,
            cholesterol: int = 1, glucose: int = 1, smoke: int = 0, alcohol: int = 0, active: int = 1) -> pd.DataFrame:
    """One patient's examination as a model-ready row, after the same plausibility checks as the data.

    age in years; sex 'female' or 'male'; height cm; weight kg; blood pressure mmHg; cholesterol and
    glucose 1 normal, 2 above normal, 3 well above normal; smoke, alcohol, active 0 or 1.
    """
    if sex not in ("female", "male"):
        raise ValueError("sex must be 'female' or 'male'")
    for name, value in [("age", age), ("height", height), ("weight", weight), ("systolic", systolic),
                        ("diastolic", diastolic)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a number, not {value!r}")
    if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
        raise ValueError(f"age {age:g} is outside the {AGE_RANGE[0]}-{AGE_RANGE[1]} years the model was trained on")
    for name, value, allowed in [("cholesterol", cholesterol, (1, 2, 3)), ("glucose", glucose, (1, 2, 3)),
                                 ("smoke", smoke, (0, 1)), ("alcohol", alcohol, (0, 1)), ("active", active, (0, 1))]:
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
    raw = pd.DataFrame([{"id": 0, "age": age * 365.25, "gender": 2 if sex == "male" else 1, "height": height,
                         "weight": weight, "ap_hi": systolic, "ap_lo": diastolic, "cholesterol": cholesterol,
                         "gluc": glucose, "smoke": smoke, "alco": alcohol, "active": active, TARGET: 0}])
    failed = audit(raw).iloc[0]
    if failed.any():
        raise ValueError("implausible examination values: " + ", ".join(failed.index[failed]))
    return add_features(prepare(raw))


def assess(bundle: dict, row: pd.DataFrame) -> dict:
    """Predicted risk, the verdict at both cut-offs, and the scorecard's points for one patient."""
    risk = float(bundle["model"].predict_proba(row[bundle["features"]])[:, 1][0])
    points = bundle["points"]
    design = points._design(row)[points.points_.index].iloc[0]
    earned = points.points_[design == 1]
    readable = {}
    for column, value in earned.items():
        factor = next(f for f in points.factors if column.startswith(f + "_"))
        readable[f"{POINTS_LABELS[factor]}: {column[len(factor) + 1:]}"] = int(value)
    return {"risk": risk,
            "verdict": "cardiovascular disease likely" if risk >= bundle["accuracy_cutoff"] else "cardiovascular disease unlikely",
            "screening": "refer for examination" if risk >= bundle["screening_cutoff"] else "no referral",
            "screening_cutoff": bundle["screening_cutoff"],
            "points": int(earned.sum()), "points_risk": float(points.predict_proba(row)[0]),
            "points_breakdown": {k: v for k, v in readable.items() if v != 0}}


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


def _cmd_train(args: argparse.Namespace) -> None:
    train, _ = train_test(args.strategy, args.data)
    bundle = build_final(train)
    path = save_final(bundle)
    print(f"gradient boosting ({FINAL_PARAMS}) fitted on {len(train):,} training patients")
    print(f"screening cut-off (detects {SCREENING_SENSITIVITY:.0%} of diseased training patients): "
          f"{bundle['screening_cutoff']:.3f}")
    print(f"written: {path}")


def _cmd_predict(args: argparse.Namespace) -> None:
    try:
        row = patient(args.age, args.sex, args.height, args.weight, args.systolic, args.diastolic,
                      args.cholesterol, args.glucose, args.smoke, args.alcohol, args.active)
    except ValueError as e:
        raise SystemExit(f"rejected: {e}")
    result = assess(load_final(), row)
    print(f"predicted risk of cardiovascular disease: {result['risk']:.0%}")
    print(f"verdict: {result['verdict']}")
    print(f"screening (cut-off {result['screening_cutoff']:.2f}): {result['screening']}")
    print(f"clinical points score: {result['points']} points, risk {result['points_risk']:.0%}")
    for k, v in result["points_breakdown"].items():
        print(f"   {v:+d}  {k}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=DATA, help="path to cardio.csv")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("clean", help="data-quality audit and cleaning strategies").set_defaults(func=_cmd_clean)
    compare = sub.add_parser("compare", help="cross-validated comparison of baselines and models")
    # 'raw' is not offered: its impossible pressures (negative, zero) break the Framingham logarithms
    compare.add_argument("--strategy", default="drop", choices=["drop", "repair"])
    compare.set_defaults(func=_cmd_compare)
    train = sub.add_parser("train", help="fit the final model and save it")
    train.add_argument("--strategy", default="drop", choices=["drop", "repair"])
    train.set_defaults(func=_cmd_train)
    predict = sub.add_parser("predict", help="assess one patient with the saved model")
    predict.add_argument("--age", type=float, required=True, help="years")
    predict.add_argument("--sex", choices=["female", "male"], required=True)
    predict.add_argument("--height", type=float, required=True, help="cm")
    predict.add_argument("--weight", type=float, required=True, help="kg")
    predict.add_argument("--systolic", type=float, required=True, help="mmHg")
    predict.add_argument("--diastolic", type=float, required=True, help="mmHg")
    predict.add_argument("--cholesterol", type=int, default=1, choices=[1, 2, 3])
    predict.add_argument("--glucose", type=int, default=1, choices=[1, 2, 3])
    predict.add_argument("--smoke", type=int, default=0, choices=[0, 1])
    predict.add_argument("--alcohol", type=int, default=0, choices=[0, 1])
    predict.add_argument("--active", type=int, default=1, choices=[0, 1])
    predict.set_defaults(func=_cmd_predict)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
