"""Tests for the paths an independent review found untested or wrong (repair order, thresholds, the
scorecard's storage and bands, the patient checks, the new reliability helpers)."""
import math

import joblib
import numpy as np
import pandas as pd
import pytest

import cardio
from test_cardio import frame, record, synthetic_patients


# ------------------------------------------------------------------------------------- repairs

def test_transposed_pairs_are_swapped_on_the_recorded_values():
    # 90/160 used to lose its diastolic to the 'extra digits' rule before the swap could see it
    out, log = cardio.repair(frame(record(1, ap_hi=90, ap_lo=160), record(2, ap_hi=95, ap_lo=170)))
    assert out.loc[0, ["ap_hi", "ap_lo"]].tolist() == [160, 90]
    assert out.loc[1, ["ap_hi", "ap_lo"]].tolist() == [170, 95]
    assert list(log["rule"]) == ["swapped", "swapped"]


def test_no_record_gets_two_stacked_guesses():
    # 11/120 would need cmHg (110) and then a swap; 7/80 the same: both are left implausible
    df = frame(record(1, ap_hi=11, ap_lo=120), record(2, ap_hi=7, ap_lo=80), record(3, ap_hi=12, ap_lo=80))
    cleaned, _ = cardio.clean(df, "repair")
    assert list(cleaned["id"]) == [3]                       # 12/80 -> 120/80 needs one rule only
    assert cleaned.loc[0, ["ap_hi", "ap_lo"]].tolist() == [120, 80]


@pytest.mark.parametrize("value, lo, hi", [
    (-12, 70, 250),      # a sign flip that would then need a unit fix: two guesses
    (250.5, 70, 250),    # fractional and above range: not a typed-digit error
    (math.inf, 70, 250),
    (math.nan, 70, 250),
])
def test_repair_pressure_refuses_two_step_or_odd_values(value, lo, hi):
    after, rule = cardio.repair_pressure(value, lo, hi)
    assert math.isnan(after) and rule is None


def test_cleaning_summary_counts_with_the_new_order():
    df = frame(record(1), record(2, ap_hi=90, ap_lo=160), record(3, ap_lo=1000), record(4, ap_hi=11, ap_lo=120))
    summary = cardio.cleaning_summary(df)
    assert summary.loc["repair", "records"] == 3 and summary.loc["repair", "records repaired"] == 2
    assert summary.loc["drop", "records"] == 1


# ----------------------------------------------------------------------------------- thresholds

@pytest.mark.parametrize("n", [5, 10, 27145, 27144])
def test_screening_threshold_is_the_highest_cut_off_reaching_the_target(n):
    rng = np.random.default_rng(n)
    risk = rng.random(n)
    y = np.ones(n, dtype=int)
    t = cardio.screening_threshold(y, risk, 0.8)
    caught = (risk >= t).mean()
    assert caught >= 0.8
    higher = risk[risk > t]
    if len(higher):                                         # the next cut-off up would miss the target
        assert (risk >= higher.min()).mean() < 0.8


def test_threshold_at_specificity():
    y = np.array([0] * 10 + [1] * 10)
    risk = np.r_[np.linspace(0, 0.9, 10), np.linspace(0.5, 1.0, 10)]
    t = cardio.threshold_at_specificity(y, risk, 0.8)
    spec = (risk[y == 0] < t).mean()
    assert spec >= 0.8
    assert (risk[y == 0] < np.nextafter(t, -np.inf)).mean() < 0.8 or t == -np.inf or spec == 0.8


# ------------------------------------------------------------------------------------ scorecard

def test_scorecard_round_trips_as_plain_numbers_and_ranks_by_points():
    train = synthetic_patients(2000)
    card = cardio.PointsScore(factors=cardio.CLINICAL_FACTORS).fit(train, train["cardio"])
    d = card.to_dict()
    assert set(d) == {"unit", "factors", "points", "risk_points", "risk_values"}
    again = cardio.PointsScore.from_dict(d)
    assert np.allclose(again.predict_proba(train), card.predict_proba(train))
    totals, risks = card.total(train).to_numpy(), card.predict_proba(train)
    order = np.argsort(totals, kind="stable")
    assert np.all(np.diff(risks[order]) >= 0)                # more points never means less risk
    tied = np.diff(totals[order]) > 0
    assert np.all(np.diff(risks[order])[tied] > 0)           # and distinct totals get distinct risks


def test_scorecard_refuses_values_outside_its_bands():
    train = synthetic_patients(800)
    card = cardio.PointsScore(factors=["ap_hi", "cholesterol"]).fit(train, train["cardio"])
    bad = train.head(1).assign(cholesterol=7)
    with pytest.raises(ValueError, match="cholesterol"):
        card.predict_proba(bad)


def test_saved_bundle_holds_no_pickled_scorecard_class(tmp_path):
    from sklearn.linear_model import LogisticRegression
    train = synthetic_patients(1500)
    model = LogisticRegression(max_iter=3000).fit(train[cardio.FEATURES], train["cardio"])
    bundle = cardio.build_final(train, model=model)
    path = cardio.save_final(bundle, str(tmp_path / "model.joblib"))     # a plain string path works too
    raw = joblib.load(path)
    assert isinstance(raw["points"], dict)                   # loads in any program, no class needed
    loaded = cardio.load_final(path)
    row = cardio.patient(55, "female", 160, 69, 130, 85, cholesterol=2)
    assert cardio.assess(loaded, row) == cardio.assess(bundle, row)


# -------------------------------------------------------------------------------------- patient

@pytest.mark.parametrize("age", [12, 250, float("nan"), True])
def test_patient_rejects_ages_outside_the_training_range(age):
    with pytest.raises(ValueError, match="age"):
        cardio.patient(age, "male", 175, 80, 120, 80)


def test_patient_rejects_non_numbers():
    with pytest.raises(ValueError, match="weight"):
        cardio.patient(50, "male", 175, "80", 120, 80)


# ---------------------------------------------------------------------------- reliability tools

def test_drop_group_importance_finds_the_planted_signal():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = synthetic_patients(1500)
    table = cardio.drop_group_importance(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                         train, n_jobs=1)
    assert table["AUC lost"].idxmax() == "blood pressure"
    assert table.loc["blood pressure", "AUC lost"] > 0.05


def test_compare_models_reports_the_sample_sd_for_every_row():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    train = synthetic_patients()
    specs = {"Logistic regression": (LogisticRegression(max_iter=2000), {"C": [1.0]})}
    table, _ = cardio.compare_models(train, specs=specs, n_jobs=1)
    aucs = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), train[cardio.FEATURES], train["cardio"],
                           cv=cardio.cv_splitter(), scoring="roc_auc")
    assert table.loc["Logistic regression", "auc_sd"] == pytest.approx(np.std(aucs, ddof=1))
    with pytest.raises(ValueError):                       # an empty spec dict is not 'run everything'
        cardio.choose_model(cardio.compare_models(train, specs={}, n_jobs=1)[0])


def test_strategy_comparison_and_excluded_patients_on_a_small_file(tmp_path):
    from sklearn.linear_model import LogisticRegression
    records = [record(i, weight=50 + i % 40, age=15000 + 7 * i, ap_hi=110 + i % 50, cardio_=int(i % 50 > 25))
               for i in range(400)]
    for i in range(0, 400, 20):                           # some repairable and some hopeless records
        records[i]["ap_lo"] = 1000 if i % 40 else 0
    path = tmp_path / "cardio.csv"
    frame(*records).to_csv(path, index=False)
    table = cardio.strategy_comparison({"lr": LogisticRegression(max_iter=3000)}, path, n_boot=50)
    assert set(table["strategy"]) == set(cardio.STRATEGIES)
    assert (table.loc[table["strategy"] == "drop", "95% CI"] == "-").all()
    train, _ = cardio.train_test("drop", path)
    model = LogisticRegression(max_iter=3000).fit(train[cardio.FEATURES], train["cardio"])
    ex = cardio.excluded_test_patients(model, path)
    removed = ex.loc["removed as implausible by 'drop'", "patients"]
    recovered = ex.loc["  of which recovered by 'repair', scored after repair", "patients"]
    assert 0 < recovered <= removed
    assert ex.loc["valid + recovered", "patients"] == ex.iloc[0]["patients"] + recovered
