import math

import pandas as pd
import pytest

import cardio


def record(id_, ap_hi=120, ap_lo=80, height=170, weight=70, cardio_=0, age=18000, gender=1):
    return {"id": id_, "age": age, "gender": gender, "height": height, "weight": weight,
            "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0,
            "active": 1, "cardio": cardio_}


def frame(*records):
    return pd.DataFrame(list(records))


@pytest.mark.parametrize("value, lo, hi, expected, rule", [
    (120, 70, 250, 120, None),             # already plausible
    (-120, 70, 250, 120, "sign"),
    (12, 70, 250, 120, "cmHg"),
    (8, 40, 150, 80, "cmHg"),
    (1000, 40, 150, 100, "extra digits"),
    (14020, 70, 250, 140, "extra digits"),
    (906, 70, 250, 90, "extra digits"),
    (8099, 40, 150, 80, "extra digits"),
])
def test_repair_pressure_rules(value, lo, hi, expected, rule):
    assert cardio.repair_pressure(value, lo, hi) == (expected, rule)


@pytest.mark.parametrize("value, lo, hi", [
    (0, 40, 150),      # missing, recorded as zero
    (1, 70, 250),      # x10 is still implausible
    (20, 40, 150),     # x10 = 200 is above the diastolic range
    (2088, 40, 150),   # 208 is too high, 20 too low
    (309, 70, 250),    # 30 is too low
])
def test_repair_pressure_gives_up(value, lo, hi):
    after, rule = cardio.repair_pressure(value, lo, hi)
    assert math.isnan(after) and rule is None


def test_deduplicate_keeps_contradictory_records():
    df = frame(record(1), record(2), record(3, cardio_=1))
    out = cardio.deduplicate(df)
    assert list(out["id"]) == [1, 3]   # 2 duplicates 1 (ignoring id); 3 differs in its diagnosis


def test_audit_flags_each_problem():
    df = frame(record(1), record(2, ap_hi=16020), record(3, ap_hi=80, ap_lo=120),
               record(4, height=55), record(5, height=120, weight=110))
    flags = cardio.audit(df)
    assert not flags.iloc[0].any()
    assert flags.loc[1, "ap_hi out of range"]
    assert flags.loc[2, "diastolic >= systolic"]
    assert flags.loc[3, "height out of range"]
    assert flags.loc[4, "BMI out of range"] and not flags.loc[4, "height out of range"]


def test_repair_swaps_transposed_readings_and_logs_every_change():
    df = frame(record(1, ap_hi=80, ap_lo=120), record(2, ap_hi=150, ap_lo=1000), record(3, ap_lo=0))
    out, log = cardio.repair(df)
    assert out.loc[0, ["ap_hi", "ap_lo"]].tolist() == [120, 80]
    assert out.loc[1, ["ap_hi", "ap_lo"]].tolist() == [150, 100]
    assert math.isnan(out.loc[2, "ap_lo"])
    assert sorted(log["rule"]) == ["extra digits", "swapped", "unrecoverable"]


def test_strategies():
    df = frame(record(1), record(2), record(3, ap_lo=1100), record(4, ap_lo=0), record(5, height=60))
    raw, _ = cardio.clean(df, "raw")
    drop, _ = cardio.clean(df, "drop")
    repair, _ = cardio.clean(df, "repair")
    assert list(raw["id"]) == [1, 3, 4, 5]
    assert list(drop["id"]) == [1]
    assert list(repair["id"]) == [1, 3] and repair.loc[1, "ap_lo"] == 110
    assert repair["ap_hi"].dtype.kind == "i"
    with pytest.raises(ValueError):
        cardio.clean(df, "impute")


def test_summary_counts_only_repairs_that_were_kept():
    # record 2's pressure is repaired but its height is impossible, so it is removed anyway
    df = frame(record(1, ap_lo=1000), record(2, ap_lo=1000, height=60))
    summary = cardio.cleaning_summary(df)
    assert summary.loc["repair", "records repaired"] == 1
    assert summary.loc["repair", "records"] == 1


def test_prepare_and_features():
    df = frame(record(1, gender=2, age=36525, height=200, weight=80, ap_hi=135, ap_lo=85))
    out = cardio.add_features(cardio.prepare(df))
    row = out.iloc[0]
    assert row["male"] == 1 and row["age_years"] == pytest.approx(100)
    assert row["bmi"] == pytest.approx(20)
    assert row["pulse_pressure"] == 50
    assert row["mean_arterial_pressure"] == pytest.approx(305 / 3)
    assert row["bp_stage"] == 2
    assert set(cardio.FEATURES) <= set(out.columns)


@pytest.mark.parametrize("sys_, dia, stage", [
    (110, 70, 0), (125, 75, 1), (125, 85, 2), (135, 70, 2), (145, 70, 3), (110, 95, 3),
])
def test_bp_stage(sys_, dia, stage):
    assert cardio.bp_stage(pd.Series([sys_]), pd.Series([dia])).iloc[0] == stage


def test_split_is_stratified_and_reproducible():
    df = frame(*[record(i, weight=50 + i, cardio_=i % 2) for i in range(100)])
    train, test = cardio.split_ids(df)
    assert len(test) == 20 and not train & test
    assert sum(i % 2 for i in test) == 10
    assert cardio.split_ids(df) == (train, test)


def test_train_test_uses_the_same_patients_for_every_strategy(tmp_path):
    records = [record(i, weight=50 + i % 40, age=15000 + i, cardio_=i % 2) for i in range(200)]
    records[3]["ap_lo"] = 1000      # repairable
    records[8]["height"] = 60       # implausible, no repair
    path = tmp_path / "cardio.csv"
    frame(*records).to_csv(path, index=False)
    train_d, test_d = cardio.train_test("drop", path)
    train_r, test_r = cardio.train_test("repair", path)
    assert not set(train_d["id"]) & set(test_d["id"])
    assert set(test_d["id"]) <= set(test_r["id"]) and set(train_d["id"]) <= set(train_r["id"])
    assert 3 in set(train_r["id"]) | set(test_r["id"]) and 3 not in set(train_d["id"]) | set(test_d["id"])
    assert 8 not in set(train_r["id"]) | set(test_r["id"])
    assert set(cardio.FEATURES) <= set(train_d.columns)


# ------------------------------------------------------------------------------------------ models

def synthetic_patients(n=600, seed=0):
    """Prepared patients whose diagnosis depends on systolic pressure and age, plus noise."""
    import numpy as np
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": range(n), "age": rng.integers(40 * 365, 65 * 365, n), "gender": rng.integers(1, 3, n),
        "height": rng.normal(165, 8, n).round(), "weight": rng.normal(74, 12, n).round(),
        "ap_hi": rng.normal(128, 16, n).round().astype(int), "cholesterol": rng.integers(1, 4, n),
        "gluc": rng.integers(1, 4, n), "smoke": rng.integers(0, 2, n), "alco": rng.integers(0, 2, n),
        "active": rng.integers(0, 2, n)})
    df["ap_lo"] = (df["ap_hi"] * 0.65).round().astype(int)
    logit = 0.08 * (df["ap_hi"] - 128) + 0.0003 * (df["age"] - 52 * 365)
    df["cardio"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return cardio.add_features(cardio.prepare(df))


def test_framingham_risk_matches_the_published_equation():
    df = pd.DataFrame({"age_years": [50.0, 60.0], "bmi": [25.0, 30.0], "ap_hi": [120, 150],
                       "smoke": [0, 1], "gluc": [1, 3], "male": [1, 0]})
    # by hand: man 50, BMI 25, 120 mmHg -> 8.48%; woman 60, BMI 30, 150 mmHg, smoker, diabetic -> 42.86%
    assert cardio.framingham_risk(df).round(4).tolist() == [0.0848, 0.4286]


def test_classification_metrics():
    m = cardio.classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert m == {"auc": 0.75, "accuracy": 0.5, "sensitivity": 0.5, "specificity": 0.5}


def test_accuracy_threshold_finds_the_separating_cut():
    assert 3 < cardio.accuracy_threshold([0, 0, 0, 1, 1, 1], [1, 2, 3, 7, 8, 9]) <= 7


def test_compare_models_scores_baselines_and_models_on_the_same_folds():
    from sklearn.linear_model import LogisticRegression
    train = synthetic_patients()
    specs = {"Logistic regression": (LogisticRegression(max_iter=2000), {"C": [0.1, 1.0]})}
    table, searches = cardio.compare_models(train, specs=specs, n_jobs=1)
    assert list(table["kind"]).count("clinical baseline") == 3
    assert table.loc["Systolic pressure alone", "auc"] > 0.7          # the planted signal is found
    assert table.loc["Logistic regression", "auc"] > 0.7
    assert set(searches) == {"Logistic regression"}
    assert cardio.choose_model(table) == "Logistic regression"


@pytest.mark.parametrize("lr_auc, expected", [
    (0.802, "Logistic regression"),   # 0.001 behind: within one standard error (0.005 / sqrt 5 = 0.0022)
    (0.800, "Gradient boosting"),     # 0.003 behind: a real difference
])
def test_choose_model_prefers_the_simpler_of_tied_models(lr_auc, expected):
    table = pd.DataFrame({"kind": ["machine learning"] * 3 + ["clinical baseline"],
                          "auc": [lr_auc, 0.803, 0.790, 0.9], "auc_sd": [0.005] * 4},
                         index=["Logistic regression", "Gradient boosting", "Random forest", "Rule"])
    assert cardio.choose_model(table) == expected


# ------------------------------------------------------------------------------------- evaluation

def test_bootstrap_resamples_all_scores_together():
    from sklearn.metrics import roc_auc_score
    y = [0, 1] * 50
    boot = cardio.bootstrap(y, {"perfect": y, "flipped": [1 - v for v in y]}, roc_auc_score, n=50)
    assert len(boot) == 50 and (boot["perfect"] == 1).all() and (boot["flipped"] == 0).all()
    lo, hi = cardio.interval(pd.Series(range(101)))
    assert (lo, hi) == pytest.approx((2.5, 97.5))


def test_screening_threshold_reaches_the_sensitivity():
    import numpy as np
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    risk = rng.random(1000) * 0.5 + 0.5 * y
    t = cardio.screening_threshold(y, risk, 0.8)
    assert (risk[y == 1] >= t).mean() >= 0.8


def test_points_score_is_a_whole_point_scorecard_that_ranks_patients():
    from sklearn.metrics import roc_auc_score
    train = synthetic_patients(2000)
    score = cardio.PointsScore().fit(train, train["cardio"])
    table = score.table()
    assert table["points"].dtype.kind == "i"
    assert set(table["factor"]) == set(cardio.POINTS_LABELS.values())
    sbp = table[table["factor"] == "Systolic pressure (mmHg)"]["points"].tolist()
    assert sbp[0] == 0 and sbp[-1] > 5                       # the planted effect earns points
    assert roc_auc_score(train["cardio"], score.predict_proba(train)) > 0.7
    risks = score.risk_by_total([0, 10, 20])
    assert risks.is_monotonic_increasing


def test_explanations_find_the_planted_signal():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = synthetic_patients(2000)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(train[cardio.FEATURES], train["cardio"])
    imp = cardio.grouped_permutation_importance(model, train, n_repeats=3)
    assert imp.index[0] == "blood pressure" and imp.loc["blood pressure", "auc drop"] > 0.1
    pd_curve = cardio.partial_dependence_curve(model, train, "ap_hi", [110, 130, 150])
    assert pd_curve.is_monotonic_increasing


# ------------------------------------------------------------------------- reliability and use

def test_smote_creates_points_between_same_class_neighbours():
    import numpy as np
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0], [11.0, 10.0], [10.0, 11.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    sx, sy = cardio.smote(X, y, n_new=20, k=2)
    assert len(sx) == 40 and (sy == 1).sum() == 20
    assert (sx[sy == 0] <= 1).all() and (sx[sy == 1] >= 10).all()


def test_ppv_at_prevalence():
    assert cardio.ppv_at_prevalence(0.8, 0.8, 0.5) == pytest.approx(0.8)
    assert cardio.ppv_at_prevalence(0.8, 0.8, 0.1) == pytest.approx(0.08 / (0.08 + 0.18))


def test_subgroup_table_covers_sex_and_age():
    train = synthetic_patients(800)
    table = cardio.subgroup_table(train, train["ap_hi"] / 200)
    assert set(table.index) == {"women", "men", "age < 50", "age 50-59", "age 60+", "everyone"}
    assert table.loc["women", "patients"] + table.loc["men", "patients"] == table.loc["everyone", "patients"]


def test_patient_checks_plausibility_and_builds_features():
    row = cardio.patient(55, "male", 175, 90, 150, 95, cholesterol=2)
    assert row.loc[0, "male"] == 1 and row.loc[0, "age_years"] == pytest.approx(55)
    assert row.loc[0, "bp_stage"] == 3 and set(cardio.FEATURES) <= set(row.columns)
    with pytest.raises(ValueError, match="ap_hi"):
        cardio.patient(55, "male", 175, 90, 1500, 95)
    with pytest.raises(ValueError, match="diastolic >= systolic"):
        cardio.patient(55, "female", 165, 60, 80, 120)
    with pytest.raises(ValueError, match="sex"):
        cardio.patient(55, "m", 175, 90, 120, 80)


def test_assess_a_patient_with_a_small_final_bundle(tmp_path):
    from sklearn.linear_model import LogisticRegression
    train = synthetic_patients(1500)
    model = LogisticRegression(max_iter=3000).fit(train[cardio.FEATURES], train["cardio"])
    bundle = cardio.build_final(train, model=model)
    path = cardio.save_final(bundle, tmp_path / "model.joblib")
    loaded = cardio.load_final(path)
    high = cardio.assess(loaded, cardio.patient(62, "male", 170, 95, 170, 100, cholesterol=3))
    low = cardio.assess(loaded, cardio.patient(42, "female", 165, 58, 110, 70))
    assert high["risk"] > low["risk"] and high["points"] > low["points"]
    assert high["verdict"] == "cardiovascular disease likely"
    assert 0 < loaded["screening_cutoff"] < 1


def test_points_score_with_a_subset_of_factors():
    train = synthetic_patients(1500)
    score = cardio.PointsScore(factors=["age_years", "ap_hi"]).fit(train, train["cardio"])
    assert set(score.table()["factor"]) == {"Age (years)", "Systolic pressure (mmHg)"}
    assert len(score.predict_proba(train)) == len(train)
