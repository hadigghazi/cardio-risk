# Predicting Cardiovascular Disease from a Routine Medical Examination

**Hadi Ghazi**, Lebanese University

*Topic 2 of the project list: Patient's Sickness Prediction (`cardio.csv`).*

## 1. Introduction

Cardiovascular diseases (CVDs) kill an estimated 17.9 million people every year, more than any other
cause. Many of these deaths could be delayed by finding people at risk early, before the first heart
attack or stroke. That makes screening a natural task for machine learning. A model takes the few
numbers recorded at a routine check-up (age, sex, height, weight, blood pressure, cholesterol,
glucose and three lifestyle questions) and estimates whether the patient has a cardiovascular disease.

The dataset has 70,000 patients, which is large for a medical tabular dataset. Anyone who looks
closely meets two facts at once. First, the raw file contains impossible measurements, such as blood
pressures of −150 or 16,020 mmHg. Second, the predictive signal is modest: models reported for this dataset
typically reach about 73% accuracy. A clinician needs no model to see that high blood pressure is the main
risk. So the useful question is not "which algorithm scores highest" but what a model adds, and when
its numbers can be trusted.

### Research questions

These questions, and the evaluation protocol in Section 2, were fixed before any model was trained:

- **RQ1. What does machine learning add over a clinical rule?** We compare increasingly flexible
  models against a single threshold doctors already use: systolic pressure of at least 130 mmHg,
  which is stage 1 hypertension in the ACC/AHA 2017 guideline.
- **RQ2. How much do data-cleaning choices matter?** We train on the data as recorded, with
  implausible records removed, and with recognisable entry errors repaired. All three are scored on
  the same test patients.
- **RQ3. Can the predictions be trusted for individual patients?** We ask which factors drive them,
  whether predicted risks match observed frequencies (calibration), and whether errors are evenly
  spread across sexes and age groups.

```python
import sys
from pathlib import Path

# the notebook may run from the project folder or from outputs/; cardio.py sits in the project folder
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "cardio.py").exists())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cardio

FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.precision", 3)

# One look for every figure: thin marks, recessive axes, a fixed colour for each class
NO_CVD, CVD, FLAG, INK, MUTED = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9.5, "legend.frameon": False,
})
```

## 2. Evaluation protocol

The protocol is fixed before modelling, so that no result can steer the choices behind it.

1. **Exact duplicates are removed first** (24 records that repeat another record in every column but
   `id`). If a pair were split between training and test data, the model would be tested on a
   patient it had already seen.
2. **One stratified train/test split**: 80% train, 20% test, seed 42. It is drawn once from the
   deduplicated raw data. Every cleaning strategy uses the same patient ids, so all models are
   judged on the same patients (RQ2).
3. Cleaning applies fixed per-record rules and learns nothing from the data. So it cannot leak test
   information into training, whether it runs before or after the split. Anything that *is* learned
   (scaling, imputation, model parameters) is fitted inside cross-validation folds on training data.
4. **Model selection uses 5-fold stratified cross-validation on the training set only.** The test
   set is scored once per final model.
5. **Metrics:** ROC AUC (ranking quality, independent of a threshold) is the primary metric.
   Accuracy, sensitivity and specificity are reported at stated thresholds, and the Brier score and
   calibration curves cover RQ3. Each test metric gets a 95% bootstrap confidence interval, because
   differences of a few tenths of a point are within noise at this test-set size.

## 3. The data

```python
raw = cardio.load()
print(f"{len(raw):,} records, {raw.shape[1]} columns")
raw.head()
```

| Column | Meaning | Type |
|---|---|---|
| `age` | age in days | integer, objective |
| `gender` | 1 or 2 (a code, see below) | categorical, objective |
| `height`, `weight` | cm, kg | numeric, objective |
| `ap_hi`, `ap_lo` | systolic and diastolic blood pressure, mmHg | numeric, measured at the exam |
| `cholesterol`, `gluc` | 1 normal, 2 above normal, 3 well above normal | ordinal, lab result |
| `smoke`, `alco`, `active` | smoking, alcohol intake, physical activity | binary, **self-reported** |
| `cardio` | presence of cardiovascular disease | binary target |

```python
raw.describe().T
```

```python
print("Target balance:", raw["cardio"].value_counts(normalize=True).round(3).to_dict())
print("Missing values:", int(raw.isna().sum().sum()))
print("Mean height by gender code:", raw.groupby("gender")["height"].mean().round(1).to_dict())
```

The classes are almost perfectly balanced and no cell is empty, so the problems hide in the values
themselves. The documentation never says which gender code is which. Code 2 is about 8.5 cm taller
on average, the usual adult sex difference, so code 2 is taken as male. Age is in days and is
converted to years. Ages run from 30 to 65, so this is a middle-aged adult population.

## 4. Data quality

### 4.1 Plausibility audit

Each record is checked against deliberately wide physiological bounds. They are meant to catch values
that cannot be a measurement of a living adult, not patients who are merely unusual:

```python
pd.DataFrame(
    [(col, lo, hi) for col, (lo, hi) in cardio.BOUNDS.items()] + [("BMI", *cardio.BMI_BOUNDS)],
    columns=["quantity", "lower bound", "upper bound"],
).set_index("quantity")
```

```python
deduped = cardio.deduplicate(raw)
flags = cardio.audit(deduped)
print(f"{len(raw) - len(deduped)} exact duplicates removed")
print(f"{flags.any(axis=1).sum():,} records ({flags.any(axis=1).mean():.1%}) fail at least one check\n")
flags.sum().rename("records failing").to_frame()
```

About 2% of records fail, and nearly all failures are blood pressures. They are not random noise.
The most frequent impossible values show clear, repeated patterns:

```python
def top_values(col, n=12):
    lo, hi = cardio.BOUNDS[col]
    bad = deduped.loc[~deduped[col].between(lo, hi), col]
    return bad.value_counts().head(n).rename_axis(col).rename("records").to_frame().T

display(top_values("ap_hi"))
display(top_values("ap_lo"))
```

- **Extra digits.** A diastolic pressure of 1000 or 1100 (over 800 records) is 100 or 110 with an extra
  zero. A systolic pressure of 14020 is 140 with stray digits.
- **The wrong unit.** A systolic pressure of 12, 14 or 11 is 120, 140 or 110 written in centimetres
  of mercury, the unit many people use when they say "twelve over eight".
- **A stray minus sign.** −120 is 120.
- **Swapped fields.** Pairs like 80/120 have systolic and diastolic entered the wrong way round.

```python
pairs = deduped.loc[flags["diastolic >= systolic"], ["ap_hi", "ap_lo"]].value_counts().head(8)
pairs.rename("records").to_frame()
```

The figure shows the same thing. Every point is one patient's blood-pressure pair. On the left, the
raw values on a logarithmic scale, with records that fail a check highlighted. On the right, the same
patients after repair.

```python
repaired, log = cardio.clean(raw, "repair")
bad = flags.any(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
for mask, colour, label, size in [(~bad, NO_CVD, "passes every check", 4), (bad, FLAG, "fails a check", 9)]:
    d = deduped[mask]
    ax.scatter(d["ap_hi"].abs().clip(lower=1), d["ap_lo"].abs().clip(lower=1), s=size,
               color=colour, alpha=0.35 if colour == NO_CVD else 0.7, linewidths=0, label=f"{label} ({mask.sum():,})")
ax.set(xscale="log", yscale="log", xlabel="systolic, mmHg (absolute value, log scale; 0 drawn at 1)",
       ylabel="diastolic, mmHg (log scale; 0 drawn at 1)", title="As recorded")
ax.plot([1, 2e4], [1, 2e4], color=MUTED, lw=0.8, ls="--")
ax.annotate("diastolic = systolic", (2.2e3, 3.3e3), color=MUTED, fontsize=8, rotation=38)
ax.legend(loc="upper left", markerscale=2.5)

ax = axes[1]
ax.scatter(repaired["ap_hi"], repaired["ap_lo"], s=4, color=NO_CVD, alpha=0.35, linewidths=0)
fixed = repaired["id"].isin(log.loc[log["rule"] != "unrecoverable", "id"])
ax.scatter(repaired.loc[fixed, "ap_hi"], repaired.loc[fixed, "ap_lo"], s=9, color=CVD, alpha=0.7,
           linewidths=0, label=f"repaired ({fixed.sum():,})")
ax.set(xlabel="systolic, mmHg", ylabel="diastolic, mmHg", title="After repair")
ax.legend(loc="upper left", markerscale=2.5)
fig.suptitle("Blood pressure: about 2% of records hold values no patient can have", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig1_blood_pressure_cleaning.png")
plt.show()
```

The contradictory records are a smaller issue. Some records share every feature value but carry
different diagnoses:

```python
features_only = deduped.drop(columns=["id", "cardio"])
groups = deduped.groupby(list(features_only.columns))["cardio"].nunique()
print(f"{(groups > 1).sum()} groups of records identical in every feature but with different diagnoses")
```

These are kept. They are a reminder that the features do not determine the diagnosis: no model can
classify all of these records correctly.

### 4.2 Three cleaning strategies

Each strategy becomes its own training set (RQ2):

```python
summary = cardio.cleaning_summary(raw)
summary.style.format({"share kept": "{:.1%}", "records": "{:,}"})
```

`repair` corrects a blood pressure only when a recognisable entry error explains it. It turns the
recorded value back into a plausible one with the rules shown above, applied in a fixed order: sign,
then unit, then extra digits, then swapped fields. Anything no rule recovers is removed, and so are
implausible heights and weights, whose errors show no pattern that would identify the true value.
Every change is logged:

```python
print(log.groupby(["field", "rule"]).size().rename("values").to_string())
log[log["rule"] != "unrecoverable"].groupby("rule").head(3).sort_values("rule")
```

The repairs are guesses, however well motivated. A recorded 906 may be 90 with a stray digit, or
96. This is why `repair` is compared against `drop`, which invents nothing, rather than silently
replacing it.

### 4.3 What cleaning changes

```python
rows = []
for s in cardio.STRATEGIES:
    d, _ = cardio.clean(raw, s)
    rows.append({"strategy": s, "records": len(d),
                 "corr(systolic, cardio)": d["ap_hi"].corr(d["cardio"]),
                 "corr(diastolic, cardio)": d["ap_lo"].corr(d["cardio"]),
                 "mean systolic": d["ap_hi"].mean(), "max systolic": d["ap_hi"].max()})
pd.DataFrame(rows).set_index("strategy")
```

This is the most important number in the cleaning section. In the raw data, systolic blood pressure
has almost no linear relationship with the diagnosis (r = 0.05). After either kind of cleaning it is
the strongest single predictor (r = 0.43). Fewer than 2% of records cause this. A handful of values
in the thousands dominate the variance and hide everything else. Anything linear, such as a
correlation or a logistic regression on unscaled values, reads the raw data wrongly. Tree models
split on ranks and should suffer far less. The modelling section tests whether that holds.

### 4.4 Derived features

Four clinically standard quantities are added to the eleven recorded ones:

| Feature | Definition | Why |
|---|---|---|
| `bmi` | weight / height² (kg/m²) | the usual measure of excess weight; height and weight alone do not express it |
| `pulse_pressure` | systolic − diastolic | a marker of arterial stiffness |
| `mean_arterial_pressure` | (systolic + 2 × diastolic) / 3 | average pressure over the cardiac cycle |
| `bp_stage` | ACC/AHA 2017: 0 normal, 1 elevated, 2 stage 1, 3 stage 2 | the categories clinicians act on |

```python
data = cardio.dataset("drop")
rate = data.groupby("bp_stage")["cardio"].agg(["mean", "size"])

fig, ax = plt.subplots(figsize=(6.4, 3.4))
bars = ax.bar(range(4), rate["mean"], width=0.62, color=CVD)
for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
    ax.text(i, m + 0.015, f"{m:.0%}", ha="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.text(i, 0.02, f"n = {n:,}", ha="center", color="white", fontsize=8)
ax.set_xticks(range(4), cardio.BP_STAGE_NAMES)
ax.set(ylim=(0, 1), ylabel="share with cardiovascular disease", xlabel="ACC/AHA 2017 blood-pressure category",
       title="Disease rate climbs with every blood-pressure category")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax.grid(axis="x", visible=False)
fig.savefig(FIGURES / "fig2_disease_by_bp_stage.png")
plt.show()
```

The large stage 1 group follows from the guideline, not from an error. The most common reading in
the data, 120/80, is stage 1 because a diastolic pressure of 80 mmHg starts stage 1 in the 2017
guideline.

After cleaning, the data tells the clinically expected story. The disease rate rises steadily from
normal blood pressure to stage 2 hypertension, where most patients are diagnosed with CVD. That
check is the main reason to trust the cleaned data more than the raw file.
