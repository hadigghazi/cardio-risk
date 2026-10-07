# Patient's Sickness Prediction: Detecting Cardiovascular Disease from a Routine Medical Examination

**Hadi Ghazi**, Lebanese University

*Project topic 2: Patient's Sickness Prediction, using the `cardio` dataset.*

## 1. Introduction

Cardiovascular diseases (CVDs) kill an estimated 17.9 million people every year, more than any other
cause. Many of these deaths could be delayed if patients at risk were identified early. Doctors find
this hard to do reliably by hand: no single measurement settles the question, and the risk builds up
from many small factors.

**Objective.** Build a model that uses the values collected at a routine medical examination (age,
sex, height, weight, blood pressure, cholesterol, glucose, and three lifestyle questions) to tell
whether a patient has a cardiovascular disease. Then evaluate how well it does this. The final model
is delivered with a function that gives a verdict and a risk for a new patient (Section 10).

The dataset has 70,000 patients, which is large for medical tabular data, but it is not clean. Some
recorded blood pressures are impossible, such as −150 or 16,020 mmHg. The signal is also modest:
careful studies of this dataset report about 73% accuracy. Other published papers claim 86–94% on the
same data, which is hard to reconcile with that. Three questions therefore guide the evaluation, so
that the final model comes with an honest account of what it can do:

- **Q1. How much does machine learning improve on what a doctor already has?** The models are
  compared with two clinical baselines. The first is a blood-pressure threshold from the ACC/AHA 2017
  guideline: systolic pressure of at least 130 mmHg, stage 1 hypertension. The second is the
  Framingham office-based risk score, which uses age, sex, BMI, blood pressure, smoking and diabetes.
- **Q2. How much do data-handling choices change the measured performance?** This covers the
  cleaning strategy, and the common mistakes that let information from the test set leak into
  training. Reproducing those mistakes shows how accuracies far above 73% can be produced.
- **Q3. Can the final model's predictions be trusted and explained?** This covers which factors drive
  the predictions, whether a simple points score can match the best model, whether predicted risks
  match observed frequencies, and whether errors are evenly spread across sexes and age groups.

**Structure.** Section 2 fixes the evaluation protocol before any model is trained. Section 3
describes the data, Section 4 cleans it, Section 5 explores it, and Section 6 derives features.
Section 7 trains and compares the models (Q1), Section 8 evaluates the chosen model on the test set
and explains it, and Section 9 runs the reliability checks (Q2, Q3). Section 10 shows the model in
use, and Section 11 concludes, with the study's limitations.

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

1. **Exact duplicates are removed first.** There are 24 records that repeat another record in every
   column except `id`. If a pair were split between training and test data, the model would be
   tested on a patient it had already seen.
2. **One stratified train/test split.** 80% of patients go to training and 20% to test (seed 42).
   The split is drawn once, by patient id, from the deduplicated raw data. Every later comparison
   uses the same ids. Exploratory analysis (Section 5) and every modelling decision use the training
   patients only.
3. **Cleaning learns nothing from the data.** It applies fixed per-record rules, so it cannot leak
   test information into training. Anything that *is* learned (scaling, model parameters, thresholds)
   is fitted on training data only, inside cross-validation folds during model selection.
4. **Model selection.** Models are tuned and compared with 5-fold stratified cross-validation on the
   training set. The test set is scored once, for the final comparison, and never used to choose
   anything.
5. **Common test set.** When cleaning strategies are compared, all of them are scored on the same
   test patients: those whose records pass every plausibility check. A deployed system would reject
   an impossible input at entry rather than predict from it. How each strategy handles the remaining
   records is reported separately.
6. **Metrics.** The primary metric is ROC AUC: how well the model ranks diseased patients above
   healthy ones, independent of any threshold. Accuracy, sensitivity (the share of diseased patients
   detected) and specificity (the share of healthy patients cleared) are reported at the 0.5
   threshold and at a screening threshold chosen on training data. The Brier score and calibration
   curves measure how far predicted risks can be trusted. Every test metric gets a 95% bootstrap
   confidence interval, because with about 14,000 test patients a difference of one point in
   accuracy can be noise.
7. **The leakage experiments (Q2) are kept separate.** They reproduce mistakes on purpose. They are
   labelled as such and never feed into the choice of the final model.

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
split on ranks and should suffer far less. Section 9 tests whether that holds.

## 5. Exploratory data analysis

From here on, the analysis uses the **training patients only**, cleaned with the `drop` strategy (the
choice of strategy is revisited in Section 9). The test patients stay unseen until Section 8.

```python
train, test = cardio.train_test("drop")
print(f"training patients: {len(train):,}   test patients: {len(test):,}")
print(f"share with CVD in training: {train['cardio'].mean():.1%}")
print(f"age {train['age_years'].min():.0f}-{train['age_years'].max():.0f} years, "
      f"median {train['age_years'].median():.0f}; {train['male'].mean():.0%} men")
```

### 5.1 Disease rate by risk factor

The most direct way to see what the data holds is the share of patients with CVD at each level of
each factor. The dashed line marks the overall rate.

```python
def band(series, edges, labels):
    return pd.cut(series, edges, labels=labels, right=False)

LEVELS = ["normal", "above", "well above"]
panels = [
    ("Age (years)", band(train["age_years"], [0, 40, 45, 50, 55, 60, 70],
                        ["<40", "40-44", "45-49", "50-54", "55-59", "60+"])),
    ("Systolic pressure (mmHg)", band(train["ap_hi"], [0, 120, 130, 140, 160, 400],
                                      ["<120", "120-129", "130-139", "140-159", "160+"])),
    ("BMI (kg/m²)", band(train["bmi"], [0, 18.5, 25, 30, 35, 100],
                         ["<18.5", "18.5-25", "25-30", "30-35", "35+"])),
    ("Cholesterol", pd.Categorical(train["cholesterol"].map(dict(enumerate(LEVELS, 1))), LEVELS)),
    ("Glucose", pd.Categorical(train["gluc"].map(dict(enumerate(LEVELS, 1))), LEVELS)),
]
lifestyle = [(f"{name}\n{answer}", train.loc[train[col] == value, "cardio"])
             for col, name in [("smoke", "smokes"), ("alco", "alcohol"), ("active", "active")]
             for value, answer in [(0, "no"), (1, "yes")]]
overall = train["cardio"].mean()

fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
for ax, panel in zip(axes.flat, panels + [("Lifestyle (self-reported)", None)]):
    title, groups = panel
    if groups is None:
        labels = [name for name, _ in lifestyle]
        rates = [s.mean() for _, s in lifestyle]
    else:
        g = train.groupby(pd.Series(groups, index=train.index), observed=False)["cardio"].mean()
        labels, rates = list(g.index), list(g)
    x = np.arange(len(labels))
    ax.bar(x, rates, width=0.66, color=CVD)
    for xi, r in zip(x, rates):
        ax.text(xi, r + 0.02, f"{r:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(overall, color=MUTED, lw=0.9, ls="--")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(title=title, ylim=(0, 0.9))
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
for ax in axes[:, 0]:
    ax.set_ylabel("share with CVD")
fig.suptitle("Blood pressure, age, weight and cholesterol carry the signal; the lifestyle answers barely do",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig2_disease_rate_by_factor.png")
plt.show()
```

Four factors carry most of the signal, and each rises steadily with no reversal:

- **Systolic blood pressure** is the strongest. The disease rate climbs from about a quarter of
  patients below 120 mmHg to most patients at 160 mmHg and above.
- **Age.** The rate roughly triples between patients under 40 and those over 60.
- **Weight.** The rate rises across the BMI categories, from about a quarter of underweight patients
  to about two thirds of those with a BMI of 35 or more.
- **Cholesterol** is the strongest of the lab results. Three quarters of patients with "well above
  normal" cholesterol have CVD. Glucose follows the same pattern, but more weakly.

The three lifestyle answers barely move the rate, and two point the "wrong" way: smokers and drinkers
have slightly *lower* disease rates than other patients. Only physical activity has the expected
protective sign.

### 5.2 The smoking paradox

Smoking is an established cause of cardiovascular disease, so a lower rate among smokers needs an
explanation. The first suspect is a confounder. Smoking in this data is almost entirely male, so the
comparison is redone within each sex and age group:

```python
print("share who smoke:", train.groupby("male")["smoke"].mean().rename({0: "women", 1: "men"})
      .map("{:.1%}".format).to_dict())

men = train[train["male"] == 1]
men_band = band(men["age_years"], [0, 45, 50, 55, 60, 70], ["<45", "45-49", "50-54", "55-59", "60+"])
by_age = men.groupby([men_band, "smoke"], observed=True)["cardio"].mean().unstack()
by_age.columns = ["non-smokers", "smokers"]

fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(by_age))
ax.vlines(x, by_age.min(axis=1), by_age.max(axis=1), color="#cfcec8", lw=2, zorder=1)
for col, colour in [("non-smokers", NO_CVD), ("smokers", CVD)]:
    ax.plot(x, by_age[col], "o", color=colour, ms=8, label=col, zorder=2)
ax.set_xticks(x, by_age.index)
ax.set(xlabel="age (years)", ylabel="share with CVD", title="Men: smokers have less CVD in every age group")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(loc="upper left")
fig.savefig(FIGURES / "fig3_smoking_paradox.png")
plt.show()
by_age.style.format("{:.1%}")
```

The paradox survives both checks. Among men (women hardly smoke here), smokers have a lower disease
rate than non-smokers of the same age. The likeliest explanation lies in how the data was collected,
not in biology. The diagnosis and the answers were recorded **at the same examination**, so a patient
who already has heart disease has usually been told to quit. Such a patient answers "no" to smoking,
which moves diseased former smokers into the non-smoker group. This effect has a name in
epidemiology: the "sick quitter".

This has two consequences for the rest of the study:

1. The task is **detection of an existing disease**, not prediction of a future one. A model trained
   here can say "this profile looks like the profiles of patients who have CVD". It cannot say "this
   person will develop CVD".
2. Feature effects learned from this data are **associations, not causes**. A model that gives
   smoking a negative weight is describing the data correctly, but no one should be told that smoking
   protects the heart. Section 8 returns to this when the model is explained.

### 5.3 Correlations between features

```python
cols = cardio.BASE_FEATURES + ["bmi", "cardio"]
corr = train[cols].corr()

fig, ax = plt.subplots(figsize=(7.4, 6.2))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
ax.set_yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.iloc[i, j]
        if i != j and abs(v) >= 0.1:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
ax.grid(False)
ax.set_title("Pearson correlation (training patients)")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(FIGURES / "fig4_correlations.png")
plt.show()
corr["cardio"].drop("cardio").sort_values(ascending=False).rename("correlation with CVD").to_frame()
```

The last column restates Section 5.1 as single numbers. Systolic pressure, diastolic pressure, age,
cholesterol and BMI lead, while sex and the lifestyle answers are close to zero. Among the features
themselves, the two blood pressures are strongly correlated (about 0.7), and so are weight and BMI,
and sex and height. For tree models this hardly matters. For logistic regression it means individual
coefficients must be read with care, since correlated features share their effect.

### 5.4 What the exploration implies for modelling

- The strongest relationships are **monotonic**: the risk rises steadily with blood pressure, age,
  BMI and cholesterol. A linear model on suitable features should capture most of the signal, so
  flexible models have limited room to improve on it. This is the expectation Q1 will test.
- **No class imbalance**: 49.5% of training patients have CVD. Oversampling methods such as SMOTE
  have nothing to fix here. In Section 9 they appear only as one of the mistakes the leakage
  experiments reproduce.
- The lifestyle answers are weak and partly distorted by reverse causation. They are kept, because a
  model should use what the examination records, but nothing causal is read into them.

## 6. Feature engineering

Four clinically standard quantities are added to the eleven recorded ones:

| Feature | Definition | Why |
|---|---|---|
| `bmi` | weight / height² (kg/m²) | the usual measure of excess weight; height and weight alone do not express it |
| `pulse_pressure` | systolic − diastolic | a marker of arterial stiffness |
| `mean_arterial_pressure` | (systolic + 2 × diastolic) / 3 | average pressure over the cardiac cycle |
| `bp_stage` | ACC/AHA 2017: 0 normal, 1 elevated, 2 stage 1, 3 stage 2 | the categories clinicians act on |

```python
rate = train.groupby("bp_stage")["cardio"].agg(["mean", "size"])

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.bar(range(4), rate["mean"], width=0.62, color=CVD)
for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
    ax.text(i, m + 0.015, f"{m:.0%}", ha="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.text(i, 0.02, f"n = {n:,}", ha="center", color="white", fontsize=8)
ax.set_xticks(range(4), cardio.BP_STAGE_NAMES)
ax.set(ylim=(0, 1), ylabel="share with cardiovascular disease", xlabel="ACC/AHA 2017 blood-pressure category",
       title="Disease rate climbs with every blood-pressure category")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax.grid(axis="x", visible=False)
fig.savefig(FIGURES / "fig5_disease_by_bp_stage.png")
plt.show()
```

The large stage 1 group follows from the guideline, not from an error. The most common reading in
the data, 120/80, is stage 1 because a diastolic pressure of 80 mmHg starts stage 1 in the 2017
guideline.

The derived category summarises both pressures in the terms clinicians act on, and the disease rate
rises steadily across it: from about one patient in five at normal pressure to most patients at
stage 2.

Two points about these features:

- **They are fixed formulas, not learned transformations.** Like the cleaning rules, they cannot
  leak test information.
- **They overlap with the recorded values on purpose.** BMI is a function of height and weight, and
  the three pressure features are functions of the two readings. Tree models gain from having the
  combination precomputed, since a split on BMI replaces many splits on height and weight. For
  logistic regression the overlap is handled by regularisation. Section 7 checks whether the derived
  features help at all, by training the same models with and without them.
## 7. Models

### 7.1 What is compared

Every candidate is evaluated with 5-fold cross-validation on the training patients, using the same
five folds. The test patients are not touched in this section.

| Candidate | Type | What it uses | Settings searched |
|---|---|---|---|
| Rule: systolic ≥ 130 mmHg | clinical baseline | one reading, guideline threshold | none |
| Systolic pressure alone | clinical baseline | one reading; cut-off chosen on the training folds | none |
| Framingham (published) | clinical baseline | age, sex, BMI, systolic pressure, smoking, diabetes | none: the 2008 equations as published |
| Framingham form (refit) | logistic regression | the same terms, one equation per sex | coefficients refitted on these patients |
| Logistic regression | linear model | all 15 features, standardised | regularisation strength C |
| Random forest | tree ensemble | all 15 features | leaf size, features per split |
| Gradient boosting | tree ensemble (histogram GBDT) | all 15 features | learning rate, tree size, leaf size |
| Neural network (MLP) | multilayer perceptron | all 15 features, standardised | layer sizes, weight decay |

Some notes on the baselines:

- **The Framingham score answers a different question.** It was built to predict a *first*
  cardiovascular event over the next ten years in people free of CVD, while this dataset records
  existing disease. The dataset also has no field for blood-pressure treatment, so the equation for
  untreated patients is used, and no diabetes field, so glucose "well above normal" stands in for
  diabetes. Using its risk to rank these patients is therefore a reference point, not a fair contest.
  The refitted version keeps Framingham's functional form but learns the weights from these
  patients. It shows how much of any gap comes from the form and how much from the population.
- **Two baselines need a cut-off.** Systolic pressure alone and the published Framingham risk are
  continuous scores, so the cut-off that classifies the four training folds most accurately is
  applied to the fifth. Their AUC needs no cut-off.

**Selection rule, fixed in advance.** The final model is the *simplest* machine-learning model whose
mean cross-validated AUC is within one standard error of the best one. The order of simplicity is:
Framingham form, logistic regression, gradient boosting, random forest, neural network. A difference
smaller than the fold-to-fold noise is not evidence, so it should not buy complexity.

### 7.2 Cross-validated comparison

```python
import time

t0 = time.time()
comparison, searches = cardio.compare_models(train)
print(f"{len(searches)} grid searches and 3 baselines, {cardio.CV_FOLDS}-fold CV: {time.time() - t0:.0f} s")
(comparison.sort_values("auc", ascending=False)
 .style.format({"auc": "{:.4f}", "auc_sd": "{:.4f}", "accuracy": "{:.1%}", "sensitivity": "{:.1%}",
                "specificity": "{:.1%}", "fit seconds": "{:.1f}"}))
```

```python
order = comparison.sort_values("auc")
fig, ax = plt.subplots(figsize=(8.2, 3.9))
y = np.arange(len(order))
colours = [NO_CVD if k == "machine learning" else MUTED for k in order["kind"]]
ax.errorbar(order["auc"], y, xerr=order["auc_sd"], fmt="none", ecolor="#b5b4ae", elinewidth=1.5, capsize=3)
ax.scatter(order["auc"], y, s=60, color=colours, zorder=3)
for yi, (auc, acc) in enumerate(zip(order["auc"], order["accuracy"])):
    ax.text(auc + 0.006, yi, f"AUC {auc:.3f}  ·  accuracy {acc:.1%}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y, order.index)
ax.set(xlabel="cross-validated ROC AUC (mean ± SD over 5 folds)", xlim=(order["auc"].min() - 0.03, 0.86),
       title="Clinical baselines (grey) and machine-learning models (blue)")
ax.grid(axis="y", visible=False)
fig.savefig(FIGURES / "fig6_model_comparison.png")
plt.show()
```

### 7.3 Do the derived features help?

Section 6 added BMI, pulse pressure, mean arterial pressure and the blood-pressure stage. Here the
best settings of logistic regression and gradient boosting are cross-validated again, with and
without them:

```python
features_effect = cardio.feature_set_comparison(train, searches, ["Logistic regression", "Gradient boosting"])
features_effect["gain"] = features_effect["recorded + derived"] - features_effect["recorded only"]
features_effect.style.format("{:.4f}")
```

### 7.4 Would more data help?

The same models are trained on growing shares of a training fold. If the curve is still rising at
the full training set, more patients would help. If it is flat, the limit lies in the features.

```python
curve = cardio.learning_curve_auc(train, {name: searches[name].best_estimator_
                                          for name in ["Logistic regression", "Gradient boosting"]})
fig, ax = plt.subplots(figsize=(6.6, 3.5))
for (name, g), colour in zip(curve.groupby("model"), [CVD, NO_CVD]):
    ax.plot(g["patients"], g["auc"], "-o", color=colour, lw=2, ms=5, label=name)
    ax.fill_between(g["patients"], g["auc"] - g["auc_sd"], g["auc"] + g["auc_sd"], color=colour, alpha=0.12, lw=0)
ax.set(xscale="log", xlabel="training patients (log scale)", ylabel="validation ROC AUC",
       title="Validation AUC as the training set grows")
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.legend(loc="lower right")
fig.savefig(FIGURES / "fig7_learning_curve.png")
plt.show()
curve.pivot(index="patients", columns="model", values="auc").style.format("{:.4f}")
```

### 7.5 The chosen model

```python
chosen = cardio.choose_model(comparison)
ml = comparison[comparison["kind"] == "machine learning"]
best = ml["auc"].idxmax()
print(f"best cross-validated AUC: {best} ({ml.loc[best, 'auc']:.4f} ± {ml.loc[best, 'auc_sd']:.4f})")
print(f"one standard error: {ml.loc[best, 'auc_sd'] / np.sqrt(cardio.CV_FOLDS):.4f}")
print(f"chosen by the one-standard-error rule: {chosen} ({ml.loc[chosen, 'auc']:.4f}), "
      f"setting: {ml.loc[chosen, 'best setting']}")
```
