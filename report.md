# Patient's Sickness Prediction: Detecting Cardiovascular Disease from a Routine Medical Examination

**Hadi Ghazi**, Lebanese University

*Project topic 2: Patient's Sickness Prediction, using the `cardio` dataset.*

## 1. Introduction

Cardiovascular diseases (CVDs) kill an estimated 17.9 million people every year, more than any other
cause [1]. Many of these deaths could be delayed if patients at risk were identified early. Doctors find
this hard to do reliably by hand: no single measurement settles the question, and the risk builds up
from many small factors.

**Objective.** Build a model that uses the values collected at a routine medical examination (age,
sex, height, weight, blood pressure, cholesterol, glucose, and three lifestyle questions) to tell
whether a patient has a cardiovascular disease. Then evaluate how well it does this. The final model
is delivered with a function that gives a verdict and a risk for a new patient (Section 10).

The dataset [2] has 70,000 patients, which is large for medical tabular data, but it is not clean. Some
recorded blood pressures are impossible, such as −150 or 16,020 mmHg. The signal is also modest:
careful studies of this dataset report about 73% accuracy [5]. Other published papers report 88% [6]
and even 94.5% [7] on the same data, which is hard to reconcile with that. Three questions therefore guide the evaluation, so
that the final model comes with an honest account of what it can do:

- **Q1. How much does machine learning improve on what a doctor already has?** The models are
  compared with two clinical baselines. The first is a blood-pressure threshold from the ACC/AHA 2017
  guideline [3]: systolic pressure of at least 130 mmHg, stage 1 hypertension. The second is the
  Framingham office-based risk score [4], which uses age, sex, BMI, blood pressure, smoking and
  diabetes.
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
epidemiology: the "sick quitter" [8].

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
five folds. All models are implemented with scikit-learn [12]. The test patients are not touched in this section.

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

**Every machine-learning model beats every clinical baseline, but the margins are small.**

- **Best model.** Gradient boosting reaches a cross-validated AUC of 0.800 and an accuracy of
  73.5%. The guideline rule (systolic ≥ 130 mmHg) is right for 71.3% of patients. Out of every 1,000
  patients, the best model classifies about 22 more correctly than one blood-pressure reading.
- **The flexible models are tied.** Gradient boosting, random forest and the neural network lie
  within 0.002 AUC of each other, close to the fold-to-fold standard deviation (about 0.003).
  Logistic regression is 0.010 behind, so the interactions and curves that trees and networks can
  represent are worth a little, but only a little.
- **The guideline's threshold is the best cut-off.** When systolic pressure alone is given the
  cut-off that classifies the training folds most accurately, the data chooses 130 mmHg, the
  guideline's own value: the accuracy, sensitivity and specificity are identical to the rule's. The
  AUC differs only because the rule turns the reading into a yes/no answer.
- **The published Framingham score ranks these patients worse than systolic pressure alone**
  (AUC 0.696 vs 0.751). Refitting the same five terms on these patients raises it to 0.780. The
  functional form is sound; its weights do not transfer to a different population and a different
  question (existing disease here, a first event within ten years in Framingham). A risk score
  carried over to a new population has to be recalibrated before its numbers mean anything.

The cross-validated score of each model's *best* setting is slightly optimistic, because that
setting was chosen on the same folds. With grids this small and 55,000 patients the bias is tiny, but
it is one reason the test set in Section 8 has the final word.

### 7.3 Do the derived features help?

Section 6 added BMI, pulse pressure, mean arterial pressure and the blood-pressure stage. Here the
best settings of logistic regression and gradient boosting are cross-validated again, with and
without them:

```python
features_effect = cardio.feature_set_comparison(train, searches, ["Logistic regression", "Gradient boosting"])
features_effect["gain"] = features_effect["recorded + derived"] - features_effect["recorded only"]
features_effect.style.format("{:.4f}")
```

The derived features change the AUC by less than 0.001 in either direction. That is well inside the
noise. Both models can build what they need from the recorded values on their own, which is a common
finding with tabular data this simple. The derived features stay in the final model anyway: they
cost nothing in accuracy, and they make the explanations in Section 8 easier to read. A clinician
thinks in BMI and blood-pressure stages, not in height and weight.

### 7.4 Would more data help?

The same models are trained on growing shares of a training fold. If the curve is still rising at
the full training set, more patients would help. If it is flat, the limit lies in the features.

```python
curve = cardio.learning_curve_auc(train, {name: searches[name].best_estimator_
                                          for name in ["Logistic regression", "Gradient boosting"]})
fig, ax = plt.subplots(figsize=(6.6, 3.5))
for name, g in curve.groupby("model"):
    colour = NO_CVD if name == "Gradient boosting" else MUTED
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

The two curves tell different stories:

- **Logistic regression has learned all it can from about 2,000 patients.** Its validation AUC moves
  by only 0.003 between 2,000 and 44,000 patients.
- **Gradient boosting needs more data, then slows down.** It needs more patients to estimate its
  interactions, overtakes logistic regression at around 2,000 patients, and keeps improving. But the
  last doubling, from 22,000 to 44,000 patients, adds only 0.0015.

More patients of the same kind would therefore add very little. **The ceiling of about 0.80 AUC (73%
accuracy) is set by the information in the eleven recorded values, not by the amount of data or the
choice of algorithm.** Widening the search grids points the same way. The first run placed the best
gradient-boosting and neural-network settings at the edge of their grids, so the grids were
extended. Both models moved by less than 0.0005 AUC, so they sit on a plateau.

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

Gradient boosting has the best cross-validated AUC, and no simpler model comes within one standard
error of it. Random forest misses by 0.0013 against a margin of 0.0012, a near tie, but the rule was
fixed in advance, and gradient boosting is also faster to train. **Gradient boosting is the final
model.** The next section scores it, and every other candidate, once on the test patients.

## 8. The final model on the test set

### 8.1 Every candidate, scored once

The 13,722 test patients have not been used for anything so far. Each candidate is scored on them
exactly once:

- The machine-learning models use their selected settings, refitted on all training patients.
- The baselines that need a cut-off use the one chosen on the training patients.
- AUC confidence intervals come from 2,000 bootstrap resamples of the test patients.

```python
from sklearn.metrics import roc_auc_score

final = searches[chosen].best_estimator_
y_test = test["cardio"]
test_scores = {
    f"Rule: systolic ≥ {cardio.SBP_RULE} mmHg": (test["ap_hi"] >= cardio.SBP_RULE).astype(float),
    "Systolic pressure alone": test["ap_hi"].astype(float),
    "Framingham (published)": cardio.framingham_risk(test),
    **{name: s.predict_proba(test[cardio.FEATURES])[:, 1] for name, s in searches.items()},
}
cutoffs = {name: 0.5 for name in test_scores}
cutoffs["Systolic pressure alone"] = cardio.accuracy_threshold(train["cardio"], train["ap_hi"])
cutoffs["Framingham (published)"] = cardio.accuracy_threshold(train["cardio"], cardio.framingham_risk(train))

auc_boot = cardio.bootstrap(y_test, test_scores, roc_auc_score)
rows = []
for name, score in test_scores.items():
    m = cardio.classification_metrics(y_test, score, cutoffs[name])
    lo, hi = cardio.interval(auc_boot[name])
    rows.append({"model": name, "test AUC": m["auc"], "95% CI": f"{lo:.3f}-{hi:.3f}",
                 "CV AUC": comparison.loc[name, "auc"], "accuracy": m["accuracy"],
                 "sensitivity": m["sensitivity"], "specificity": m["specificity"]})
test_table = pd.DataFrame(rows).set_index("model").sort_values("test AUC", ascending=False)
test_table.to_csv(ROOT / "outputs" / "test_results.csv")
test_table.style.format({"test AUC": "{:.3f}", "CV AUC": "{:.3f}", "accuracy": "{:.1%}",
                         "sensitivity": "{:.1%}", "specificity": "{:.1%}"})
```

**The test set confirms the cross-validation.** Every candidate's test AUC is within 0.004 of its
cross-validated AUC, and the ranking is unchanged. The models did not overfit, and choosing settings
on the cross-validation folds did not flatter them. On patients it has never seen, gradient boosting
reaches an AUC of 0.804 (95% CI 0.797-0.811) and 73.6% accuracy, against 71.3% for the
systolic ≥ 130 rule.

```python
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(5.6, 5.2))
for name, colour, lw in [(chosen, NO_CVD, 2.2), ("Logistic regression", MUTED, 1.4),
                         ("Systolic pressure alone", "#9b9a94", 1.4), ("Framingham (published)", "#c9c8c1", 1.4)]:
    fpr, tpr, _ = roc_curve(y_test, test_scores[name])
    ax.plot(fpr, tpr, color=colour, lw=lw, label=f"{name} (AUC {roc_auc_score(y_test, test_scores[name]):.3f})")
rule = cardio.classification_metrics(y_test, test_scores[f"Rule: systolic ≥ {cardio.SBP_RULE} mmHg"], 0.5)
ax.plot(1 - rule["specificity"], rule["sensitivity"], "D", color=CVD, ms=8, label=f"rule: systolic ≥ {cardio.SBP_RULE}")
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=MUTED)
ax.set(xlabel="1 − specificity (healthy patients flagged)", ylabel="sensitivity (diseased patients detected)",
       title="ROC curves on the test patients", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="lower right", fontsize=8)
fig.savefig(FIGURES / "fig8_roc_test.png")
plt.show()
```

The rule is one point on the ROC plane; the model is a whole curve. Holding the rule's specificity
fixed shows what the model adds at that point:

```python
rule_spec = rule["specificity"]
fpr, tpr, thresholds = roc_curve(y_test, test_scores[chosen])
at_rule = tpr[np.searchsorted(fpr, 1 - rule_spec, side="right") - 1]
n_sick = int(y_test.sum())
print(f"at the rule's specificity ({rule_spec:.1%}): rule detects {rule['sensitivity']:.1%} of diseased patients, "
      f"{chosen.lower()} {at_rule:.1%}")
print(f"that is {round((at_rule - rule['sensitivity']) * n_sick):,} more of the {n_sick:,} diseased test patients, "
      f"with the same number of healthy patients flagged")
```

At the rule's own operating point, the model finds 66.2% of the diseased patients instead of 61.8%:
301 more of the 6,785 diseased test patients, with no extra healthy patients flagged. This is the
most concrete form of the answer to Q1. The gain is real but modest, about one more patient detected
for every 23 diseased patients examined.

### 8.2 Are the differences real?

The same bootstrap resamples score every model, so each AUC difference can be given its own
interval. If an interval excludes zero, the difference is unlikely to be chance.

```python
pairs = [(chosen, other) for other in ["Random forest", "Neural network (MLP)", "Logistic regression",
                                        "Framingham form (refit)", "Systolic pressure alone"]]
diffs = []
for a, b in pairs:
    d = auc_boot[a] - auc_boot[b]
    lo, hi = cardio.interval(d)
    diffs.append({"comparison": f"{a} − {b}", "AUC difference": roc_auc_score(y_test, test_scores[a])
                  - roc_auc_score(y_test, test_scores[b]), "95% CI": f"{lo:+.4f} to {hi:+.4f}",
                  "excludes zero": lo > 0 or hi < 0})
pd.DataFrame(diffs).set_index("comparison").style.format({"AUC difference": "{:+.4f}"})
```

All five intervals exclude zero, but they mean different things:

- **Over random forest and the neural network**, gradient boosting leads by less than 0.002 AUC.
  With 13,722 test patients even that is statistically detectable, but it would not change a single
  clinical decision in any meaningful way. The three flexible models are equally good in practice.
- **Over logistic regression** the lead is 0.012. Small, but real: the trees capture something a
  linear model on these features cannot (Section 8.4 shows what).
- **Over Framingham's form refitted** (0.024) and **over systolic pressure alone** (0.050), the gaps
  are clear.

### 8.3 Choosing the threshold

A predicted risk becomes a verdict only through a cut-off. At 0.5 the model maximises accuracy. A
screening programme usually cares more about not missing sick patients, so a second cut-off is
chosen **on the training patients**: the highest one that detects at least 80% of the diseased
training patients, using predictions from models that did not see them (out-of-fold).

```python
from sklearn.metrics import confusion_matrix

oof = cardio.out_of_fold(final, train)
cut_screen = cardio.screening_threshold(train["cardio"], oof, 0.80)
risk_test = test_scores[chosen]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
summary = []
for ax, (label, cut) in zip(axes, [("accuracy cut-off (0.50)", 0.5), (f"screening cut-off ({cut_screen:.2f})", cut_screen)]):
    cm = confusion_matrix(y_test, risk_test >= cut)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n{v / cm[i].sum():.0%} of row", ha="center", va="center", fontsize=9,
                color="white" if v > cm.max() * 0.6 else INK)
    ax.set_xticks([0, 1], ["predicted healthy", "predicted CVD"])
    ax.set_yticks([0, 1], ["healthy", "has CVD"])
    ax.grid(False)
    ax.set_title(label)
    m = cardio.classification_metrics(y_test, risk_test, cut)
    summary.append({"cut-off": label, **m})
fig.suptitle(f"{chosen} on the test patients", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig9_confusion_matrices.png")
plt.show()
pd.DataFrame(summary).set_index("cut-off").drop(columns="auc").style.format("{:.1%}")
```

The screening cut-off, chosen on the training patients alone, detects 80.1% of the diseased test
patients: the target carries over to unseen patients. The price is specificity. At 0.50 the model
misses 2,112 of the 6,785 diseased test patients and flags 1,510 of the 6,937 healthy ones. At the
screening cut-off it misses 1,347 and flags 2,476. Which trade-off is right is not a statistical
question. It depends on what follows a positive result. For a cheap follow-up examination, the
screening cut-off is the sensible choice.

### 8.4 What drives the predictions

Two complementary views. **Grouped permutation importance** asks how much test AUC is lost when a
factor is scrambled across patients. Related features are scrambled together, so the four
blood-pressure features cannot cover for one another. **Partial dependence** asks how the average
predicted risk changes as one factor is varied for everyone.

```python
importance = cardio.grouped_permutation_importance(final, test)

fig, ax = plt.subplots(figsize=(6.6, 3.6))
order = importance.iloc[::-1]
ax.barh(order.index, order["auc drop"], xerr=order["sd"], color=NO_CVD, height=0.6,
        error_kw={"ecolor": MUTED, "elinewidth": 1})
for yi, v in enumerate(order["auc drop"]):
    ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
ax.set(xlabel="drop in test AUC when the factor is scrambled", title="What the model relies on")
ax.grid(axis="y", visible=False)
fig.savefig(FIGURES / "fig10_importance.png")
plt.show()
```

**Blood pressure dominates.** Scrambling the five blood-pressure features costs 0.21 AUC: without
them the AUC would fall from 0.80 to about 0.59. Age (0.036) and cholesterol (0.030) follow,
and body size adds a little (0.007). Glucose, the three lifestyle answers and sex together contribute
almost nothing. This matches the exploratory analysis in Section 5.

```python
curves = [("ap_hi", range(100, 191, 5), "systolic pressure (mmHg)"),
          ("age_years", range(40, 66, 2), "age (years)"),
          ("weight", range(50, 121, 5), "weight (kg)"),
          ("cholesterol", [1, 2, 3], "cholesterol (1 normal - 3 well above)"),
          ("gluc", [1, 2, 3], "glucose (1 normal - 3 well above)"),
          ("smoke", [0, 1], "smokes (0 no, 1 yes)"),
          ("active", [0, 1], "physically active (0 no, 1 yes)"),
          ("alco", [0, 1], "drinks alcohol (0 no, 1 yes)")]
sample = test.sample(3000, random_state=cardio.SEED)

fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
for ax, (feature, values, label) in zip(axes.flat, curves):
    pdc = cardio.partial_dependence_curve(final, sample, feature, list(values))
    ax.plot(pdc.index, pdc.values, "-o", color=NO_CVD, lw=2, ms=4)
    ax.set(xlabel=label)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
for ax in axes[:, 0]:
    ax.set_ylabel("average predicted risk")
fig.suptitle("How the predicted risk moves with each factor (partial dependence, 3,000 test patients)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig11_partial_dependence.png")
plt.show()
```

The partial-dependence curves show the *shape* of each effect, and why the trees beat logistic
regression:

- **Systolic pressure works like a switch, not a slope.** The average predicted risk stays around
  30-40% below 125 mmHg, climbs steeply between 125 and 150 mmHg, and then stays flat at about 84%.
  A logistic regression on raw pressure has to draw one straight line in log-odds through this
  S-shape. The trees follow it. The steep part sits exactly on the guideline's stage 1 and stage 2
  thresholds (130 and 140 mmHg).
- **Age, weight and cholesterol raise the risk steadily.** Age goes from about 38% at 40 to 65% at 64,
  and cholesterol "well above normal" adds about 25 percentage points.
- **Glucose, smoking and alcohol are flat or slightly *protective*.** This is the reverse-causation
  pattern from Section 5.2. The model has learned the data correctly, and the data reflects patients
  who changed their habits after diagnosis. **These curves describe associations in this dataset and
  must not be read as causal effects**: nothing here suggests that smoking is harmless.

### 8.5 A points score a doctor can use on paper

If the best model gains only a little over simpler ones, the most useful result might be a
transparent score. Here each factor is cut into clinical bands, a logistic regression is fitted on
the bands, and its coefficients are rounded to whole points (one point = 0.1 in log-odds). The total
then maps to a risk.

```python
points = cardio.PointsScore().fit(train, train["cardio"])
card = points.table()
card.set_index(["factor", "band"])
```

The scorecard reads like the model. Systolic pressure earns up to 27 points, age up to 11,
cholesterol up to 11, and BMI up to 4. The lifestyle answers and high glucose earn *negative* points,
for the reverse-causation reason above. No clinician could hand out a score that rewards smoking,
and these factors carry almost no information anyway (Section 8.4). So a **clinical version** keeps
only the four factors with a sound direction: age, systolic pressure, BMI and cholesterol.

```python
test_scores["Points score"] = points.predict_proba(test)
boot_points = cardio.bootstrap(y_test, {"Points score": test_scores["Points score"], chosen: test_scores[chosen]},
                               roc_auc_score)
m = cardio.classification_metrics(y_test, test_scores["Points score"], 0.5)
lo, hi = cardio.interval(boot_points["Points score"])
d_lo, d_hi = cardio.interval(boot_points[chosen] - boot_points["Points score"])
print(f"points score on the test patients: AUC {m['auc']:.3f} ({lo:.3f}-{hi:.3f}), accuracy {m['accuracy']:.1%}, "
      f"sensitivity {m['sensitivity']:.1%}, specificity {m['specificity']:.1%}")
print(f"{chosen} minus points score: AUC {roc_auc_score(y_test, test_scores[chosen]) - m['auc']:+.4f} "
      f"(95% CI {d_lo:+.4f} to {d_hi:+.4f})")

totals = points.total(test)
observed = test.groupby(totals)["cardio"].agg(["mean", "size"])
observed = observed[observed["size"] >= 50]
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.scatter(observed.index, observed["mean"], s=np.sqrt(observed["size"]) * 2, color=NO_CVD, alpha=0.8,
           label="observed share with CVD (test, ≥ 50 patients)")
grid = np.arange(observed.index.min(), observed.index.max() + 1)
ax.plot(grid, points.risk_by_total(grid), color=CVD, lw=2, label="risk the score assigns")
ax.set(xlabel="total points", ylabel="share with CVD", title="Points score: total points against observed risk")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(loc="upper left", fontsize=8)
fig.savefig(FIGURES / "fig12_points_score.png")
plt.show()
```

```python
clinical = cardio.PointsScore(factors=cardio.CLINICAL_FACTORS).fit(train, train["cardio"])
test_scores["Points score (clinical, 4 factors)"] = clinical.predict_proba(test)
for name in ["Points score", "Points score (clinical, 4 factors)", "Logistic regression", chosen]:
    m = cardio.classification_metrics(y_test, test_scores[name], 0.5)
    print(f"{name:<38} AUC {m['auc']:.3f}   accuracy {m['accuracy']:.1%}")
clinical.table().set_index(["factor", "band"]).T
```

**Dropping the four questionable factors costs almost nothing.** The four-factor clinical scorecard
reaches an AUC of 0.793 and 72.3% accuracy: exactly logistic regression on all fifteen features, and
0.011 AUC below gradient boosting. A doctor with a pen can add up four numbers (age, systolic
pressure, BMI, cholesterol) and get within about one point of accuracy of the best machine-learning
model. This is in line with Rudin's argument that for tabular problems like this one, an
interpretable model is often nearly as accurate as a black box [9].

## 9. Reliability

Section 8 measured how well the model works. This section asks how far those numbers can be
trusted: how they depend on data handling choices (Q2), and whether the predicted risks hold up as
risks, for every kind of patient (Q3).

### 9.1 Does the cleaning strategy matter?

The final gradient-boosting setting and the logistic regression are trained three times: on the raw
training data, on the `drop` training data and on the `repair` training data. All six models are
scored on the same test patients (those whose records pass every check), so only the training data
differs.

```python
strategies = cardio.strategy_comparison({"Gradient boosting": final, "Logistic regression": searches["Logistic regression"].best_estimator_})
(strategies.pivot(index="strategy", columns="model", values=["test AUC", "test accuracy"])
 .reindex(list(cardio.STRATEGIES)).style.format("{:.4f}"))
```

The answer depends on the model, exactly as Section 4.3 predicted:

- **Gradient boosting does not care.** Its test AUC is 0.804 whether it is trained on the raw,
  dropped or repaired data. Trees split on the order of values, so a pressure of 16,020 mmHg simply
  lands in the top branch with every other high reading.
- **Logistic regression is hurt by the raw data.** Its AUC falls from 0.793 to 0.776. A handful of
  absurd values pull its single straight-line coefficient for blood pressure towards zero, which
  drowns out the strongest predictor.
- **Repairing and dropping give the same result.** The 1,257 repaired records are under 2% of the
  data, too few to change what the model learns. Repair keeps more patients, but its value lies in
  not discarding real people rather than in accuracy.

The practical lesson: cleaning is essential for linear models and for any analysis that reads
coefficients or correlations (Section 4.3), even when a tree model would have survived without it.

### 9.2 How mistakes inflate reported accuracy

Papers on this dataset report accuracies from 73% [5] to 94.5% [7]. This experiment reproduces three
mistakes common in published machine-learning work on medical data and measures the accuracy each
one would let a study report. Each mistake is shown next to the honest figure for the same model.
These pipelines are wrong on purpose, and none of them feeds into the model chosen above. Which
mistakes, if any, particular papers made cannot be known from here.

```python
t0 = time.time()
correct_acc = cardio.classification_metrics(y_test, test_scores[chosen], 0.5)["accuracy"]
leakage = cardio.leakage_experiments(train, test, correct_acc)
print(f"leakage experiments: {time.time() - t0:.0f} s")
leakage.style.format({"reported accuracy": "{:.1%}"}).hide(axis="index")
```

The same data can "report" anything from 64% to 99.9% accuracy, depending on how the evaluation is
set up:

1. **Accuracy measured on the training patients.** A forest of fully grown trees memorises its
   training patients and scores 99.9% on them, against 71.0% on patients it has never seen. Any
   figure that does not come from held-out patients says nothing about the model.
2. **SMOTE before the split** [10]. Synthetic patients are interpolated between real neighbours,
   so after a random split most test patients have a near-twin in the training set. A one-nearest-
   neighbour classifier then "reaches" 85.3% and the forest 86.7%: inflated by 16-21 points. These
   accuracies are close to those reported in [6]. Applied correctly, to the training patients only,
   SMOTE does nothing useful here (the classes are already balanced) and both models fall to 64-71%.
   The same mistake explained a published 99% accuracy on another cardiovascular dataset; once it was
   corrected, accuracy fell to 77% [11].
3. **Trying many random splits and reporting the best one.** This is the mildest mistake: the best
   of 50 splits is 73.4%, against 72.8% for the median split. With 14,000 test patients the luck of
   the split is worth less than one point. It matters far more in small datasets.

None of this shows what the authors of [6] or [7] actually did; their papers would have to be
reproduced to know. What it does show is that **an accuracy far above 74% on this dataset should be
read as a sign of leakage until the pipeline proves otherwise**: the learning curve in Section 7.4
gives no reason to expect a model to gain 15 or 20 points.

### 9.3 Can the predicted risks be taken at face value?

A risk of 70% should mean that about 70 of 100 such patients have CVD. The calibration curve groups
the test patients by predicted risk and compares each group's average prediction with the share who
actually have CVD. The Brier score (the mean squared error of the risks) sums up both calibration
and discrimination: lower is better, and always predicting 50% scores 0.25.

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

fig, ax = plt.subplots(figsize=(5.4, 5.0))
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=MUTED, label="perfect calibration")
for name, colour in [(chosen, NO_CVD), ("Logistic regression", MUTED), ("Points score", CVD)]:
    observed, predicted = calibration_curve(y_test, test_scores[name], n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_test, test_scores[name])
    ax.plot(predicted, observed, "-o", color=colour, lw=1.8, ms=4, label=f"{name} (Brier {brier:.4f})")
ax.set(xlabel="predicted risk (test patients, 10 equal-sized groups)", ylabel="share with CVD",
       title="Calibration on the test patients", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="upper left", fontsize=8)
fig.savefig(FIGURES / "fig13_calibration.png")
plt.show()
```

**Gradient boosting's risks can be read as risks.** Its ten points lie on the diagonal: patients
given about 20% have CVD about 20% of the time, and patients given 80% about 80% of the time. It also
has the lowest Brier score (0.179). Logistic regression bends around the diagonal: it overstates low
risks and understates the risk around 60-70%, the S-shaped error of a straight line forced through
the switch-like effect of blood pressure (Section 8.4). The points score is close to calibrated, with
a small excess of observed cases in the middle range.

### 9.4 Does the model work equally well for everyone?

```python
subgroups = cardio.subgroup_table(test, test_scores[chosen])
subgroups.style.format({"patients": "{:,}", "share with CVD": "{:.1%}", "mean predicted risk": "{:.1%}",
                        "AUC": "{:.3f}", "sensitivity": "{:.1%}", "specificity": "{:.1%}"})
```

- **Women and men are served equally.** Their AUCs (0.803 and 0.808) and error rates are close, and
  in both groups the mean predicted risk matches the observed share within 1 point.
- **Age is different.** Discrimination falls with age, from an AUC of 0.835 below 50 to 0.718 at 60
  and over. Among older patients most have high blood pressure and most have CVD, so the main signal
  separates them less well. The risks remain calibrated in every age group (mean predicted vs
  observed within 1.5 points). But at the 0.5 cut-off the model flags 88% of diseased patients over 60
  while clearing only 35% of the healthy ones; below 50 it is the reverse (59% and 91%). A single
  cut-off does not suit every age. A deployed tool could use age-specific cut-offs, or report the risk
  itself and leave the threshold to the clinician.

### 9.5 What the numbers mean outside this dataset

Half the patients in this dataset have CVD. In the population that attends a routine check-up the
share is much lower, and this changes what a positive prediction means. Sensitivity and
specificity describe the model and carry over. The **positive predictive value**, the chance that a
flagged patient really has CVD, depends on how common the disease is:

```python
m_acc = cardio.classification_metrics(y_test, test_scores[chosen], 0.5)
m_scr = cardio.classification_metrics(y_test, test_scores[chosen], cut_screen)
ppv = pd.DataFrame({f"{label} (sens {m['sensitivity']:.0%}, spec {m['specificity']:.0%})":
                    {f"{p:.0%}": cardio.ppv_at_prevalence(m["sensitivity"], m["specificity"], p)
                     for p in (0.5, 0.3, 0.2, 0.1, 0.05)}
                    for label, m in [("accuracy cut-off", m_acc), ("screening cut-off", m_scr)]})
ppv.index.name = "share of patients with CVD"
ppv.style.format("{:.0%}")
```

In this dataset 76% of the patients the model flags really have CVD. In a check-up population where
one patient in five has CVD, the same model, with the same sensitivity and specificity, would be right
about only 44% of the patients it flags (36% at the screening cut-off). The model is still useful
there, because it concentrates the disease: 44% is more than twice the 20% base rate. But its verdict
has to be read as "worth a closer look", not as a diagnosis. And before its *risks* could be used in
such a population, it would have to be recalibrated to that population's prevalence.

## 10. Using the model

The project's goal was a model that checks a patient for cardiovascular disease. `cardio.py` turns
the chosen model into a tool that does this. It saves the model with its two cut-offs and the
scorecard. It also checks a new examination with the same plausibility rules as the data, so an
impossible reading is rejected instead of scored, and then reports the risk, the verdict and the
points.

```python
# the setting saved for the command line must be the one this report selected
selected = {k: v for k, v in final.get_params().items() if k in cardio.FINAL_PARAMS}
assert selected == cardio.FINAL_PARAMS, (selected, cardio.FINAL_PARAMS)

bundle = cardio.build_final(train, model=final)
print("saved:", cardio.save_final(bundle).relative_to(ROOT))
print(f"screening cut-off: {bundle['screening_cutoff']:.3f}")
```

Two invented examination records show the output:

```python
examples = {
    "62-year-old man, 165/100 mmHg, BMI 33, cholesterol well above normal":
        dict(age=62, sex="male", height=172, weight=98, systolic=165, diastolic=100, cholesterol=3),
    "44-year-old woman, 115/75 mmHg, BMI 22, all results normal":
        dict(age=44, sex="female", height=165, weight=60, systolic=115, diastolic=75),
    "55-year-old woman, 130/85 mmHg, BMI 27, cholesterol above normal":
        dict(age=55, sex="female", height=160, weight=69, systolic=130, diastolic=85, cholesterol=2),
}
rows = []
for label, values in examples.items():
    r = cardio.assess(bundle, cardio.patient(**values))
    rows.append({"patient": label, "predicted risk": r["risk"], "verdict": r["verdict"],
                 "screening": r["screening"], "points": r["points"], "risk from points": r["points_risk"]})
pd.DataFrame(rows).set_index("patient").style.format({"predicted risk": "{:.0%}", "risk from points": "{:.0%}"})
```

The three invented patients get the answers a clinician would expect. The 62-year-old man with
stage 2 hypertension, obesity and high cholesterol is at 87%. The young woman with normal results is
at 14%. The 55-year-old woman in between is at 59%, so she is referred at either cut-off. The saved
tool uses the four-factor clinical scorecard (Section 8.5), and it ranks the three patients in the
same order as the model.

```python
try:
    cardio.patient(age=50, sex="male", height=175, weight=80, systolic=1400, diastolic=90)
except ValueError as e:
    print("rejected:", e)
```

The same works from a terminal, after `python cardio.py train`:

```
$ python cardio.py predict --age 62 --sex male --height 172 --weight 98 --systolic 165 --diastolic 100 --cholesterol 3
```

## 11. Conclusion

**Objective met.** A gradient-boosting model detects cardiovascular disease from the eleven values of
a routine examination. On 13,722 patients it never saw, it reaches an AUC of 0.804 (95% CI
0.797-0.811) and 73.6% accuracy. Its risks are well calibrated, and it works equally well for women
and men. `cardio.py` packages it as a tool that checks a new examination, rejects impossible values,
and returns a risk, a verdict and a points breakdown.

**Q1. What machine learning adds over what a doctor already has: something real, but small.**

- **Over the guideline rule.** The model beats systolic pressure ≥ 130 mmHg by 2.3 points of
  accuracy. At the rule's specificity, it detects 301 more of 6,785 diseased patients.
- **Between the models.** The three flexible models are tied, and logistic regression is only 0.012
  AUC behind.
- **The scorecard.** A four-factor paper score comes within 0.011 AUC of the best model.
- **Why the gain is small.** The learning curve is flat, so the ceiling of about 0.80 AUC (73-74%
  accuracy) is set by what the examination records, not by the algorithm or the amount of data.
- **The Framingham score.** As published, it ranks these patients worse than blood pressure alone,
  and it only becomes competitive once refitted to them.

**Q2. Data handling decides more than the algorithm does.**

- **Cleaning.** Fewer than 2% of records hide the strongest predictor from any linear analysis, and
  cost logistic regression 0.017 AUC. Tree models survive the dirty data, and repairing the records
  is as good as dropping them.
- **Leakage.** Common pipeline mistakes turn the same data into reported accuracies of 85-100%. Claims
  far above 74% on this dataset should be treated as signs of leakage until shown otherwise.

**Q3. The predictions can be trusted, within limits.**

- **Calibration.** The risks are calibrated overall, for both sexes and in every age group.
- **What drives them.** The explanations are dominated by blood pressure, age and cholesterol, in the
  direction medicine expects.
- **Weak spots.** Discrimination is weaker in patients over 60. The lifestyle answers carry
  reverse-caused associations that must not be read as effects.

### Limitations

1. **Detection, not prediction.** Diagnosis and features come from the same examination. The model
   recognises existing disease. It cannot forecast who will develop CVD.
2. **Unknown provenance and outcome definition.** The dataset's source and the exact meaning of
   "cardiovascular disease" are undocumented. If the label includes diagnosed hypertension, part of
   blood pressure's predictive power is circular.
3. **Treatment is not recorded.** Medication lowers blood pressure and cholesterol in diagnosed
   patients, and diagnosis changes habits. Both weaken or reverse associations (Section 5.2), and
   neither can be corrected for here.
4. **A 50/50 sample.** Predictive values and absolute risks apply to this sample. Another population
   needs recalibration (Section 9.5).
5. **One dataset.** All results come from one source with one train/test split. The split is large
   and every estimate carries a confidence interval, but there is no external validation.
6. **Coarse measurements.** Cholesterol and glucose come in three categories, and diabetes had to be
   approximated for the Framingham score.

The next step for this model would be validation on data from a known clinical source, with
recorded treatment and, ideally, outcomes followed over time. Those are what would turn a detector of
existing disease into a tool for prevention.

## References

1. World Health Organization. *Cardiovascular diseases (CVDs)*, fact sheet.
2. S. Ulianova. *Cardiovascular Disease dataset*. Kaggle, 2019.
3. P. K. Whelton et al. 2017 ACC/AHA guideline for the prevention, detection, evaluation, and
   management of high blood pressure in adults. *Hypertension*, 71(6):e13-e115, 2018.
4. R. B. D'Agostino Sr et al. General cardiovascular risk profile for use in primary care: the
   Framingham Heart Study. *Circulation*, 117(6):743-753, 2008.
5. R. S. Ramesh, R. T. S. Udupa, Monisha J and Kushi K. K. S. Cardiovascular disease prediction using
   machine learning: a comparative analysis. arXiv:2507.21898, 2025.
6. D. Y. Omkari and K. Shaik. An integrated two-layered voting (TLV) framework for coronary artery
   disease prediction using machine learning classifiers. *IEEE Access*, 2024.
   doi:10.1109/ACCESS.2024.3389707.
7. A. R. Zafar et al. Cardiovascular diagnostic detection using multilayer perceptron approach.
   *Systems and Soft Computing*, 2026. doi:10.1016/j.sasc.2026.200490.
8. A. G. Shaper, G. Wannamethee and M. Walker. Alcohol and mortality in British men: explaining the
   U-shaped curve. *The Lancet*, 2(8623):1267-1273, 1988.
9. C. Rudin. Stop explaining black box machine learning models for high stakes decisions and use
   interpretable models instead. *Nature Machine Intelligence*, 1:206-215, 2019.
10. N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer. SMOTE: synthetic minority
    over-sampling technique. *Journal of Artificial Intelligence Research*, 16:321-357, 2002.
11. Eltawil et al. Comment on Iacobescu et al. "Evaluating binary classifiers for cardiovascular
    disease prediction: enhancing early diagnostic capabilities". *Journal of Cardiovascular
    Development and Disease*, 13(1):46, 2026. doi:10.3390/jcdd13010046.
12. F. Pedregosa et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning
    Research*, 12:2825-2830, 2011.
