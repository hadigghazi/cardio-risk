# Patient's Sickness Prediction: Detecting Cardiovascular Disease from a Routine Medical Examination

**Hadi Ghazi**, Lebanese University

*Project topic: Patient's Sickness Prediction, using the `cardio` dataset.*

## 1. Introduction

Cardiovascular diseases (CVDs) are the leading cause of death worldwide: an estimated 19.8 million
people died of them in 2022, about a third of all deaths [1]. Many of these deaths could be delayed if
patients at risk were identified early. Doctors find this hard to do reliably by hand: no single
measurement settles the question, and the risk builds up from many small factors.

**Objective.** Build a model that uses the values collected at a routine medical examination (age,
sex, height, weight, blood pressure, cholesterol, glucose, and three lifestyle questions) to tell
whether a patient has a cardiovascular disease. Then evaluate how well it does this. The final model
is delivered with a function that gives a verdict and a risk for a new patient (Section 10).

The dataset [2] has 70,000 patients, which is large for medical tabular data, but it is not clean. Some
recorded blood pressures are impossible, such as −150 or 16,020 mmHg. The signal is also modest: a
comparative study of a cleaned version of this dataset reports 73.4% accuracy for its best model [5].
Other papers report 88% [6] and 94.5% [7] on the same 70,000 records. A gap that large deserves an
explanation. Three questions therefore guide the evaluation, so that the final model comes with an
honest account of what it can do:

- **Q1. How much does machine learning improve on what a doctor already has?** The models are
  compared with two clinical baselines. The first is a blood-pressure threshold from the ACC/AHA 2017
  guideline [3]: systolic pressure of at least 130 mmHg, stage 1 hypertension. The second is the
  Framingham office-based risk score [4], which uses age, sex, BMI, blood pressure, smoking and
  diabetes.
- **Q2. How much do data-handling choices change the measured performance?** This covers the
  cleaning strategy, and the common evaluation mistakes that let information from the test set leak
  into training. Reproducing those mistakes shows how much they can inflate a reported accuracy.
- **Q3. Can the final model's predictions be trusted and explained?** This covers which factors drive
  the predictions, whether a simple points score can match the best model, whether predicted risks
  match observed frequencies, and how the model performs for different sexes and age groups.

**Structure.** Section 2 fixes the evaluation protocol before any model is trained. Section 3
describes the data, Section 4 cleans it, Section 5 explores it, and Section 6 derives features.
Section 7 trains and compares the models (Q1), Section 8 evaluates the chosen model on the test set
and explains it, and Section 9 runs the reliability checks (Q2, Q3). Section 10 shows the model in
use, and Section 11 concludes, with the study's limitations.

All the project's code is in this notebook, each part placed where it is first needed. The first
cell holds the libraries, the file locations and the constants used throughout. The notebook finds
its folder by looking for `data/cardio.csv`. Running every cell takes about 15 minutes on four
processor cores.

<!-- code: setup -->

```python
import matplotlib.pyplot as plt

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
   uses the same ids. From Section 4 on, every figure that involves the diagnosis is computed on the
   training patients only, and so is every modelling decision.
3. **Cleaning learns nothing from the data.** It applies fixed per-record rules, so it cannot leak
   test information into training. (The rules were written after looking at the impossible values of
   the whole file, but they never look at the diagnosis.) Anything that *is* learned (scaling, model
   parameters, cut-offs) is fitted on training data only, inside cross-validation folds during model
   selection.
4. **Model selection.** Models are tuned and compared with 5-fold stratified cross-validation on the
   training set. The test set is scored once, for the final comparison, and never used to choose
   anything.
5. **Common test set.** When cleaning strategies are compared, all of them are scored on the same
   test patients: those whose records pass every plausibility check. A deployed system would reject
   an impossible input at entry rather than predict from it. How the final model does on the records
   that fail the checks is reported separately (Section 9.1).
6. **Metrics.** The primary metric is ROC AUC: how well the model ranks diseased patients above
   healthy ones, independent of any cut-off. Accuracy, sensitivity (the share of diseased patients
   detected) and specificity (the share of healthy patients cleared) are reported at the 0.5 cut-off
   and at a screening cut-off chosen on training data. The Brier score and calibration curves measure
   how far predicted risks can be trusted. The main results get 95% bootstrap confidence intervals:
   every test AUC, the accuracy of each candidate, the final model's gain over the rule, the
   differences between models and between cleaning strategies, and the AUC within each subgroup.
   These intervals reflect which patients happen to be in the test set; they do not include the
   randomness of training. Other figures are point estimates.
7. **The leakage experiments (Q2) are kept separate.** They reproduce mistakes on purpose. They are
   labelled as such and never feed into the choice of the final model.

## 3. The data

The dataset is read exactly as distributed. Exact duplicate records, identical in every column but
`id`, are removed before anything else, and the train/test split of Section 2 is drawn here too.

<!-- code: loading -->

```python
raw = load()
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
    [(col, lo, hi) for col, (lo, hi) in BOUNDS.items()] + [("BMI", *BMI_BOUNDS)],
    columns=["quantity", "lower bound", "upper bound"],
).set_index("quantity")
```

The code below defines the plausibility checks, the repair rules of Section 4.2 and the three
cleaning strategies. Nothing in it is learned from the data: every rule looks at one record at a time.

<!-- code: data quality -->

```python
deduped = deduplicate(raw)
flags = audit(deduped)
failing = flags.any(axis=1)
train_ids, test_ids = split_ids(raw)
in_train = deduped["id"].isin(train_ids)
print(f"{len(raw) - len(deduped)} exact duplicates removed")
print(f"{failing.sum():,} records ({failing.mean():.1%}) fail at least one check")
print(f"share with CVD, training patients: {deduped.loc[failing & in_train, 'cardio'].mean():.1%} of those "
      f"failing a check, {deduped.loc[~failing & in_train, 'cardio'].mean():.1%} of the others\n")
flags.sum().rename("records failing").to_frame()
```

About 2% of records fail, and nearly all failures are blood pressures. They are not random in two
ways. First, they come mostly from sick patients: among training patients, three in four records
that fail a check (74%) belong to patients with CVD, against half of the others. Errors may have been
more common where sick patients' records were entered, but the data cannot say why. Removing these
records therefore removes mostly sick patients, which Section 9.1 takes into account. Second, the most
frequent impossible values show clear, repeated patterns:

```python
def top_values(col, n=12):
    lo, hi = BOUNDS[col]
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
- **Swapped fields.** Pairs like 80/120 or 90/160 have systolic and diastolic entered the wrong way
  round.

```python
pairs = deduped.loc[flags["diastolic >= systolic"], ["ap_hi", "ap_lo"]].value_counts().head(8)
pairs.rename("records").to_frame()
```

The figure shows the same thing. Every point is one patient's blood-pressure pair. On the left, the
raw values on a logarithmic scale, with records that fail a check highlighted. On the right, the same
patients after repair.

```python
repaired, log = clean(raw, "repair")
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
groups = deduped[in_train].groupby(list(features_only.columns))["cardio"].nunique()
print(f"{(groups > 1).sum()} groups of training records identical in every feature but with different diagnoses")
```

These are kept. They are a reminder that the features do not determine the diagnosis: no model can
classify all of these records correctly.

### 4.2 Three cleaning strategies

Each strategy becomes its own training set (Q2):

```python
summary = cleaning_summary(raw)
summary.style.format({"share kept": "{:.1%}", "records": "{:,}"})
```

`repair` corrects a blood pressure only when a single recognisable entry error explains it. Readings
typed into each other's fields are swapped first, on the values as recorded, and only when the
swapped pair is plausible. Each remaining reading may then be fixed by one of the other rules: sign,
unit or extra digits. A record that would need two guesses (a unit fix followed by a swap, say) is
not repaired. Anything no rule recovers is removed, and so are implausible heights and weights,
whose errors show no pattern that would identify the true value. Every change is logged:

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
for s in STRATEGIES:
    d, _ = clean(raw, s)
    d = d[d["id"].isin(train_ids)]
    rows.append({"strategy": s, "training records": len(d),
                 "corr(systolic, cardio)": d["ap_hi"].corr(d["cardio"]),
                 "corr(diastolic, cardio)": d["ap_lo"].corr(d["cardio"]),
                 "mean systolic": d["ap_hi"].mean(), "max systolic": d["ap_hi"].max()})
pd.DataFrame(rows).set_index("strategy")
```

This is the most important number in the cleaning section. In the raw training data, systolic blood
pressure has almost no linear relationship with the diagnosis (r = 0.05). After either kind of
cleaning it is the strongest single predictor (r = 0.43). Fewer than 2% of records cause this. A
handful of values in the thousands dominate the variance and hide everything else. Anything linear,
such as a correlation or a logistic regression on unscaled values, reads the raw data wrongly. Tree
models split on the order of values and should suffer far less. Section 9.1 tests whether that holds.

## 5. Exploratory data analysis

From here on, the analysis uses the **training patients only**, cleaned with the `drop` strategy
(Section 9.1 compares it with the other two strategies). The test patients stay unseen until
Section 8.

The modelling table is built by the functions below. They convert units, add the derived features
explained in Section 6, and divide the patients by the fixed train/test ids of Section 2.

<!-- code: features -->

```python
train, test = train_test("drop")
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
protective sign. Section 5.2 checks whether this survives a fairer comparison.

### 5.2 The smoking paradox

Smoking is an established cause of cardiovascular disease, so a lower rate among smokers needs an
explanation. The first suspect is a confounder. Smoking in this data is almost entirely male, and
risk rises with age, so the comparison is redone among men, age group by age group. The gap is then
averaged over the age groups (weighted by their size), with a bootstrap confidence interval:

```python
print("share who smoke:", train.groupby("male")["smoke"].mean().rename({0: "women", 1: "men"})
      .map("{:.1%}".format).to_dict())

men = train[train["male"] == 1].copy()
men["age band"] = band(men["age_years"], [0, 45, 50, 55, 60, 70], ["<45", "45-49", "50-54", "55-59", "60+"])
by_age = men.groupby(["age band", "smoke"], observed=True)["cardio"].agg(["mean", "size"]).unstack()
rate = by_age["mean"].rename(columns={0: "non-smokers", 1: "smokers"})


def age_standardised_gap(d, col):
    """CVD rate of the 'yes' group minus the 'no' group within each age band, averaged over the bands
    with the band sizes as weights."""
    r = d.groupby(["age band", col], observed=True)["cardio"].mean().unstack()
    w = d.groupby("age band", observed=True).size()
    return float(((r[1] - r[0]) * w).sum() / w.sum())


rng = np.random.default_rng(SEED)
for col, label in [("smoke", "smokers"), ("alco", "drinkers")]:
    gap = age_standardised_gap(men, col)
    boot = [age_standardised_gap(men.iloc[rng.integers(0, len(men), len(men))], col) for _ in range(1000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"men, {label} minus the others, same age: {gap * 100:+.1f} points (95% CI {lo * 100:+.1f} to {hi * 100:+.1f})")
print(f"men who drink: {men.loc[men['smoke'] == 1, 'alco'].mean():.0%} of smokers, "
      f"{men.loc[men['smoke'] == 0, 'alco'].mean():.0%} of non-smokers")

fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(rate))
ax.vlines(x, rate.min(axis=1), rate.max(axis=1), color="#cfcec8", lw=2, zorder=1)
for col, colour in [("non-smokers", NO_CVD), ("smokers", CVD)]:
    ax.plot(x, rate[col], "o", color=colour, ms=8, label=col, zorder=2)
ax.set_xticks(x, rate.index)
ax.set(xlabel="age (years)", ylabel="share with CVD", title="Men: CVD rate by age group, smokers and non-smokers")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(loc="upper left")
fig.savefig(FIGURES / "fig3_smoking_paradox.png")
plt.show()
by_age.style.format("{:.1%}", subset="mean").format("{:,}", subset="size")
```

For smoking the paradox survives. Among men (women hardly smoke here), smokers have the same or a
lower disease rate than non-smokers in every age group. The gap is small and uncertain in the
youngest and oldest groups, but compared age for age, smokers have 3.1 points less CVD (95% CI 1.5 to 4.9 points less) points less CVD, and
the confidence interval excludes zero. For alcohol it does not survive: the age-standardised gap is
−2.0 (95% CI −4.3 to +0.3) points and its interval includes zero. Smokers also drink far more than non-smokers
(31% against 5%), which this simple comparison does not separate.

A plausible explanation lies in how the data was collected rather than in biology, though the data
cannot test it: it has no "former smoker" answer. The diagnosis and the answers were recorded **at
the same examination**. A patient who already has heart disease may have been told to quit, and then
answers "no" to smoking, which moves diseased former smokers into the non-smoker group. This "sick
quitter" effect was first described for alcohol [8].

Two consequences for the rest of the study follow from how the data was recorded, whatever the
explanation for smoking:

1. The task is **detection of an existing disease**, not prediction of a future one. A model trained
   here can say "this profile looks like the profiles of patients who have CVD". It cannot say "this
   person will develop CVD".
2. Feature effects learned from this data are **associations, not causes**. A model that gives
   smoking a negative weight is describing the data correctly, but no one should be told that smoking
   protects the heart. Section 8 returns to this when the model is explained.

### 5.3 Correlations between features

```python
cols = BASE_FEATURES + ["bmi", "cardio"]
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
themselves, the two blood pressures are strongly correlated (0.73), and so are weight and BMI (0.86);
sex and height moderately (0.52), and cholesterol and glucose too (0.45). For tree models this hardly
matters. For logistic regression it means individual coefficients must be read with care, since
correlated features share their effect.

### 5.4 What the exploration implies for modelling

- The strongest relationships are **monotonic**: the risk rises steadily with blood pressure, age,
  BMI and cholesterol. A linear model on suitable features should capture most of the signal, so
  flexible models have limited room to improve on it. This is the expectation Q1 will test.
- **No class imbalance**: 49.5% of training patients have CVD. Oversampling methods such as SMOTE
  have nothing to fix here. In Section 9 they appear only as one of the mistakes the leakage
  experiments reproduce.
- The lifestyle answers are weak, and for smoking the association is probably distorted by reverse
  causation. They are kept, because a model should use what the examination records, but nothing
  causal is read into them.

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
ax.set_xticks(range(4), BP_STAGE_NAMES)
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

Three points about these features:

- **They are fixed formulas, not learned transformations.** Like the cleaning rules, they cannot
  leak test information.
- **They overlap with the recorded values on purpose.** BMI is a function of height and weight, and
  the three pressure features are functions of the two readings. Tree models gain from having the
  combination precomputed, since a split on BMI replaces many splits on height and weight. For
  logistic regression the overlap is handled by regularisation. Section 7 checks whether the derived
  features help at all, by training the same models with and without them.
- **The stage builds the guideline's thresholds into the data.** It steps at 120, 130 and 140 mmHg
  systolic, so any model can use those thresholds directly. Section 8.4 keeps this in mind when
  reading the shape of the model.

## 7. Models

### 7.1 What is compared

Every candidate is evaluated with 5-fold cross-validation on the training patients, using the same
five folds. All models are implemented with scikit-learn [12]. The test patients are not touched in
this section.

| Candidate | Type | What it uses | Settings searched |
|---|---|---|---|
| Rule: systolic ≥ 130 mmHg | clinical baseline | one reading, guideline threshold | none |
| Systolic pressure alone | clinical baseline | one reading; cut-off chosen on the training folds | none |
| Framingham (published) | clinical baseline | age, sex, BMI, systolic pressure, smoking, diabetes | none: the 2008 equations as published |
| Framingham form (refit) | logistic regression | the same terms, one equation per sex | 1 setting: coefficients refitted on these patients |
| Logistic regression | linear model | all 15 features, standardised | 4 settings: regularisation strength C |
| Random forest | tree ensemble (200 trees) | all 15 features | 9 settings: leaf size, features per split |
| Gradient boosting | tree ensemble (histogram GBDT, early stopping) | all 15 features | 27 settings: learning rate, tree size, leaf size |
| Neural network (MLP) | multilayer perceptron | all 15 features, standardised | 16 settings: layer sizes, weight decay |

The search grids differ in size: the more settings a model has that matter, the more were tried.
The effect of this on the comparison is examined in Section 7.5.

Some notes on the baselines:

- **The Framingham score answers a different question.** It was built to predict a *first*
  cardiovascular event over the next ten years in people free of CVD, while this dataset records
  existing disease. The dataset also has no field for blood-pressure treatment, so the equation for
  untreated patients is used, and no diabetes field, so glucose "well above normal" stands in for
  diabetes. Using its risk to rank these patients is therefore a reference point, not a fair contest.
  The refitted version keeps Framingham's terms but learns the weights from these patients. It shows
  how much of any gap comes from the terms and how much from the weights.
- **Two baselines need a cut-off.** Systolic pressure alone and the published Framingham risk are
  continuous scores, so the cut-off that classifies the four training folds most accurately is
  applied to the fifth. Their AUC needs no cut-off.

**Selection rule, fixed in advance.** The final model is the *simplest* machine-learning model whose
mean cross-validated AUC is within one standard error of the best one. The order of simplicity: the
Framingham form and logistic regression first (a few coefficients that can be read directly), then
the tree ensembles from smaller to larger (gradient boosting builds small trees, at most 31 leaves
each; random forest grows 200 deep ones), and the neural network last, the hardest to inspect. A
difference smaller than the fold-to-fold noise is not evidence, so it should not buy complexity.

### 7.2 Cross-validated comparison

The baselines and models of Section 7.1, and the cross-validated comparison that scores them all on
the same five folds:

<!-- code: clinical baselines -->

<!-- code: models -->

```python
import time

t0 = time.time()
comparison, searches = compare_models(train)
print(f"{len(searches)} grid searches and 3 baselines, {CV_FOLDS}-fold CV: {time.time() - t0:.0f} s")
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
- **The flexible models are close.** Gradient boosting (0.8005), the neural network (0.7996) and random forest (0.7992)
  lie within 0.0013 AUC of each other, less than the fold-to-fold standard deviation (about 0.003).
  Logistic regression is 0.010 behind, so the curves and interactions that trees and networks can
  represent are worth a little, but only a little.
- **The guideline's threshold is also the most accurate cut-off.** When systolic pressure alone is
  given the cut-off that classifies the training folds most accurately, the search lands on 130 mmHg,
  the guideline's own value, in every fold: the accuracy, sensitivity and specificity are identical to
  the rule's. (Almost all readings are multiples of 10, so the search chooses among a handful of
  values; this is a consistency check, not a discovery.) The AUC differs only because the rule turns
  the reading into a yes/no answer.
- **The published Framingham score ranks these patients worse than systolic pressure alone**
  (AUC 0.696 vs 0.751). Refitting the same five terms on these patients raises it to
  0.780: the terms are informative, but the published weights do not transfer to this
  population and question (existing disease here, a first event within ten years in Framingham).
  Section 8.1 shows where they go wrong.

How flat is each search? If the best setting is barely better than its neighbours, nothing much
hinges on the grid:

```python
flat = []
for name, s in searches.items():
    scores = np.sort(s.cv_results_["mean_test_auc"])[::-1]
    flat.append({"model": name, "settings tried": len(scores), "best": scores[0],
                 "second best": scores[1] if len(scores) > 1 else np.nan,
                 "median setting": float(np.median(scores)), "worst": scores[-1],
                 "best setting": comparison.loc[name, "best setting"]})
(pd.DataFrame(flat).set_index("model")
 .style.format("{:.4f}", subset=["best", "second best", "median setting", "worst"], na_rep="-"))
```

The searches are flat where it matters. All 27 gradient-boosting settings land between 0.7995 and
0.8005, and its two best settings are tied. Its best learning rate is the smallest tried (0.01), but
early stopping ended that model after 593 of the 1,000 trees allowed, so a smaller rate would mostly
have needed more trees. The best random-forest and neural-network settings sit inside their grids,
which were widened after a first run found the best settings at their edges. A wider search is
therefore unlikely to change the comparison.

The cross-validated score of each model's *best* setting is slightly optimistic, because that
setting was chosen on the same folds. With these grids and 55,000 patients the bias is small, but it
is one reason the test set in Section 8 has the final word.

### 7.3 Do the derived features help?

Section 6 added BMI, pulse pressure, mean arterial pressure and the blood-pressure stage. Here the
best settings of logistic regression and gradient boosting are cross-validated again, with and
without them:

```python
features_effect = feature_set_comparison(train, searches, ["Logistic regression", "Gradient boosting"])
features_effect["gain"] = features_effect["recorded + derived"] - features_effect["recorded only"]
features_effect.style.format("{:.4f}")
```

The derived features change the AUC by less than 0.001 in either direction (+0.0003 for gradient
boosting, −0.0005 for logistic regression), well inside the noise. Both models can build what they need from the recorded values on their own, which is a
common finding with tabular data this simple. The derived features stay in the final model anyway:
they cost nothing in accuracy, and they make the explanations in Section 8 easier to read. A clinician
thinks in BMI and blood-pressure stages, not in height and weight.

### 7.4 Would more data help?

The same models are trained on growing shares of a training fold. If the curve is still rising at
the full training set, more patients would help. If it has levelled off, the limit lies elsewhere.

```python
curve = learning_curve_auc(train, {name: searches[name].best_estimator_
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

- **Logistic regression levels off early.** Its validation AUC gains only 0.003 between 2,000 and
  44,000 patients.
- **Gradient boosting keeps improving, but slowly.** It needs more patients to estimate its curves and
  interactions, overtakes logistic regression at around 2,000 patients, and keeps improving. But the
  last doubling, from 22,000 to 44,000 patients, adds only 0.0015.

More patients of the same kind would therefore add little. Together with the closeness of very
different algorithms (Section 7.2) and the flatness of their searches, this suggests that the limit
near 0.80 AUC (about 73% accuracy) lies mainly in what the eleven recorded values can tell, not in
the amount of data or the choice of algorithm. It is a reasoned reading, not a proof: other models
or other features could still do somewhat better.

### 7.5 The chosen model

```python
chosen = choose_model(comparison)
ml = comparison[comparison["kind"] == "machine learning"]
best = ml["auc"].idxmax()
se = ml.loc[best, "auc_sd"] / np.sqrt(CV_FOLDS)
print(f"best cross-validated AUC: {best} ({ml.loc[best, 'auc']:.4f}, fold SD {ml.loc[best, 'auc_sd']:.4f})")
print(f"one standard error: {se:.4f}; within it of the best: {', '.join(ml.index[ml['auc'] >= ml.loc[best, 'auc'] - se])}")
print(f"chosen by the one-standard-error rule: {chosen}, setting: {ml.loc[chosen, 'best setting']}")
gb, rf = searches["Gradient boosting"].best_estimator_, searches["Random forest"].best_estimator_
print(f"model size: gradient boosting {gb.n_iter_} trees of at most {gb.max_leaf_nodes} leaves; "
      f"random forest {len(rf.estimators_)} trees with {sum(t.tree_.n_leaves for t in rf.estimators_):,} leaves in all")
```

Three models are within one standard error (0.0013) of the best: gradient boosting, the neural
network and random forest. The rule fixed in advance then takes the simplest of the tied models, and
in its order gradient boosting comes first: it is far smaller than the forest (593 trees of at most
15 leaves, under 9,000 leaves in all, against 114,770), and far easier to inspect than the network.
**Gradient boosting is the final model.** The choice matters little in practice: Section 8.2 shows
how close the three remain on the test set.

## 8. The final model on the test set

### 8.1 Every candidate, scored once

The 13,722 test patients have not been used for anything so far. Each candidate is scored on them
exactly once:

- The machine-learning models use their selected settings, refitted on all training patients.
- The baselines that need a cut-off use the one chosen on the training patients.
- Confidence intervals come from 2,000 bootstrap resamples of the test patients, the same resamples
  for every candidate.

The helpers for the test-set evaluation: paired bootstrap intervals, out-of-fold predictions, and the
cut-offs chosen on training data.

<!-- code: evaluation -->

```python
from sklearn.metrics import roc_auc_score

final = searches[chosen].best_estimator_
y_test = test["cardio"]
RULE = f"Rule: systolic ≥ {SBP_RULE} mmHg"
test_scores = {
    RULE: (test["ap_hi"] >= SBP_RULE).astype(float),
    "Systolic pressure alone": test["ap_hi"].astype(float),
    "Framingham (published)": framingham_risk(test),
    **{name: s.predict_proba(test[FEATURES])[:, 1] for name, s in searches.items()},
}
cutoffs = {name: 0.5 for name in test_scores}
cutoffs["Systolic pressure alone"] = accuracy_threshold(train["cardio"], train["ap_hi"])
cutoffs["Framingham (published)"] = accuracy_threshold(train["cardio"], framingham_risk(train))
predictions = {name: (np.asarray(score) >= cutoffs[name]).astype(int) for name, score in test_scores.items()}

auc_boot = bootstrap(y_test, test_scores, roc_auc_score)
acc_boot = bootstrap(y_test, predictions, lambda y, p: (y == p).mean())
rows = []
for name, score in test_scores.items():
    m = classification_metrics(y_test, score, cutoffs[name])
    lo, hi = interval(auc_boot[name])
    alo, ahi = interval(acc_boot[name])
    rows.append({"model": name, "test AUC": m["auc"], "AUC 95% CI": f"{lo:.3f}-{hi:.3f}",
                 "CV AUC": comparison.loc[name, "auc"], "accuracy": m["accuracy"],
                 "accuracy 95% CI": f"{alo:.1%}-{ahi:.1%}",
                 "sensitivity": m["sensitivity"], "specificity": m["specificity"]})
test_table = pd.DataFrame(rows).set_index("model").sort_values("test AUC", ascending=False)
test_table.to_csv(ROOT / "outputs" / "test_results.csv")
gain_lo, gain_hi = interval(acc_boot[chosen] - acc_boot[RULE])
print(f"{chosen} minus the rule, accuracy: {(test_table.loc[chosen, 'accuracy'] - test_table.loc[RULE, 'accuracy']) * 100:+.1f} "
      f"points (95% CI {gain_lo * 100:+.1f} to {gain_hi * 100:+.1f})")
test_table.style.format({"test AUC": "{:.3f}", "CV AUC": "{:.3f}", "accuracy": "{:.1%}",
                         "sensitivity": "{:.1%}", "specificity": "{:.1%}"})
```

**The test set confirms the cross-validation.** Every candidate's test AUC is within 0.004 of its
cross-validated AUC. On patients it has never seen, gradient boosting reaches an AUC of 0.804 (95% CI
0.797-0.811) and 73.6% accuracy (72.9-74.4%), against 71.3% for the systolic ≥ 130 rule: a gain of
2.3 points (95% CI 1.8 to 2.9). The test set re-orders the two runners-up (random forest 0.803, the
neural network 0.800), a reminder that their cross-validated differences were within the noise.

Why does the published Framingham score rank these patients so poorly? Its weights can be checked
one at a time:

```python
fram = framingham_risk(test)
rows = []
for label, mask in [("women", test["male"] == 0), ("men", test["male"] == 1), ("everyone", test["male"] >= 0)]:
    rows.append({"patients": label, "share with CVD": y_test[mask].mean(), "mean Framingham risk": fram[mask].mean(),
                 "Framingham AUC": roc_auc_score(y_test[mask], fram[mask]),
                 "systolic alone AUC": roc_auc_score(y_test[mask], test.loc[mask, "ap_hi"])})
print("published Framingham AUC with the smoking term switched off:",
      f"{roc_auc_score(y_test, framingham_risk(test.assign(smoke=0))):.3f}")
(pd.DataFrame(rows).set_index("patients")
 .style.format({"share with CVD": "{:.1%}", "mean Framingham risk": "{:.1%}", "Framingham AUC": "{:.3f}",
                "systolic alone AUC": "{:.3f}"}))
```

Two of the published weights go wrong for these patients. **Sex:** the equations give men about
twice the risk of women (a mean of 15.9% against 7.0%), but CVD is equally common in both sexes here
(49.9% and 49.2%). Within women, the published score ranks patients about as well as systolic
pressure alone (AUC 0.752 against 0.756); within men it does worse (0.695 against 0.752).
**Smoking:** the score adds risk for smokers, while smokers in this data have slightly less CVD
(Section 5.2); switching the smoking term off raises the overall AUC from 0.693 to 0.709. The
published equations were built for a different population and a different question, and their
weights must be re-estimated before they are used on data like this.

```python
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(5.6, 5.2))
for name, colour, lw in [(chosen, NO_CVD, 2.2), ("Logistic regression", MUTED, 1.4),
                         ("Systolic pressure alone", "#9b9a94", 1.4), ("Framingham (published)", "#c9c8c1", 1.4)]:
    fpr, tpr, _ = roc_curve(y_test, test_scores[name])
    ax.plot(fpr, tpr, color=colour, lw=lw, label=f"{name} (AUC {roc_auc_score(y_test, test_scores[name]):.3f})")
rule = classification_metrics(y_test, test_scores[RULE], 0.5)
ax.plot(1 - rule["specificity"], rule["sensitivity"], "D", color=CVD, ms=8, label=f"rule: systolic ≥ {SBP_RULE}")
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=MUTED)
ax.set(xlabel="1 − specificity (healthy patients flagged)", ylabel="sensitivity (diseased patients detected)",
       title="ROC curves on the test patients", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="lower right", fontsize=8)
fig.savefig(FIGURES / "fig8_roc_test.png")
plt.show()
```

The rule is one point on the ROC plane; the model is a whole curve. To compare them at the rule's own
operating point without touching the test set, the model's cut-off is chosen on the training
patients: the lowest cut-off at which the model, using out-of-fold predictions, clears at least as
many healthy training patients as the rule does. Both are then scored once on the test patients:

```python
oof = out_of_fold(final, train)
rule_specificity_train = (train.loc[train["cardio"] == 0, "ap_hi"] < SBP_RULE).mean()
cut_rule = threshold_at_specificity(train["cardio"], oof, rule_specificity_train)
flags_model = (np.asarray(test_scores[chosen]) >= cut_rule).astype(int)
flags_rule = predictions[RULE]
y = y_test.to_numpy()
detected = bootstrap(y, {"model": flags_model, "rule": flags_rule}, lambda t, f: int(((f == 1) & (t == 1)).sum()))
flagged = bootstrap(y, {"model": flags_model, "rule": flags_rule}, lambda t, f: int(((f == 1) & (t == 0)).sum()))
more = int(((flags_model == 1) & (y == 1)).sum() - ((flags_rule == 1) & (y == 1)).sum())
extra = int(((flags_model == 1) & (y == 0)).sum() - ((flags_rule == 1) & (y == 0)).sum())
lo, hi = interval(detected["model"] - detected["rule"])
flo, fhi = interval(flagged["model"] - flagged["rule"])
print(f"cut-off chosen on training patients: {cut_rule:.3f} (rule's specificity there: {rule_specificity_train:.1%})")
for label, f in [("rule", flags_rule), (chosen, flags_model)]:
    m = classification_metrics(y, f, 0.5)
    print(f"  {label:<20} test sensitivity {m['sensitivity']:.1%}, specificity {m['specificity']:.1%}")
print(f"diseased test patients detected by the model and not the rule: {more:+,} (95% CI {lo:+.0f} to {hi:+.0f}) "
      f"of {int(y.sum()):,}")
print(f"healthy test patients flagged, model minus rule: {extra:+,} (95% CI {flo:+.0f} to {fhi:+.0f}) "
      f"of {int((y == 0).sum()):,}")
```

At an operating point chosen without the test set, the model detects 65.8% of the diseased test
patients where the rule detects 61.8%, at a slightly higher specificity (81.0% against 80.6%). That
is 274 more of the 6,785 diseased test patients (95% CI 220 to 332), with 29 fewer healthy patients
flagged (95% CI 76 fewer to 19 more: no clear difference). This is the most concrete form of the
answer to Q1: about one more sick patient found for every 25 diseased patients examined, at no clear
cost in false alarms.

### 8.2 How large are the differences?

The same bootstrap resamples score every model, so each AUC difference can be given its own
interval. These intervals cover the sampling of test patients only: each model was fitted once, with
one seed. Five comparisons are made at once, so a Bonferroni-adjusted interval (99%) is shown too.

```python
pairs = [(chosen, other) for other in ["Random forest", "Neural network (MLP)", "Logistic regression",
                                        "Framingham form (refit)", "Systolic pressure alone"] if other != chosen]
diffs = []
for a, b in pairs:
    d = auc_boot[a] - auc_boot[b]
    lo, hi = interval(d)
    blo, bhi = interval(d, level=1 - 0.05 / len(pairs))
    diffs.append({"comparison": f"{a} − {b}",
                  "AUC difference": roc_auc_score(y_test, test_scores[a]) - roc_auc_score(y_test, test_scores[b]),
                  "95% CI": f"{lo:+.4f} to {hi:+.4f}", f"{1 - 0.05 / len(pairs):.0%} CI (adjusted)": f"{blo:+.4f} to {bhi:+.4f}"})
pd.DataFrame(diffs).set_index("comparison").style.format({"AUC difference": "{:+.4f}"})
```

- **Over random forest and the neural network**, the lead is small: 0.0015 and 0.0040 AUC. The
  random-forest interval excludes zero at 95% but not once the five comparisons are adjusted for; the
  lead over the network survives the adjustment but is still only 0.004. None of these differences
  would change a clinical decision.
- **Over logistic regression** the lead is 0.012 (95% CI 0.009 to 0.014). Small, but consistent: the
  trees capture something a linear model on these features does not (Section 8.4 suggests what).
- **Over Framingham's terms refitted** (0.024) and **over systolic pressure alone** (0.050), the gaps
  are clear.

### 8.3 Choosing the threshold

A predicted risk becomes a verdict only through a cut-off. At 0.5 the model maximises accuracy. A
screening programme usually cares more about not missing sick patients, so a second cut-off is
chosen **on the training patients**: the highest one that detects at least 80% of the diseased
training patients, using predictions from models that did not see them (the out-of-fold predictions
of Section 8.1).

```python
from sklearn.metrics import confusion_matrix

cut_screen = screening_threshold(train["cardio"], oof, 0.80)
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
    m = classification_metrics(y_test, risk_test, cut)
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

Three complementary views. **Grouped permutation importance** asks how much test AUC is lost when a
factor is scrambled across patients, that is, how much this fitted model leans on it. Related
features are scrambled together, so the blood-pressure features cannot cover for one another.
**Retraining without a factor** asks a different question: what a model of the same kind can do
without it. **Partial dependence** asks how the average predicted risk changes as one factor is
varied for everyone.

<!-- code: explanation -->

```python
importance = grouped_permutation_importance(final, test)

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

```python
retrained = drop_group_importance(final, train)
retrained.style.format("{:.4f}")
```

**Blood pressure dominates.** Scrambling the five blood-pressure features costs this model 0.210 test
AUC; age (0.036) and cholesterol (0.030) follow, body size adds a little (0.007), and glucose, the
three lifestyle answers and sex together contribute almost nothing. Retraining tells the same story
in different numbers. A model built without blood pressure still reaches a cross-validated AUC of
0.702: the other factors carry real information, they are simply weaker. Leaving out age costs
0.020, cholesterol 0.009, and every other factor about 0.002 or less. The two views agree on the
order, which matches the exploratory analysis in Section 5.

```python
curves = [("ap_hi", range(100, 181, 10), "systolic pressure (mmHg)"),
          ("age_years", range(40, 66, 2), "age (years)"),
          ("weight", range(50, 121, 5), "weight (kg)"),
          ("cholesterol", [1, 2, 3], "cholesterol (1 normal - 3 well above)"),
          ("gluc", [1, 2, 3], "glucose (1 normal - 3 well above)"),
          ("smoke", [0, 1], "smokes (0 no, 1 yes)"),
          ("active", [0, 1], "physically active (0 no, 1 yes)"),
          ("alco", [0, 1], "drinks alcohol (0 no, 1 yes)")]
sample = test.sample(3000, random_state=SEED)

fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
for ax, (feature, values, label) in zip(axes.flat, curves):
    pdc = partial_dependence_curve(final, sample, feature, list(values))
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

The partial-dependence curves show the *shape* of each effect:

- **Systolic pressure works more like a switch than a slope.** The average predicted risk is about
  31-39% from 100 to 120 mmHg, rises to 54% at 130 mmHg and 80% at 140 mmHg, then levels off at about
  84%. Two things make the steps sharper than the biology need be: the stage feature builds the
  guideline's thresholds into the data (Section 6), and almost every reading is a multiple of 10.
  Still, the shape is clearly not a straight line, and a logistic regression, which adds each
  feature's effect along a straight line in log-odds, can only approximate it. This may explain part
  of its 0.012 AUC gap. (At 100-110 mmHg the curve also lowers some diastolic readings, so that end is
  not a pure systolic effect.)
- **Age, weight and cholesterol raise the risk steadily.** Age goes from about 38% at 40 to 65% at 64,
  weight from about 41% at 50 kg to 58% at 120 kg, and cholesterol "well above normal" adds about 25
  percentage points.
- **Glucose, smoking and alcohol are flat or slightly *protective*,** and physical activity slightly
  lowers the risk. For smoking this is the pattern of Section 5.2. For glucose it probably reflects
  its correlation with cholesterol (0.45): its rate rises on its own (Section 5.1), but once
  cholesterol is known the model gives it no extra weight. **These curves describe associations in
  this dataset and must not be read as causal effects**: nothing here suggests that smoking is
  harmless.

### 8.5 A points score a doctor can use on paper

If the best model gains only a little over simpler ones, the most useful result might be a
transparent score. Here each factor is cut into clinical bands, a logistic regression is fitted on
the bands, and its coefficients are rounded to whole points (one point = 0.1 in log-odds). The total
then maps to a risk: the share of training patients with CVD at that total, smoothed so that more
points always mean more risk.

<!-- code: points score -->

```python
points = PointsScore().fit(train, train["cardio"])
card = points.table()
card.set_index(["factor", "band"])
```

The scorecard reads like the model: systolic pressure earns up to 27 points, age up to 11,
cholesterol up to 11, and BMI up to 4. Three factors earn *negative* points: smoking (−2), alcohol
(−2) and glucose "well above normal" (−3). No clinician could hand out a score that rewards smoking
or high glucose. For smoking this is probably the reverse causation of Section 5.2; for glucose, its
overlap with cholesterol (Section 8.4). Physical activity earns −2, the protective direction medicine
expects.

So a **clinical version** is built by a rule stated here: keep the factors that are *measured* at the
examination and whose points in this training fit go the way medicine expects. That removes the three
self-reported answers, whatever their sign (they are the answers most open to reverse causation), and
glucose, and keeps age, systolic pressure, BMI and cholesterol. The rule was set after seeing the full
card's points, which come from the training patients only; the test set played no part in it.

```python
clinical = PointsScore(factors=CLINICAL_FACTORS).fit(train, train["cardio"])
test_scores["Points score (all factors)"] = points.predict_proba(test)
test_scores["Points score (clinical)"] = clinical.predict_proba(test)
cards = ["Points score (all factors)", "Points score (clinical)"]
boot_cards = bootstrap(y_test, {k: test_scores[k] for k in cards + [chosen]}, roc_auc_score)
for name in cards + ["Logistic regression", chosen]:
    m = classification_metrics(y_test, test_scores[name], 0.5)
    ci = "(95% CI {:.3f}-{:.3f})".format(*interval(boot_cards[name])) if name in boot_cards else ""
    print(f"{name:<28} AUC {m['auc']:.3f} {ci:<20} accuracy {m['accuracy']:.1%}")
lo, hi = interval(boot_cards[chosen] - boot_cards["Points score (clinical)"])
gap = roc_auc_score(y_test, test_scores[chosen]) - roc_auc_score(y_test, test_scores["Points score (clinical)"])
print(f"{chosen} minus the clinical card: AUC {gap:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
clinical.table().set_index(["factor", "band"]).T
```

```python
totals = clinical.total(test)
observed = test.groupby(totals)["cardio"].agg(["mean", "size"])
observed = observed[observed["size"] >= 50]
grid = np.arange(totals.min(), totals.max() + 1)
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.scatter(observed.index, observed["mean"], s=np.sqrt(observed["size"]) * 2, color=NO_CVD, alpha=0.8,
           label="observed share with CVD (test patients; totals with ≥ 50)")
ax.plot(grid, clinical.risk_by_total(grid), color=CVD, lw=2, label="risk the card assigns (from training patients)")
ax.set(xlabel="total points", ylabel="share with CVD", title="Clinical points score: total points against observed risk")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(loc="upper left", fontsize=8)
fig.savefig(FIGURES / "fig12_points_score.png")
plt.show()
```

**Dropping the self-reported answers and glucose costs almost nothing.** The four-factor clinical
card reaches an AUC of 0.793 (95% CI 0.785-0.800) and 72.4% accuracy: the same AUC as logistic
regression on all fifteen features, with 0.2 points less accuracy, and 0.012 AUC below gradient
boosting (95% CI 0.009 to 0.014). The full card does a little better (0.795, 73.0%), as it should
with more factors. The card's risks come straight from the training patients' observed rates at each
total, so they follow the test patients' rates closely in the figure (Section 9.3 checks them
formally). A doctor with a pen can add up four numbers (age, systolic pressure, BMI, cholesterol) and
get within about one point of the best model's accuracy. This is in line with Rudin's argument that
for tabular problems like this one, an interpretable model is often nearly as accurate as a black box
[9].

## 9. Reliability

Section 8 measured how well the model works. This section asks how far those numbers can be
trusted: how they depend on data-handling choices (Q2), and whether the predicted risks hold up as
risks, for different kinds of patients (Q3).

### 9.1 Does the cleaning strategy matter?

The final gradient-boosting setting and the logistic regression are trained three times: on the raw
training data, on the `drop` training data and on the `repair` training data. All six models are
scored on the same test patients (those whose records pass every check), so only the training data
differs.

The reliability checks of this section: the cleaning-strategy comparison, the test patients that
`drop` removes, a minimal SMOTE, the three deliberate mistakes, subgroup metrics and predictive value
at other prevalences.

<!-- code: reliability -->

```python
strategies = strategy_comparison({"Gradient boosting": final, "Logistic regression": searches["Logistic regression"].best_estimator_})
strategies.set_index(["model", "strategy"]).style.format(
    {"training patients": "{:,}", "test AUC": "{:.4f}", "test accuracy": "{:.1%}", "AUC change vs drop": "{:+.4f}"})
```

The answer depends on the model, as Section 4.3 predicted:

- **Gradient boosting barely notices.** Trained on the raw data it loses 0.0008 AUC (95% CI 0.0001 to
  0.0014): detectable with 13,722 patients, but negligible. Trees split on the order of values, so a
  pressure of 16,020 mmHg simply lands in the top branch with every other high reading.
- **Logistic regression is hurt by the raw data.** Its AUC falls by 0.017 (95% CI 0.014 to 0.019),
  from 0.793 to 0.776. A handful of absurd values pull its straight-line coefficient for blood
  pressure towards zero.
- **Repairing and dropping give the same result** for both models (differences under 0.001). The
  repaired records are under 2% of the data, too few to change what a model learns. Repair keeps more
  patients, but its value lies in not discarding real people rather than in accuracy.

The practical lesson: cleaning is essential for linear models and for any analysis that reads
coefficients or correlations (Section 4.3), even when a tree model would have survived without it.

The common test set leaves out the test patients whose records fail a check. The protocol promised
to report how the final model does on them:

```python
excluded = excluded_test_patients(final)
excluded.style.format({"patients": "{:,}", "share with CVD": "{:.1%}", "AUC": "{:.3f}", "accuracy": "{:.1%}"},
                      na_rep="-")
```

The 274 test patients that `drop` removes are very different from the rest: 77% of them have CVD,
against 49% of the valid test patients. The `repair` rules recover 257 of them (78% CVD). Scored on
those repaired records, the final model, trained on valid records only, reaches an AUC of 0.707 (on
only 257 patients, an imprecise figure) and 82.5% accuracy; the accuracy is high mostly because most
of these patients are sick and the model flags most of them. Added to the main test set, they move
the headline slightly up (AUC 0.806, accuracy 73.8%), so the results of Section 8 are, if anything, a
little conservative. Within these records, though, the model separates sick from healthy less well
than in the main test set.

### 9.2 How mistakes inflate reported accuracy

Papers on this dataset report accuracies from 73% [5] to 94.5% [7]. This experiment reproduces three
mistakes common in published machine-learning work on medical data and measures the accuracy each
one would let a study report, next to the honest figure for the same model. These pipelines are
wrong on purpose, and none of them feeds into the model chosen above. They show what *can* produce
high accuracies on this data; which mistakes, if any, particular papers made cannot be known without
their code. Leakage is not the only route to a flattering figure either: removing hard records from
the test set, or a small test set, can also raise a reported accuracy.

```python
t0 = time.time()
correct_acc = classification_metrics(y_test, test_scores[chosen], 0.5)["accuracy"]
leakage = leakage_experiments(train, test, correct_acc)
print(f"leakage experiments: {time.time() - t0:.0f} s")
leakage.style.format({"reported accuracy": "{:.1%}"}).hide(axis="index")
```

The same data can "report" anything from 64% to 99.9% accuracy, depending on how the evaluation is
set up:

1. **Accuracy measured on the training patients.** A forest of fully grown trees memorises its
   training patients and scores 99.9% on them, against 71.0% on patients it has never seen. Any
   figure that does not come from held-out patients says nothing about the model.
2. **SMOTE before the split** [10]. Synthetic patients are interpolated between real neighbours, so
   after a random split most test patients have a near-twin in the training set. A one-nearest-
   neighbour classifier then "reaches" 85.3% and the forest 86.7%. On unseen patients the same models
   score 64.4% and 71.0% without oversampling, and 64.0% and 70.5% with SMOTE done correctly (on the
   training patients only): oversampling brings nothing here, since the classes are already
   balanced, and the 15-21-point "gain" comes entirely from the leak. A published comment traced a
   reported 99% accuracy on another cardiovascular dataset (BRFSS 2021) largely to SMOTE-ENN applied
   before the split, together with a 2-nearest-neighbour model; with resampling on the training data
   only, that model reached 77% [11].
3. **Trying many random splits and reporting the best one.** This is the mildest mistake: the best
   of 50 splits is 73.4%, against 72.8% for the median split. With 14,000 test patients the luck of
   the split is worth less than one point. It matters far more in small datasets.

None of this shows what the authors of [6] or [7] actually did; their papers would have to be
reproduced to know. What it does show is that two common mistakes can turn 73% into 85-100% on this
very dataset, while the learning curve (Section 7.4) gives no reason to expect an honest model to gain
15 or 20 points. An accuracy far above 74% on this dataset is a reason to check the evaluation before
believing it.

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
for name, colour in [(chosen, NO_CVD), ("Logistic regression", MUTED), ("Points score (clinical)", CVD)]:
    observed, predicted = calibration_curve(y_test, test_scores[name], n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_test, test_scores[name])
    ax.plot(predicted, observed, "-o", color=colour, lw=1.8, ms=4, label=f"{name} (Brier {brier:.4f})")
ax.set(xlabel="predicted risk (test patients, 10 equal-sized groups)", ylabel="share with CVD",
       title="Calibration on the test patients", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="upper left", fontsize=8)
fig.savefig(FIGURES / "fig13_calibration.png")
plt.show()
```

**Gradient boosting's risks can be read as risks, in this sample.** Its ten points lie close to the
diagonal: patients given about 20% have CVD about 20% of the time, and patients given 80% about 80% of
the time. It also has the lowest Brier score (0.179). Logistic regression bends around the diagonal:
it overstates the lower risks and understates those around 60-70%, the error of a straight line
forced through a curved relationship (Section 8.4). The clinical card's risks, read from the training
patients' rates at each total, are close to calibrated too (Brier 0.184), with a small excess of
observed cases around 35-45%. All of these risks hold for a population in which half the patients are
sick (Section 9.5).

### 9.4 How does the model do for different patients?

```python
subgroups = subgroup_table(test, test_scores[chosen])
rng = np.random.default_rng(SEED)
y_arr, risk_arr, male = y_test.to_numpy(), np.asarray(test_scores[chosen]), test["male"].to_numpy()
sex_gaps = []
for _ in range(1000):
    i = rng.integers(0, len(y_arr), len(y_arr))
    t, r, m = y_arr[i], risk_arr[i], male[i]
    sex_gaps.append(roc_auc_score(t[m == 1], r[m == 1]) - roc_auc_score(t[m == 0], r[m == 0]))
lo, hi = np.percentile(sex_gaps, [2.5, 97.5])
print(f"AUC, men minus women: {subgroups.loc['men', 'AUC'] - subgroups.loc['women', 'AUC']:+.3f} "
      f"(95% CI {lo:+.3f} to {hi:+.3f})")
subgroups.style.format({"patients": "{:,}", "share with CVD": "{:.1%}", "mean predicted risk": "{:.1%}",
                        "AUC": "{:.3f}", "systolic alone AUC": "{:.3f}", "sensitivity": "{:.1%}",
                        "specificity": "{:.1%}"})
```

- **Women and men are served similarly.** Their AUCs are 0.803 and 0.808; the difference (+0.005, 95%
  CI −0.011 to +0.020) is within the noise, and in both groups the mean predicted risk is within 1
  point of the observed share.
- **Discrimination is lower in older groups, for every method.** The model's AUC falls from 0.835
  under 50 to 0.768 at 50-59 and 0.718 at 60 and over, and systolic pressure alone falls the same way
  (0.796, 0.730, 0.681). So this reflects the information more than the model: blood pressure
  separates sick from healthy less well among older patients. AUCs of these groups are also not like
  for like: the groups differ in prevalence (37%, 51%, 68%), and within a narrow age band age itself
  no longer separates anyone, which lowers any model's AUC there.
- **The risks hold in every group on average.** The mean predicted risk is within 1.5 points of the
  observed share in each group (calibration-in-the-large; the full calibration curve of Section 9.3
  covers all patients). Because most patients over 60 are sick (68%), a calibrated model flags most of
  them at the 0.5 cut-off (88% sensitivity, 35% specificity), while under 50 it is the reverse (59%
  and 91%). That is the same cut-off applied to different risks, not a different rule for different
  ages; whether older patients should be referred at a different risk is a clinical question about
  costs, not a statistical one.

### 9.5 What the numbers mean outside this dataset

Half the patients in this dataset have CVD. In the population that attends a routine check-up the
share is much lower, and this changes what a positive prediction means. Sensitivity and specificity
describe the model; they are assumed here to carry over to a population with a similar mix of
patients, which is a simplification (a milder case mix would make sick patients harder to detect).
The **positive predictive value**, the chance that a flagged patient really has CVD, depends on how
common the disease is. These are point estimates:

```python
m_acc = classification_metrics(y_test, test_scores[chosen], 0.5)
m_scr = classification_metrics(y_test, test_scores[chosen], cut_screen)
ppv = pd.DataFrame({f"{label} (sens {m['sensitivity']:.0%}, spec {m['specificity']:.0%})":
                    {f"{p:.0%}": ppv_at_prevalence(m["sensitivity"], m["specificity"], p)
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

The project's goal was a model that checks a patient for cardiovascular disease. The code below
turns the chosen model into that check. It saves the model with its two cut-offs and the clinical
scorecard to `outputs/model.joblib`. It also checks a new examination with the same plausibility
rules as the data, and only for ages the model was trained on (30 to 65), so an impossible or
out-of-range input is rejected instead of scored. It then reports the risk, the verdict and the
points.

<!-- code: using the model -->

```python
print("selected setting:", {k: v for k, v in final.get_params().items() if k in FINAL_PARAMS})
bundle = build_final(train, model=final)
print("saved:", save_final(bundle).relative_to(ROOT))
print(f"screening cut-off: {bundle['screening_cutoff']:.3f}")
```

Three invented examination records show the output:

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
    r = assess(bundle, patient(**values))
    rows.append({"patient": label, "predicted risk": r["risk"], "verdict": r["verdict"],
                 "screening": r["screening"], "points": r["points"], "risk from points": r["points_risk"]})
pd.DataFrame(rows).set_index("patient").style.format({"predicted risk": "{:.0%}", "risk from points": "{:.0%}"})
```

The three invented patients get the answers a clinician would expect. The 62-year-old man with
stage 2 hypertension, obesity and high cholesterol is at 87% (the clinical card gives him 50 points,
85%). The young woman with normal results is at 14% (0 points, 8%). The 55-year-old woman in between
is at 59% (26 points, 68%), so she is referred at either cut-off. The card and the model rank the
three patients in the same order.

```python
for values in [dict(age=50, sex="male", height=175, weight=80, systolic=1400, diastolic=90),
               dict(age=80, sex="female", height=160, weight=60, systolic=120, diastolic=80)]:
    try:
        patient(**values)
    except ValueError as e:
        print("rejected:", e)
```

The saved file holds the model, the two cut-offs and the scorecard (the scorecard as plain
numbers). With this notebook's code loaded, `assess(load_final(), patient(...))` gives the same
answer in a later session, without retraining.

## 11. Conclusion

**Objective met.** A gradient-boosting model detects cardiovascular disease from the eleven values of
a routine examination. On 13,722 patients it never saw, it reaches an AUC of 0.804 (95% CI
0.797-0.811) and 73.6% accuracy (72.9-74.4%). Its risks are well calibrated in this sample, and it
performs similarly for women and men. Section 10 packages it as a check of a new examination that
rejects impossible values and returns a risk, a verdict and a points breakdown.

**Q1. What machine learning adds over what a doctor already has: something real, but small.**

- **Over the guideline rule.** The model beats systolic pressure ≥ 130 mmHg by 2.3 points of accuracy
  (95% CI 1.8 to 2.9). At an operating point matched to the rule's specificity on training data, it
  detects 274 more of the 6,785 diseased test patients (95% CI 220 to 332), with no clear difference
  in false alarms.
- **Between the models.** Gradient boosting, random forest and the neural network are within the
  cross-validation noise of each other and within 0.004 AUC on the test set; logistic regression is
  0.012 AUC behind.
- **The scorecard.** A four-factor paper score comes within 0.012 AUC and about one point of accuracy
  of the best model.
- **Why the gain is small.** The learning curves level off, very different algorithms end up close
  together, and the searches are flat: the limit near 0.80 AUC (about 73% accuracy) seems to lie
  mainly in what the examination records, rather than in the algorithm or the amount of data.
- **The Framingham score.** As published, it ranks these patients worse than blood pressure alone,
  mostly because of its sex and smoking weights; refitted to these patients, its terms become
  competitive.

**Q2. Data handling can change the reported performance far more than the choice of algorithm.**

- **Cleaning.** Fewer than 2% of records hide the strongest predictor from any linear analysis, and
  cost logistic regression 0.017 AUC. Tree models barely notice the dirty data, and repairing the
  records is as good as dropping them. The removed records are mostly from sick patients; adding back
  the repairable ones changes the headline very little.
- **Leakage.** Two common evaluation mistakes turn the same data into reported accuracies of 85-100%.
  Claims far above 74% on this dataset call for a close look at how they were evaluated.

**Q3. The predictions can be trusted, within limits.**

- **Calibration.** The risks are calibrated in this sample, and on average within every sex and age
  group.
- **What drives them.** Blood pressure, age and cholesterol dominate, in the direction medicine
  expects.
- **Limits.** Discrimination is lower among older patients, for every method including the guideline
  rule. The lifestyle answers carry associations, probably reverse-caused for smoking, that must not
  be read as effects.

### Limitations

1. **Detection, not prediction.** Diagnosis and features come from the same examination. The model
   recognises existing disease. It cannot forecast who will develop CVD.
2. **Unknown provenance and outcome definition.** The dataset's source and the exact meaning of
   "cardiovascular disease" are undocumented. If the label includes diagnosed hypertension, part of
   blood pressure's predictive power is circular.
3. **Treatment is not recorded.** Medication lowers blood pressure and cholesterol in diagnosed
   patients, and diagnosis may change habits. Both can weaken or reverse associations (Section 5.2),
   and neither can be corrected for here.
4. **A 50/50 sample.** Predictive values and absolute risks apply to this sample. Another population
   needs recalibration (Section 9.5).
5. **Records with entry errors are mostly from sick patients.** The main test set leaves them out;
   Section 9.1 reports how the model does on those that can be repaired.
6. **One dataset, one split, one fit per model.** All results come from one source and one
   train/test split, with no external validation. The confidence intervals cover the sampling of test
   patients, not the randomness of training, and the search grids differ in size between models.
7. **Coarse measurements.** Cholesterol and glucose come in three categories, almost all blood
   pressures are rounded to 10 mmHg, and diabetes had to be approximated for the Framingham score.

The next step for this model would be validation on data from a known clinical source, with
recorded treatment and, ideally, outcomes followed over time. Those are what would turn a detector of
existing disease into a tool for prevention.

## References

1. World Health Organization. *Cardiovascular diseases (CVDs)*, fact sheet, updated 31 July 2025.
2. S. Ulianova. *Cardiovascular Disease dataset*. Kaggle, 2019.
3. P. K. Whelton et al. 2017 ACC/AHA guideline for the prevention, detection, evaluation, and
   management of high blood pressure in adults. *Hypertension*, 71(6):e13-e115, 2018.
4. R. B. D'Agostino Sr et al. General cardiovascular risk profile for use in primary care: the
   Framingham Heart Study. *Circulation*, 117(6):743-753, 2008.
5. R. S. Ramesh, R. T. S. Udupa, Monisha J and Kushi K. K. S. Cardiovascular disease prediction using
   machine learning: a comparative analysis. arXiv:2507.21898, 2025.
6. D. Y. Omkari and K. Shaik. An integrated two-layered voting (TLV) framework for coronary artery
   disease prediction using machine learning classifiers. *IEEE Access*, 12:56275-56290, 2024.
   doi:10.1109/ACCESS.2024.3389707. A correction was published (*IEEE Access*, 12:162564, 2024,
   doi:10.1109/ACCESS.2024.3490072); it could not be consulted here.
7. R. B. Z. Auvee et al. Cardiovascular diagnostic detection using multilayer perceptron approach.
   *Systems and Soft Computing*, 8:200490, 2026. doi:10.1016/j.sasc.2026.200490.
8. A. G. Shaper, G. Wannamethee and M. Walker. Alcohol and mortality in British men: explaining the
   U-shaped curve. *The Lancet*, 2(8623):1267-1273, 1988.
9. C. Rudin. Stop explaining black box machine learning models for high stakes decisions and use
   interpretable models instead. *Nature Machine Intelligence*, 1:206-215, 2019.
10. N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer. SMOTE: synthetic minority
    over-sampling technique. *Journal of Artificial Intelligence Research*, 16:321-357, 2002.
11. M. Eltawil et al. Comment on Iacobescu et al. "Evaluating binary classifiers for cardiovascular
    disease prediction: enhancing early diagnostic capabilities". *Journal of Cardiovascular
    Development and Disease*, 13(1):46, 2026. doi:10.3390/jcdd13010046.
12. F. Pedregosa et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning
    Research*, 12:2825-2830, 2011.
