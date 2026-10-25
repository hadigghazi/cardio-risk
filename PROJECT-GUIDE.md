# The complete guide to the project

**Patient's Sickness Prediction: detecting cardiovascular disease from a routine medical examination**

This guide explains the whole project: the idea, every decision and why it was taken, how the work
was done, every code cell of the notebook, the results, and the deliverables. Every number comes from
the final executed notebook (VM run 40122319735).

How to read it: Part 1 gives the idea, Part 2 the decisions, Part 3 how it was built. **Part 4 walks
through the notebook cell by cell; read it with `report.ipynb` open beside it.** Part 5 collects the
results, Part 6 describes the deliverables, and Parts 7–8 explain the concepts and prepare you for
questions.

---

## Part 1 — The idea

### 1.1 What the course asked

The brief: one student picks one of five topics, and does "practical research" by applying AI to a
public dataset. It hands in a zip with the work (as a notebook, or as a Python file) and a final
presentation. Topic 2, **Patient's Sickness Prediction**, says: "Using the cardio dataset, create a
model to check the presence or absence of cardiovascular disease of a patient."

### 1.2 The dataset

`cardio.csv` has **70,000 patients**. Each row is one medical examination with 11 values and a label:

| Column | Meaning | Kind |
|---|---|---|
| `age` | age in **days** | measured |
| `gender` | 1 or 2 (the documentation doesn't say which is which) | recorded |
| `height`, `weight` | cm, kg | measured |
| `ap_hi`, `ap_lo` | systolic and diastolic blood pressure, mmHg | measured |
| `cholesterol`, `gluc` | 1 normal, 2 above normal, 3 well above normal | lab result |
| `smoke`, `alco`, `active` | smoking, alcohol, physical activity, 0/1 | **self-reported** |
| `cardio` | has cardiovascular disease (CVD), 0/1 | **the target** |

### 1.3 The core idea in three sentences

Building a model that predicts `cardio` is easy; any classifier gets about 73%. The interesting
questions are **whether that is good**, **what it adds over what doctors already do**, and **whether
the numbers can be trusted**. So the project builds the model the brief asks for, then evaluates it
around three questions:

- **Q1 — What does machine learning add over what a doctor already has?** The comparison is with a
  guideline blood-pressure rule and the Framingham risk score.
- **Q2 — How much do data-handling choices change the measured performance?** This covers cleaning,
  and the common mistakes that inflate accuracy.
- **Q3 — Can the predictions be trusted and explained?** This covers calibration, which factors drive
  the predictions, how different patients are served, and a points score a doctor could use.

### 1.4 Why these questions (what we found before starting)

- **The data has impossible values**, such as blood pressures of −150 or 16,020 mmHg. Cleaning them
  matters.
- **Careful studies get about 73%** on this dataset (CatBoost 0.734, arXiv 2507.21898).
- **Some papers claim much more:** 88% (Omkari and Shaik, IEEE Access 2024) and 94.5% (Zafar et al.,
  Systems and Soft Computing 2026). That gap suggests mistakes in how those models were evaluated.
- **A single rule is already 71.3% accurate**: "systolic pressure at least 130 mmHg", the threshold
  for stage 1 hypertension. So the real gain from ML is small, and measuring it honestly is the
  contribution.

### 1.5 The answer, in one paragraph

- **The model.** Gradient boosting detects CVD with **73.6% accuracy and ROC AUC 0.804** on 13,722
  patients it never saw. The guideline rule gets 71.3%.
- **The gain is real but small.** At the same false-alarm rate, the model finds 301 more of the 6,785
  sick patients. The ceiling of about 74% comes from the 11 recorded values themselves, not from the
  algorithm.
- **Evaluation mistakes explain the inflated claims.** Cleaning matters for linear models, and leaky
  evaluation can produce 85–100% accuracy from the same data.
- **The risks can be trusted, within limits.** They are well calibrated and equal for both sexes, but
  weaker for patients over 60, and lifestyle effects are reverse-caused associations.
- **A doctor's version.** A 4-factor paper scorecard comes within 0.011 AUC of the best model.

---

## Part 2 — Every decision, and why

| # | Decision | What we chose | Alternatives | Why |
|---|---|---|---|---|
| 1 | Topic | Cardio (topic 2) | water, spam, loan, credit cards | Water has no signal (correlations below 0.035). Spam is trivially 97%+ because of an Enron vocabulary shortcut and 541 duplicates. Loan is tiny (614 rows), and one rule gives 80.9%. Credit cards is clustering with no right answer to check against. Cardio is large, real, messy, and medically meaningful. |
| 2 | Framing | the model the brief asks for, plus an honest evaluation through Q1–Q3 | a pure "beat the accuracy" study, or a pure leakage audit | "Beat the accuracy" hits a ceiling at ~73% and says nothing. An audit alone would dodge the brief. |
| 3 | Structure | conventional report: data → cleaning → EDA → features → models → evaluation → use → conclusion | — | You asked that it read like the project in the brief. |
| 4 | Duplicates | remove 24 exact duplicates first; keep 17 contradictory groups | keep all | A duplicate pair split across train and test lets the model be tested on a patient it has seen. Contradictory records are honest noise. |
| 5 | Split | one stratified 80/20 split by patient id, seed 42, drawn once | random splits per experiment | The same test patients for every comparison; stratified keeps 50/50 in both parts. |
| 6 | Cleaning bounds | wide physiological bounds (systolic 70–250, diastolic 40–150, height 120–220, weight 30–200, BMI 12–70, diastolic < systolic) | trimming the top and bottom 2.5% | Percentile trimming removes real, unusual patients. The bounds remove only values no living adult can have. |
| 7 | Cleaning strategy | **drop** as the main version; **repair** and **raw** compared | repair only | Drop invents nothing. Repair is a reasoned guess (906 could be 90 or 96), so it is tested against drop rather than trusted. |
| 8 | Sex code | gender 2 = male | — | Code 2 is 8.5 cm taller on average. |
| 9 | Exploration | on training patients only | on all data | Looking at test patients could steer choices. |
| 10 | Derived features | BMI, pulse pressure, mean arterial pressure, ACC/AHA stage; kept though they add no AUC | raw values only | They cost nothing and make explanations clinical. |
| 11 | Main metric | ROC AUC; accuracy, sensitivity and specificity at stated cut-offs | accuracy only | AUC doesn't depend on a cut-off; accuracy alone hides the trade-off between missed and false cases. |
| 12 | Baselines | systolic ≥ 130 rule; systolic alone with a learned cut-off; Framingham as published; Framingham refitted | a majority-class baseline only | Q1 needs what doctors actually use. The refit separates "wrong formula" from "wrong weights". |
| 13 | Models | logistic regression, random forest, gradient boosting (sklearn's HistGradientBoosting), MLP neural network | XGBoost, LightGBM | They cover linear, bagged trees, boosted trees and a neural net. HistGradientBoosting is the same algorithm family as XGBoost, without extra dependencies. |
| 14 | Tuning | grid search with 5-fold CV on training data; grids widened when the best setting sat at an edge | fixed defaults | Fair to every model. A best value at the edge of a grid means a better one might lie beyond it. |
| 15 | Model choice | the simplest model within one standard error of the best CV AUC, fixed in advance | the highest score | Differences inside the noise should not buy complexity. |
| 16 | Test set | scored once, with 2,000 bootstrap resamples and paired differences | a single number | A one-point difference can be noise with 13,722 patients. |
| 17 | Cut-offs | 0.50 (accuracy) and a screening cut-off giving 80% sensitivity, chosen on out-of-fold training predictions | choosing it on the test set | Choosing on the test set would be leakage. |
| 18 | Explanations | grouped permutation importance plus partial dependence | single-feature importance; SHAP | The five blood-pressure features carry the same information, so scrambling one alone would understate blood pressure. SHAP would add a dependency without changing the conclusion. |
| 19 | Interpretable model | a points scorecard, then a 4-factor clinical version | none | Since ML gains little, a transparent score is the practical output. Lifestyle factors earned negative points, so they were dropped. |
| 20 | Oversampling (SMOTE) | not used in the real pipeline; only in the leakage demonstration | use it | The classes are already 50/50. |
| 21 | Leakage | three deliberate mistakes reproduced and labelled; no claim about what specific papers did | accuse the papers | We cannot see their code. |
| 22 | Where to compute | Docker on dblp-vm with pinned versions, via GitHub Actions | the laptop | Your rule: work on the VM. It is also reproducible. |
| 23 | Submission format | the notebook alone, with all code inside | notebook + .py | You confirmed the brief means either one. |
| 24 | One copy of the code | `cardio.py` is the single source; the notebook build copies its sections in | maintain two copies | Two copies drift apart. The tests and the deck builder use the same code. |
| 25 | Presentation | PowerPoint, built by a script from the Slidesgo template, Morph, about 10 minutes | Google Slides, web deck | Your choice, matching your earlier decks. |
| 26 | Repository visibility | public (made by you) | private | The private Actions allowance ran out, and public repositories get Actions for free. |

---

## Part 3 — How the work was done

### 3.1 The repository

All work lives in https://github.com/hadigghazi/cardio-risk (now public):

```
cardio.py            all the code (the single source)
report.md            the report's text + code cells, with markers for where the code sections go
build_report.py      turns report.md into report.ipynb and executes it
report.ipynb         the executed report (the deliverable)
tests/               46 unit tests
Dockerfile           Python 3.12 + pinned libraries
requirements.txt     the pinned versions
data/cardio.csv      the dataset
presentation/        build.py, template, figures, the deck, script.md
.github/workflows/   ci.yml (tests on every push), vm.yml (runs jobs on the VM)
```

### 3.2 How the notebook is produced

1. `report.md` is plain Markdown. Text becomes Markdown cells and every ```python block becomes a code
   cell. A line like `<!-- code: data quality -->` is replaced by the matching section of `cardio.py`,
   which is split at its banner comments.
2. `build_report.py` assembles the notebook and **executes it in a folder without `cardio.py`**. So
   any hidden dependency on the file would fail. It also refuses `import cardio`, and fails if fewer
   figures are embedded than the code draws.
3. The result is `report.ipynb`: text, code and outputs, self-contained.

This is why the submitted notebook shows every line of code, while the repository keeps only one copy
of it.

### 3.3 Where the computation ran

- **The VM.** Everything ran on **dblp-vm**, inside a Docker container capped at 4 GB of memory and
  4 CPUs. The VM has no swap and also serves your dblp and AutoML sites, so a runaway job must die
  alone.
- **The workflow.** `vm.yml` copies the code to the VM over SSH (using three repository secrets and a
  dedicated key, `cardio-risk-deploy`), builds the image there, runs the report (about 9 minutes),
  and returns the outputs as a downloadable artifact.
- **CI.** `ci.yml` runs the 46 tests on every push, in about a minute.

### 3.4 How it progressed

1. Read the brief and profiled all five datasets, then chose cardio.
2. Checked the literature, found the 73% versus 88–94.5% gap, and adopted Q1–Q3.
3. Built the repository, Docker image, tests and VM workflows, and set up VM access.
4. Cleaning: inspected the impossible values, found four typing-error patterns, wrote drop and repair.
5. Exploration on training patients: four strong factors and the smoking paradox.
6. Features; then models. The first VM run put two models' best settings at grid edges, so the grids
   were widened; nothing changed beyond the 4th decimal.
7. Test evaluation, explanations and the points score; the clinical card was added after lifestyle
   factors earned negative points.
8. Reliability: cleaning strategies, leakage, calibration, subgroups, predictive value.
9. The patient check (Section 10); then the deck, rendered and fixed slide by slide.
10. **Bugs caught before submission:**
    - **No figures in the notebook.** A Docker setting (`MPLBACKEND=Agg`) stopped plots from being
      embedded. Removed, and the build now checks for it.
    - **Scipy warnings.** A too-new scipy warned on every fit; it is now pinned.
    - **A rough tool.** The command line printed raw column names and crashed on bad input; both fixed.
11. Notebook-only submission: the code was moved inside the notebook. Actions ran out, you made the
    repository public, and the final run was executed.

---

## Part 4 — The notebook, cell by cell

The notebook has **91 cells: 39 text and 52 code**. Below, every code cell is listed with what it does,
how, and what it shows. "§" numbers are the report's sections.

### §1 Introduction — the problem, the objective, the three questions

**Cell 1 — Setup (imports and constants).**

- **Imports:** `math`, `os`, `Path`, `numpy`, `pandas`.
- **Finds the project folder** by looking for `data/cardio.csv` in the current folder and its
  parents, and sets `ROOT`, `DATA`, `OUTPUTS`.
- **Fixes the constants used everywhere:**
  - `SEED = 42` and `TEST_SIZE = 0.2`: the split.
  - `TARGET = "cardio"`.
  - `BOUNDS` and `BMI_BOUNDS`: the plausibility limits.
  - `STRATEGIES = ("raw", "drop", "repair")`.
  - `BASE_FEATURES`: the 11 recorded values, with age in years and sex as `male`.
  - `DERIVED_FEATURES`: BMI, pulse pressure, mean arterial pressure, BP stage.
  - `FEATURES`: all 15 together.
- No output.

**Cell 2 — Plot style.** Imports matplotlib, creates `outputs/figures`, and sets one look for every
chart: thin marks, light grid, blue for no-CVD and orange for CVD. Every figure is also saved as a PNG
there. No output.

### §2 Evaluation protocol — text only

The seven rules fixed before any model: deduplicate first; one split; cleaning learns nothing; choose
on cross-validation; one common test set; the metrics; leakage experiments kept apart.

### §3 The data

**Cell 3 — `load`, `deduplicate`.**

- `load()` reads the CSV exactly as distributed.
- `deduplicate()` drops records identical in every column except `id`, keeping the first.

**Cell 4 — First look.** `raw = load()` prints "70,000 records, 13 columns" and shows the first five
rows.

**Cell 5 — `raw.describe()`.** Summary statistics for every column. The impossible values show up
here: `ap_hi` runs from −150 to 16,020, height from 55 cm, weight from 10 kg.

**Cell 6 — Balance and coding.** Prints:

- the target balance: 50/50;
- missing values: 0;
- mean height by gender code: 161.4 cm for code 1, 169.9 cm for code 2, so 2 = male.

### §4 Data quality

**Cell 7 — Bounds table.** Shows the plausibility bounds as a table.

**Cell 8 — Cleaning code** (104 lines):

- `bmi(df)`: weight / height².
- `audit(df)`: one True/False column per check (each value out of range; diastolic ≥ systolic; BMI
  out of range).
- `repair_pressure(value, lo, hi)`: the repair rules for one reading, in order:
  - **sign**: −120 → 120;
  - **cmHg**: 12 → 120, only if the result is in range;
  - **extra digits**: divide by 10 until in range, so 1000 → 100, 14020 → 140, 906 → 90;
  - anything else becomes "unrecoverable".
- `repair(df)`: applies that to both pressures, swaps pairs typed the wrong way round (80/120 →
  120/80), and logs every change (id, field, before, after, rule).
- `clean(df, strategy)`:
  - `raw` = deduplicate only;
  - `drop` = deduplicate, then remove any record failing a check;
  - `repair` = deduplicate, repair, then remove what still fails.
- `cleaning_summary(df)`: records kept, repaired and removed per strategy.

**Cell 9 — The audit.** Prints "24 exact duplicates removed" and "1,395 records (2.0%) fail at least
one check". It also shows how many fail each check: systolic out of range 229, diastolic 1,034,
diastolic ≥ systolic 1,236, height 53, weight 7, BMI 44.

**Cell 10 — The most frequent impossible values**, for each pressure. For example diastolic 1000
(666 times) and 1100 (156), and systolic 12 (76), 14 (29), 11 (28).

**Cell 11 — Swapped pairs.** The most common "diastolic ≥ systolic" combinations, for example
150/1000 and 12/80.

**Cell 12 — Figure 1.** Two scatter plots of every patient's blood-pressure pair:

- **left:** as recorded, on a log scale, with the failing records in red, scattered far from the
  real cluster;
- **right:** after repair, with the 1,257 repaired patients in orange inside the normal cloud.

**Cell 13 — Contradictory records.** "17 groups of records identical in every feature but with
different diagnoses". They are kept: no model can classify all of them correctly.

**Cell 14 — The three strategies.** A table of records per strategy:

| Strategy | Records | Notes |
|---|---|---|
| raw | 69,976 | duplicates removed only |
| drop | 68,581 | 98.0% kept |
| repair | 69,838 | 1,257 repaired, 138 removed |

**Cell 15 — The repair log.** Counts by field and rule (e.g. diastolic "extra digits" 950, systolic
"cmHg" 179, "swapped" 94, unrecoverable 73), with three examples of each rule.

**Cell 16 — What cleaning changes.** The correlation between systolic pressure and the diagnosis is
**0.054 raw, 0.428 dropped, 0.432 repaired**. The maximum systolic is 16,020 raw and 240 cleaned.
Under 2% of the records hide the strongest predictor.

### §5 Exploratory data analysis (training patients only)

**Cell 17 — Feature and split code:**

- `prepare`: age days → years; `male` = gender 2.
- `bp_stage`: the ACC/AHA 2017 category, the higher of the systolic and diastolic categories:
  - normal: below 120/80;
  - elevated: 120–129 systolic and below 80 diastolic;
  - stage 1: 130–139 or 80–89;
  - stage 2: 140+ or 90+.
- `add_features`: BMI, pulse pressure (systolic − diastolic), mean arterial pressure
  ((systolic + 2 × diastolic) / 3), and the stage.
- `dataset(strategy)`: load → clean → prepare → add features.
- `split_ids`: the stratified 80/20 split of patient ids, seed 42, drawn from the deduplicated raw
  data so every strategy shares it.
- `train_test(strategy)`: the modelling table divided by those ids.

**Cell 18 — The split.** Prints: training patients 54,859, test patients 13,722 (the drop version);
49.5% with CVD; age 30–65, median 54; 35% men.

**Cell 19 — Figure 2:** six bar charts of the share with CVD by group. The dashed line is the overall
rate.

| Factor | Disease rate from lowest to highest group |
|---|---|
| Age | 23% (<40) → 66% (60+) |
| Systolic pressure | 23% (<120) → 86% (160+) |
| BMI | 26% → 68% |
| Cholesterol | 44% → 76% |
| Glucose | 47% → 62% |

Lifestyle: smokers 47% vs 50%; drinkers 48% vs 50%; inactive 53% vs active 49%.

**Cell 20 — Figure 3, the smoking paradox.** Prints "share who smoke": women 1.8%, men 21.9%. Then,
for men only, CVD rate by age group for smokers and non-smokers. Smokers are lower in every group
(e.g. 44% vs 49% at 50–54).

- **Explanation:** diagnosis and answers come from the same visit, so diagnosed patients have often
  quit: the "sick quitter".
- **Consequences:** this is detection, not prediction; effects are associations, not causes.

**Cell 21 — Figure 4, correlation heatmap.** The strongest pairs: systolic–diastolic 0.73,
weight–BMI 0.86, male–height 0.52. With CVD: systolic 0.43, diastolic 0.34, age 0.24, cholesterol
0.22, BMI 0.19; sex and lifestyle about 0.

### §6 Feature engineering

**Cell 22 — Figure 5.** Disease rate by ACC/AHA stage:

| Stage | Disease rate |
|---|---|
| normal | 22% |
| elevated | 32% |
| stage 1 | 38% |
| stage 2 | 78% |

The stage 1 group is large because 120/80, the most common reading, counts as stage 1 (a diastolic of
80 starts stage 1).

### §7 Models

**Cell 23 — Framingham code:**

- `FRAMINGHAM`: the published 2008 office-based coefficients for women and men (age, BMI, systolic,
  smoking, diabetes, baseline survival, mean).
- `diabetes_proxy`: glucose "well above normal", since the data has no diabetes field.
- `framingham_risk`: the published 10-year risk, with nothing fitted. A hand check: a man of 50, BMI
  25, 120 mmHg comes out at 8.48%.
- `framingham_design`: the same terms per sex, so a logistic regression can refit the formula.

**Cell 24 — Model code** (153 lines):

- `model_specs()`: each model and its grid:

  | Model | Grid searched |
  |---|---|
  | Framingham form | — |
  | Logistic regression | C ∈ {0.01, 0.1, 1, 10} |
  | Random forest | leaf size {10, 30, 100} × features per split {√, 0.5} |
  | Gradient boosting | learning rate {0.01, 0.03, 0.1} × leaves {7, 15, 31} × leaf size {20, 50, 200}, early stopping |
  | MLP | layers {(32), (64, 32), (128, 64)} × weight decay {1e-5, 1e-4, 1e-2} |

- `classification_metrics`: AUC, accuracy, sensitivity, specificity.
- `accuracy_threshold`: the most accurate cut-off on training data.
- `compare_models`: scores the 3 baselines and runs a grid search for each model, all on the same
  5 folds.
- `choose_model`: the one-standard-error rule.
- `feature_set_comparison`: models with and without derived features.
- `learning_curve_auc`: AUC as the number of training patients grows.

**Cell 25 — Running the comparison.** It takes about 4–5 minutes on the VM. The table:

| Model | CV AUC | Accuracy |
|---|---|---|
| Gradient boosting | 0.8005 | 73.5% |
| Random forest | 0.7992 | 73.3% |
| MLP | 0.7986 | 73.3% |
| Logistic regression | 0.7902 | 72.7% |
| Framingham refitted | 0.7798 | 71.8% |
| Systolic alone | 0.7506 | 71.3% |
| Rule ≥ 130 | 0.7121 | 71.3% |
| Framingham published | 0.6962 | 65.2% |

The cut-off the data chooses for systolic pressure alone is exactly 130 mmHg in every fold, the
guideline's own value.

**Cell 26 — Figure 6.** The same results as dots with error bars, baselines in grey and models in blue.

**Cell 27 — Do derived features help?** For gradient boosting the AUC changes by +0.0003; for logistic
regression by −0.0005. No.

**Cell 28 — Figure 7, learning curve.** AUC against the number of training patients, from 438 to
43,887.

- Logistic regression is flat from about 2,000 patients.
- Gradient boosting overtakes it there, and gains only 0.0015 from 22k to 44k patients.
- So more data of this kind won't help.

**Cell 29 — The choice.** Prints: best AUC is gradient boosting, 0.8005 ± 0.0026; one standard error
is 0.0012; chosen: **gradient boosting** (learning rate 0.01, 15 leaves, at least 50 patients per
leaf).

### §8 The final model on the test set

**Cell 30 — Evaluation helpers:**

- `bootstrap`: resamples the test patients 2,000 times, the same resamples for all models.
- `interval`: the 2.5–97.5 percentiles.
- `out_of_fold`: each training patient's risk from a model that didn't see them.
- `screening_threshold`: the highest cut-off still catching 80% of the sick.

**Cell 31 — Every candidate on the test set.** Each model is scored once (the table is also saved to
`outputs/test_results.csv`):

| Model | Test AUC (95% CI) | CV AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Gradient boosting** | **0.804 (0.797–0.811)** | 0.801 | **73.6%** | 68.9% | 78.2% |
| Random forest | 0.803 (0.796–0.810) | 0.799 | 73.4% | 68.5% | 78.1% |
| MLP | 0.803 (0.795–0.810) | 0.799 | 73.4% | 71.0% | 75.7% |
| Logistic regression | 0.793 (0.785–0.800) | 0.790 | 72.6% | 66.8% | 78.2% |
| Framingham refitted | 0.780 (0.773–0.788) | 0.780 | 71.4% | 66.3% | 76.3% |
| Systolic alone | 0.754 (0.746–0.761) | 0.751 | 71.3% | 61.8% | 80.6% |
| Rule ≥ 130 | 0.712 (0.704–0.719) | 0.712 | 71.3% | 61.8% | 80.6% |
| Framingham published | 0.693 (0.684–0.701) | 0.696 | 64.6% | 74.0% | 55.5% |

Every test AUC is within 0.004 of its CV AUC, so nothing overfitted.

**Cell 32 — Figure 8, ROC curves** for gradient boosting, logistic regression, systolic alone and
Framingham, with the rule shown as one point.

**Cell 33 — The gain in patients.** At the rule's specificity (80.6%), the rule detects 61.8% of the
sick and the model 66.2%: **301 more of the 6,785 diseased test patients**, with no extra false alarms.

**Cell 34 — Are the differences real?** Gradient boosting minus each other model, with 95% intervals:

| Comparison | AUC difference | 95% interval |
|---|---|---|
| vs random forest | +0.0015 | +0.0002 to +0.0028 |
| vs MLP | +0.0018 | +0.0001 to +0.0036 |
| vs logistic regression | +0.0119 | +0.0094 to +0.0144 |
| vs Framingham refitted | +0.0239 | |
| vs systolic alone | +0.0504 | |

All exclude zero, but the first two are clinically meaningless.

**Cell 35 — Figure 9, two cut-offs.** Confusion matrices:

| Cut-off | Sick missed | Healthy flagged | Sensitivity | Specificity |
|---|---|---|---|---|
| 0.50 | 2,112 | 1,510 | 68.9% | 78.2% |
| 0.38 (screening) | 1,347 | 2,476 | 80.1% | 64.3% |

The screening cut-off was chosen on training data and hits its 80% target on unseen patients.

**Cell 36 — Explanation code:**

- `FEATURE_GROUPS`: blood pressure (5 features), age, body size (height, weight, BMI), cholesterol,
  glucose, activity, smoking, alcohol, sex.
- `grouped_permutation_importance`: shuffles each group together, 10 times, and measures the AUC drop.
- `partial_dependence_curve`: sets one factor to each value for everyone (recomputing BMI and the
  pressures so no impossible combination appears) and averages the predicted risk.

**Cell 37 — Figure 10, what the model relies on.** AUC drop when each group is scrambled:

| Group | AUC drop |
|---|---|
| Blood pressure | 0.210 |
| Age | 0.036 |
| Cholesterol | 0.030 |
| Body size | 0.007 |
| Activity | 0.002 |
| Glucose | 0.002 |
| Smoking | 0.001 |
| Alcohol | 0.001 |
| Sex | 0.000 |

**Cell 38 — Figure 11, partial dependence** for eight factors (3,000 test patients):

- **Systolic pressure works like a switch:** about 31–39% below 125 mmHg, a steep jump to about 84% by
  150 mmHg, then flat. This is why trees beat a straight-line model.
- **Steady rises:** age 38% → 65%; weight rises gently; cholesterol "well above" adds about 25 points.
- **Flat or slightly protective:** glucose, smoking and alcohol (reverse causation).

**Cell 39 — `PointsScore` code** (the scorecard):

- **Bands:** age <45 / 45–49 / 50–54 / 55–59 / 60+; systolic <120 / 120–129 / 130–139 / 140–159 /
  160+; BMI <25 / 25–30 / 30–35 / 35+; three levels for cholesterol and glucose; yes/no for the
  lifestyle answers.
- **Fitting:** a logistic regression on the bands. Each coefficient is divided by 0.1 and rounded, so
  one point = 0.1 in log-odds.
- **Mapping:** a one-variable logistic regression turns the total into a risk.
- `CLINICAL_FACTORS` = age, systolic, BMI, cholesterol.

**Cell 40 — The full scorecard.** Systolic earns up to 27 points, age up to 11, cholesterol up to 11,
BMI up to 4. **Smoking −2, alcohol −2, active −2 and high glucose −3**: negative points, from reverse
causation.

**Cell 41 — Figure 12.** The full scorecard reaches test AUC 0.795 (0.788–0.803) and 72.5% accuracy,
0.009 below gradient boosting. The figure plots total points against the observed share with CVD,
which follows the score's curve.

**Cell 42 — The clinical scorecard** (4 factors):

| Factor | Points by band |
|---|---|
| Age | <45 0, 45–49 4, 50–54 5, 55–59 8, 60+ 11 |
| Systolic | <120 0, 120–129 5, 130–139 13, 140–159 25, 160+ 27 |
| BMI | <25 0, 25–30 2, 30–35 3, 35+ 4 |
| Cholesterol | normal 0, above 3, well above 9 |

**AUC 0.793, accuracy 72.3%**: the same as logistic regression on all 15 features, and 0.011 below
the best model. In risk terms: 0 points ≈ 13%, 10 ≈ 28%, 20 ≈ 52%, 30 ≈ 74%, 40 ≈ 89%.

### §9 Reliability

**Cell 43 — Reliability code:**

- `strategy_comparison`: trains on raw, drop or repair, and tests on the common test patients.
- `smote`: a minimal SMOTE, written out in 15 lines. It interpolates between a patient and one of its
  5 nearest same-class neighbours.
- `leakage_experiments`: the correct pipeline plus three mistakes.
- `subgroup_table`: metrics by sex and age group.
- `ppv_at_prevalence`: the share of flagged patients who are really sick, at another disease rate.

**Cell 44 — Cleaning strategies:**

| Trained on | Gradient boosting AUC | Logistic regression AUC |
|---|---|---|
| raw | 0.804 | **0.776** |
| drop | 0.804 | 0.793 |
| repair | 0.805 | 0.793 |

Trees ignore absurd values, since they only split on order. Logistic regression is pulled by them.

**Cell 45 — Leakage experiments** (about 2.5 minutes):

| Pipeline | Model | "Reported" accuracy |
|---|---|---|
| Correct | gradient boosting | 73.6% |
| Mistake 1: accuracy on the training patients | fully grown random forest | **99.9%** (71.0% on unseen patients) |
| Mistake 2: SMOTE before the split | 1-nearest neighbour | **85.3%** |
| Mistake 2: SMOTE before the split | random forest | **86.7%** |
| SMOTE done correctly | same models | 64.0% / 70.5% |
| Mistake 3: luckiest of 50 random splits | logistic regression | 73.4% (median 72.8%) |

**Cell 46 — Figure 13, calibration.** Predicted against observed risk in 10 groups. Gradient boosting
lies on the diagonal (Brier 0.179, the best); logistic regression bends around it (0.187); the points
score is close (0.185).

**Cell 47 — Subgroups:**

| Group | AUC | Mean predicted | Observed |
|---|---|---|---|
| Women | 0.803 | 48.7% | 49.2% |
| Men | 0.808 | 50.7% | 49.9% |
| Under 50 | 0.835 | | |
| 50–59 | 0.768 | | |
| 60+ | 0.718 | | |

At 60+, the 0.5 cut-off catches 88% of the sick but clears only 35% of the healthy.

**Cell 48 — Positive predictive value at other disease rates:**

| Share with CVD | Accuracy cut-off | Screening cut-off |
|---|---|---|
| 50% (this data) | 76% | 69% |
| 30% | 58% | 49% |
| 20% | 44% | 36% |
| 10% | 26% | 20% |
| 5% | 14% | 11% |

### §10 Using the model

**Cell 49 — Code for checking a patient:**

- `FINAL_PARAMS`: the chosen setting.
- `build_final`: fits the model, finds the screening cut-off, and fits the clinical scorecard.
- `save_final` and `load_final`: write and read `outputs/model.joblib`.
- `patient(...)`: checks a new examination with the same plausibility rules and builds its features.
  It raises `ValueError` for impossible values.
- `assess(bundle, row)`: returns the risk, the verdict, the screening decision and the points
  breakdown.

**Cell 50 — Save the model.** It first checks that the model's settings equal `FINAL_PARAMS`, then
saves it and prints "screening cut-off: 0.382".

**Cell 51 — Three invented patients:**

| Patient | Risk | Decision | Points (risk) |
|---|---|---|---|
| 62-year-old man, 165/100, BMI 33, cholesterol well above normal | 87% | likely, refer | 50 (95%) |
| 44-year-old woman, 115/75, BMI 22, all normal | 14% | unlikely, no referral | 0 (13%) |
| 55-year-old woman, 130/85, BMI 27, cholesterol above normal | 59% | likely, refer | 26 (66%) |

**Cell 52 — An impossible reading.** Systolic 1400 prints "rejected: implausible examination values:
ap_hi out of range".

### §11 Conclusion — text only

- **The three answers:**
  - **Q1.** A real but small gain; the ceiling is set by the data.
  - **Q2.** Data handling matters more than the algorithm.
  - **Q3.** Trustworthy, within limits.
- **Six limitations:**
  1. Detection, not prediction.
  2. Unknown source and diagnosis definition. If hypertension counts as CVD, part of blood pressure's
     power is circular.
  3. No treatment data.
  4. A 50/50 sample.
  5. One dataset, no external validation.
  6. Coarse lab values, and a diabetes stand-in.
- **12 references.**

---

## Part 5 — The results in one place

| Question | Answer | Evidence |
|---|---|---|
| How good is the model? | AUC 0.804 (0.797–0.811), accuracy 73.6% on 13,722 unseen patients | cell 31 |
| Better than the rule? | +2.3 points of accuracy; +301 of 6,785 sick found at the rule's specificity | cells 31, 33 |
| Better than Framingham? | Published Framingham does worse than blood pressure alone (0.693); refitted it reaches 0.780 | cells 25, 31 |
| Which algorithm? | Gradient boosting, but random forest and the MLP are practically tied; logistic regression is 0.012 behind | cells 25, 34 |
| Would more data help? | No: the learning curve is flat | cell 28 |
| What drives it? | Blood pressure (AUC drop 0.210), then age and cholesterol; risk jumps between 125 and 150 mmHg | cells 37–38 |
| Simple alternative? | 4-factor scorecard: AUC 0.793, 72.3% | cell 42 |
| Does cleaning matter? | For logistic regression yes (0.776 → 0.793); for trees no | cells 16, 44 |
| How do 88–94% claims arise? | Leaky evaluation gives 85–100% from the same data | cell 45 |
| Are risks trustworthy? | Calibrated (Brier 0.179); equal for both sexes; weaker over 60 | cells 46–47 |
| In a real population? | At 20% prevalence, 44% of flagged patients are sick | cell 48 |

---

## Part 6 — The deliverables

### 6.1 The zip

`submission/Hadi-Ghazi_Patient-Sickness-Prediction.zip` (5 MB):

| File | What it is |
|---|---|
| `report.ipynb` | the report and **all** the code, executed: 91 cells (52 code), 13 figures, no errors |
| `Cardio-Risk.pptx` | the presentation: 15 slides + 3 backup, speaker notes |
| `data/cardio.csv` | the original dataset, unchanged (all cleaning is in code) |
| `requirements.txt` | numpy 2.2.4, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.15.3, joblib 1.4.2, threadpoolctl 3.5.0, matplotlib 3.10.0, nbformat 5.10.4, nbconvert 7.16.6, ipykernel 6.29.5, pytest 8.3.5 |

- **No `.py` file.** The brief accepts either format, and the notebook contains all the code.
- **To re-run it:** open `report.ipynb` with `data/cardio.csv` beside it (or in a parent folder) and run
  all. It takes about 9–10 minutes; the model comparison and the leakage experiments take most of
  that.

### 6.2 The presentation, slide by slide

10 minutes. Morph transitions throughout; the bars of slide 7 grow into the leakage bars of slide 11.
The spoken script (1,151 words, about 8 minutes) is in the speaker notes and in
`presentation/script.md`.

| # | Title | What it shows / what you say |
|---|---|---|
| 1 | Detecting Cardiovascular Disease from a Routine Check-up | your name, Lebanese University; the question |
| 2 | Why this matters | 17.9 million deaths a year; the objective; Q1–Q3 |
| 3 | The data: 70,000 check-ups | 70,000 / 11 / 50-50; measured, lab and self-reported values; source undocumented |
| 4 | 2% of the records are impossible | Figure 1; the four typing-error patterns; r = 0.05 → 0.43 |
| 5 | What the clean data shows | Figure 2: blood pressure, age, weight, cholesterol |
| 6 | The smoking paradox | Figure 3; the sick quitter; detection not prediction; associations not causes |
| 7 | What machine learning adds | test accuracy and AUC of every candidate on one axis |
| 8 | On patients the model has never seen | 0.804, 73.6%, +301; ROC curves |
| 9 | What drives the prediction | importance bars; blood pressure as a switch |
| 10 | A score a doctor can add up | the 4-factor scorecard and its points-to-risk table |
| 11 | How 90% accuracy happens | leakage bars; the published 88% and 94.5% marked |
| 12 | Can the predictions be trusted? | calibration; women vs men; over 60; 44% at 20% prevalence |
| 13 | Checking a new patient | the notebook call and its output; a rejected reading |
| 14 | Conclusions | the three answers; the limitations |
| 15 | Thank you | questions |
| 16–18 | Backup | the full test table; cleaning strategies; subgroups |

---

## Part 7 — Concepts you need to explain

- **ROC AUC.** The chance that the model gives a random sick patient a higher risk than a random
  healthy one. 0.5 is a coin toss and 1.0 is perfect. It doesn't depend on a cut-off.
- **Accuracy, sensitivity, specificity.**
  - Accuracy: the share classified correctly.
  - Sensitivity: the share of sick patients caught.
  - Specificity: the share of healthy patients cleared.
- **Cut-off (threshold).** The risk above which a patient is called sick. Lowering it catches more sick
  patients but flags more healthy ones.
- **Train/test split, stratified.** Learn on 80%, judge on 20% never seen; stratified keeps the 50/50
  balance in both parts.
- **5-fold cross-validation.** Split the training data into 5 parts and train on 4 / test on 1, five
  times. It is used for choosing without touching the test set.
- **Grid search.** Try every combination of settings and keep the best cross-validated one.
- **One-standard-error rule.** Prefer the simplest model whose score is within the noise of the best.
- **Bootstrap confidence interval.** Recompute the metric on 2,000 resamples of the test patients and
  keep the middle 95%. Paired: every model on the same resamples, so differences get intervals too.
- **Calibration and the Brier score.** Calibrated means predicted 70% comes true about 70% of the time.
  The Brier score is the mean squared error of the risks (lower is better; always 50% gives 0.25).
- **Grouped permutation importance.** Scramble a group of features and measure the AUC lost; related
  features are scrambled together.
- **Partial dependence.** Set one factor to a value for everyone and average the predictions; it shows
  the shape of an effect.
- **Logistic regression.** A weighted sum of features turned into a probability. Simple and readable,
  but each feature can only push in a straight line.
- **Random forest.** Many decision trees on random subsets of patients and features, averaged.
- **Gradient boosting.** Small trees added one at a time, each correcting the previous errors.
- **MLP (neural network).** Layers of weighted sums with non-linear activations.
- **Early stopping.** Boosting stops adding trees when a held-out 10% stops improving.
- **SMOTE.** Synthetic patients interpolated between similar ones, used for imbalanced classes. Done
  before the split, it plants near-copies of training patients in the test set: leakage.
- **Leakage.** Any route by which test information reaches training or model choice. It inflates
  reported results without improving the model.
- **Positive predictive value.** The share of flagged patients who are really sick; it depends on how
  common the disease is.
- **Sick-quitter effect.** People change habits after diagnosis, so habits look protective in data
  collected afterwards.
- **Framingham risk score.** A standard 2008 equation for the 10-year risk of a first CVD event, from
  age, sex, BMI, systolic pressure, smoking and diabetes.
- **ACC/AHA 2017 stages.** The US guideline's blood-pressure categories: normal, elevated, stage 1
  (130/80+), stage 2 (140/90+).

---

## Part 8 — Questions you may be asked

**Why only 73.6%?** That is the ceiling for these 11 values. Careful studies get the same, five
different algorithms land within 0.01 AUC, and the learning curve is flat.

**Then why use ML?** It finds 301 more sick patients at the rule's false-alarm rate, and gives a
calibrated risk instead of a yes/no. The project also produced a 4-factor scorecard nearly as good.

**Why gradient boosting?** The rule fixed in advance chose it. Random forest and the MLP are
practically tied, so either would do.

**Why not SMOTE?** The classes are already balanced. Done correctly it made the models worse (64–71%);
done before the split it inflated accuracy to 85–87%.

**Did cleaning the test set leak anything?** No. The rules are fixed per record and learn nothing, and
all strategies are scored on the same test patients. A real system would reject impossible inputs,
which is what `patient()` does.

**Why does smoking lower the risk?** It doesn't, biologically. Diagnosed patients usually quit (the
sick quitter). The model describes associations, and the clinical scorecard leaves smoking out.

**Is the model fair?** Women and men get the same AUC and calibration. It is weaker over 60; that is
reported as a limitation.

**Can it predict future heart disease?** No. Features and diagnosis come from the same visit, so it
detects existing disease. Prediction needs patients followed over time.

**Why is there no .py file?** The brief accepts either a notebook or a Python file. The notebook
contains all the code, so it runs alone with just the data file.

**Where did it run?** On dblp-vm, in a Docker container with pinned library versions, started through
GitHub Actions. The notebook in the zip is the output of that run.

---

## Part 9 — The repository beyond the zip

| Path | Purpose |
|---|---|
| `cardio.py` | the single source of all code (copied into the notebook by the build); also a command line: `python cardio.py clean / compare / train / predict` |
| `report.md` | the report's source |
| `build_report.py` | builds and executes the notebook without `cardio.py`; checks the figures |
| `tests/` | 46 unit tests: cleaning rules, features, Framingham, metrics, scorecard, patient check, notebook build |
| `Dockerfile`, `requirements.txt` | the reproducible environment |
| `.github/workflows/ci.yml` | tests on every push |
| `.github/workflows/vm.yml` | runs the report on dblp-vm on demand (Actions → VM → Run workflow → `run`, `report`) |
| `presentation/` | `build.py` (rebuilds the deck from the saved model), the template, figures, `script.md`, the deck |
| `brief/` | the two course documents |
| `outputs/`, `submission/` | local VM outputs and the zip (not committed) |
