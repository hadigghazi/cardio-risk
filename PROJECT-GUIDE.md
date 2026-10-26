# The complete guide to the project

**Patient's Sickness Prediction: detecting cardiovascular disease from a routine medical examination**

This guide explains the whole project: the idea, every decision and why it was taken, how the work
was done, every code cell of the notebook, the results, the deliverables, and what the final audit
checked and changed. Every number comes from the final executed notebook (VM run 40391671736).

How to read it: Part 1 gives the idea, Part 2 the decisions, Part 3 how it was built. **Part 4 walks
through the notebook cell by cell; read it with `report.ipynb` open beside it.** Part 5 collects the
results, Part 6 describes the deliverables, Parts 7–8 explain the concepts and prepare you for
questions, and Part 10 explains the audit.

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

Building a model that predicts `cardio` is easy; any reasonable classifier gets about 73%. The
interesting questions are **whether that is good**, **what it adds over what doctors already do**,
and **whether the numbers can be trusted**. So the project builds the model the brief asks for, then
evaluates it around three questions:

- **Q1 — What does machine learning add over what a doctor already has?** The comparison is with a
  guideline blood-pressure rule and the Framingham risk score.
- **Q2 — How much do data-handling choices change the measured performance?** This covers cleaning,
  and the common mistakes that inflate accuracy.
- **Q3 — Can the predictions be trusted and explained?** This covers calibration, which factors drive
  the predictions, how different patients are served, and a points score a doctor could use.

### 1.4 Why these questions (what we found before starting)

- **The data has impossible values**, such as blood pressures of −150 or 16,020 mmHg. Cleaning them
  matters.
- **A comparative study of a cleaned version of the dataset reports 73.4%** for its best model
  (Ramesh et al., arXiv 2507.21898, reference [5]).
- **Some papers report much more:** 88% (Omkari and Shaik, IEEE Access 2024, [6]) and 94.5% (Auvee
  et al., Systems and Soft Computing 2026, [7]). A gap that large deserves an explanation. The report
  shows which evaluation mistakes *can* produce such numbers on this data; it does not claim to know
  what those papers did.
- **A single rule is already 71.3% accurate**: "systolic pressure at least 130 mmHg", the threshold
  for stage 1 hypertension. So the gain from ML is likely to be small, and measuring it honestly is
  the contribution.

### 1.5 The answer, in one paragraph

- **The model.** Gradient boosting detects CVD with **73.6% accuracy (95% CI 72.9–74.4%) and ROC AUC
  0.804 (0.797–0.811)** on 13,722 patients it never saw. The guideline rule gets 71.3%; the gain is
  2.3 points (1.8 to 2.9).
- **The gain is real but small.** At an operating point chosen on training data to match the rule's
  specificity, the model finds **274 more of the 6,785 sick test patients (95% CI 220 to 332)**,
  with no clear difference in false alarms. The limit near 74% seems to lie in what the 11 recorded
  values can tell, not in the algorithm or the amount of data; that is a reasoned reading, not a
  proof.
- **Data handling matters more than the algorithm.** Dirty records cost logistic regression 0.017
  AUC (trees barely notice), and two leaky evaluations make models that score 64–71% on unseen
  patients report 85–100%.
- **The risks can be trusted, within limits.** They are well calibrated in this sample and similar
  for women and men (the AUC difference is within the noise). Discrimination is weaker over 60, for
  the model and for blood pressure alone. The lifestyle effects are associations, possibly
  reverse-caused, not causes.
- **A doctor's version.** A 4-factor paper scorecard (age, systolic pressure, BMI, cholesterol)
  reaches AUC 0.793, 0.012 below the best model.

---

## Part 2 — Every decision, and why

| # | Decision | What we chose | Alternatives | Why |
|---|---|---|---|---|
| 1 | Topic | Cardio (topic 2) | water, spam, loan, credit cards | Water has no signal (correlations below 0.035). Spam is trivially 97%+ because of an Enron vocabulary shortcut and 541 duplicates. Loan is tiny (614 rows), and one rule gives 80.9%. Credit cards is clustering with no right answer to check against. Cardio is large, real, messy, and medically meaningful. |
| 2 | Framing | the model the brief asks for, plus an honest evaluation through Q1–Q3 | a pure "beat the accuracy" study, or a pure leakage audit | "Beat the accuracy" hits a limit near 73% and says nothing. An audit alone would dodge the brief. |
| 3 | Structure | conventional report: data → cleaning → EDA → features → models → evaluation → use → conclusion | — | You asked that it read like the project in the brief. |
| 4 | Duplicates | remove 24 exact duplicates first; keep the 12 contradictory groups (training) | keep all | A duplicate pair split across train and test lets the model be tested on a patient it has seen. Contradictory records are honest noise. |
| 5 | Split | one stratified 80/20 split by patient id, seed 42, drawn once from the deduplicated raw data | random splits per experiment | The same test patients for every comparison; stratified keeps 50/50 in both parts. |
| 6 | Cleaning bounds | wide physiological bounds (systolic 70–250, diastolic 40–150, height 120–220, weight 30–200, BMI 12–70, diastolic < systolic) | trimming the top and bottom 2.5% | Percentile trimming removes real, unusual patients. The bounds remove only values no living adult can have. They never look at the diagnosis. |
| 7 | Cleaning strategy | **drop** as the main version; **repair** and **raw** compared | repair only | Drop invents nothing. Repair is a reasoned guess (906 could be 90 or 96), so it is tested against drop rather than trusted. |
| 8 | Repair rules | swap first on the recorded values (only if the swapped pair is plausible); then at most one rule per reading: sign, cmHg ×10, or extra digits | stacking rules | One guess per reading is a recognisable typing error; two stacked guesses are invention. A record can still get one fix per reading (16/10 → 160/100). |
| 9 | Sex code | gender 2 = male | — | Code 2 is 8.5 cm taller on average. |
| 10 | Exploration | on training patients only | on all data | Looking at test diagnoses could steer choices. |
| 11 | Derived features | BMI, pulse pressure, mean arterial pressure, ACC/AHA stage; kept though they add no AUC | raw values only | They cost nothing and make explanations clinical. |
| 12 | Main metric | ROC AUC; accuracy, sensitivity and specificity at stated cut-offs | accuracy only | AUC doesn't depend on a cut-off; accuracy alone hides the trade-off between missed and false cases. |
| 13 | Baselines | systolic ≥ 130 rule; systolic alone with a learned cut-off; Framingham as published; Framingham refitted | a majority-class baseline only | Q1 needs what doctors actually use. The refit separates "wrong formula" from "wrong weights". |
| 14 | Models | logistic regression, random forest, gradient boosting (sklearn's HistGradientBoosting), MLP neural network | XGBoost, LightGBM | They cover linear, bagged trees, boosted trees and a neural net. HistGradientBoosting is the same algorithm family as XGBoost, without extra dependencies. |
| 15 | Tuning | grid search with 5-fold CV on training data; a grid is widened when its best setting sits at an edge | fixed defaults | Fair to every model. A best value at the edge of a grid means a better one might lie beyond it. (The second widening happened after test results had been seen; the report says so. See Part 10.) |
| 16 | Model choice | the simplest model within one standard error of the best CV AUC; the order of simplicity (linear models, then gradient boosting, random forest, the network) fixed before any result | the highest score | Differences inside the noise should not buy complexity. The standard error uses the sample SD over the five folds. |
| 17 | Test set | opened only after the model was chosen; then used to score and examine, never to fit, tune or choose | looking early | Anything chosen on the test set makes its score optimistic. |
| 18 | Uncertainty | 2,000 paired bootstrap resamples of the test patients; Bonferroni 99% intervals for the five model comparisons | a single number | A one-point difference can be noise with 13,722 patients. |
| 19 | Cut-offs | 0.50 (accuracy) and a screening cut-off giving 80% sensitivity, both chosen on out-of-fold training predictions; the comparison with the rule at a cut-off matched to the rule's *training* specificity | choosing on the test set | Choosing on the test set would be leakage. (Before the audit, the rule comparison was read off the test ROC curve; it gave +301. The honest version gives +274.) |
| 20 | Explanations | grouped permutation importance, retraining without each factor, and partial dependence | single-feature importance; SHAP | The five blood-pressure features carry the same information, so scrambling one alone would understate blood pressure. Retraining asks what a model can do without a factor. SHAP would add a dependency without changing the conclusion. |
| 21 | Interpretable model | a points scorecard; total → risk read from the training patients' rates (isotonic, so more points always mean more risk); then a 4-factor clinical card | none | Since ML gains little, a transparent score is the practical output. The clinical card keeps measured factors whose points go the way medicine expects (age, systolic, BMI, cholesterol). |
| 22 | Oversampling (SMOTE) | not used in the real pipeline; only in the leakage demonstration | use it | The classes are already 50/50. |
| 23 | Leakage | three deliberate mistakes reproduced and labelled; no claim about what specific papers did | accuse the papers | We cannot see their code. |
| 24 | Patient check | rejects implausible values and ages outside 30–65 instead of scoring them | score anything | The model knows nothing outside what it was trained on. |
| 25 | Honesty about the process | Section 2 lists every change made after test results had been seen | stay silent | The changes did not touch the chosen model, but a reader deserves to know them. |
| 26 | Where to compute | Docker on dblp-vm with pinned versions, via GitHub Actions | the laptop | Your rule: work on the VM. It is also reproducible. |
| 27 | Submission format | the notebook alone, with all code inside | notebook + .py | You confirmed the brief means either one. |
| 28 | One copy of the code | `cardio.py` is the single source; the notebook build copies its sections in | maintain two copies | Two copies drift apart. The tests and the deck builder use the same code. |
| 29 | Presentation | PowerPoint, built by a script from the Slidesgo template, Morph, about 10 minutes | Google Slides, web deck | Your choice, matching your earlier decks. |
| 30 | Repository visibility | public (made by you, when the private Actions allowance ran out) | private | Public repositories get Actions for free. **But** the brief and a full solution are public too: make it private again once the work is final (Part 9). |

---

## Part 3 — How the work was done

### 3.1 The repository

All work lives in https://github.com/hadigghazi/cardio-risk (currently public):

```
cardio.py            all the code (the single source)
report.md            the report's text + code cells, with markers for where the code sections go
build_report.py      turns report.md into report.ipynb and executes it
report.ipynb         the executed report (the deliverable)
tests/               69 unit tests
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
2. `build_report.py` assembles the notebook and **executes it in a folder without `cardio.py`**
   (`outputs/run`), so any hidden dependency on the file would fail. It also refuses `import cardio`,
   and fails if fewer figures are embedded than the code draws, or if any cell raises an error.
3. The result is `report.ipynb`: text, code and outputs, self-contained.
4. When only the *text* changes after a run, `--reuse-outputs` rebuilds the notebook with that run's
   outputs; it refuses if any code cell differs, or if a `@@NUMBER@@` placeholder is still unfilled.

This is why the submitted notebook shows every line of code, while the repository keeps only one copy
of it.

### 3.3 Where the computation ran

- **The VM.** Everything ran on **dblp-vm**, inside a Docker container capped at 4 GB of memory and
  4 CPUs. The VM has no swap and also serves your dblp and AutoML sites, so a runaway job must die
  alone.
- **The workflow.** `vm.yml` (started by hand) copies the code to the VM over SSH (three repository
  secrets and a dedicated key, `cardio-risk-deploy`), builds the image there, runs the report (about
  13 minutes of computation), and returns the outputs as a downloadable artifact.
- **CI.** `ci.yml` runs the 69 tests on every push, in about a minute.

### 3.4 How it progressed

1. Read the brief and profiled all five datasets, then chose cardio.
2. Checked the literature, found the 73% versus 88–94.5% gap, and adopted Q1–Q3.
3. Built the repository, Docker image, tests and VM workflows, and set up VM access.
4. Cleaning: inspected the impossible values, found four typing-error patterns, wrote drop and repair.
5. Exploration on training patients: four strong factors and the smoking paradox.
6. Features; then models. The first VM run (cross-validation only, before any test result) put two
   models' best settings at grid edges, so the gradient-boosting and network grids were widened.
7. Test evaluation, explanations and the points score; the clinical card.
8. Reliability: cleaning strategies, leakage, calibration, subgroups, predictive value.
9. The patient check (Section 10); then the deck, rendered and fixed slide by slide.
10. Bugs caught along the way: a Docker setting (`MPLBACKEND=Agg`) that stopped figures being
    embedded (removed; the build now counts figures); a too-new scipy that warned on every fit
    (pinned); a command line that crashed on bad input (fixed).
11. Notebook-only submission: the code was moved inside the notebook. Actions ran out, you made the
    repository public, and the report was executed again.
12. **The audit** (Part 10): three independent reviews of the code, the statistics and the claims,
    then a re-check of every sentence against the outputs and the git history. Everything found was
    fixed, the notebook was executed again, and Section 2 now lists the changes made after test
    results had been seen.

---

## Part 4 — The notebook, cell by cell

The notebook has **100 cells: 43 text and 57 code**. Below, every code cell is listed with what it
does, how, and what it shows. "§" numbers are the report's sections.

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
on cross-validation and open the test set only after the choice; one common test set; the metrics and
which results get confidence intervals; leakage experiments kept apart. Then a paragraph listing **the
changes made after test results had been seen** (see Part 10), none of which touched the split, the
checks, the gradient-boosting grid or the order of simplicity.

### §3 The data

**Cell 3 — `load`, `deduplicate`, `split_ids`.**

- `load()` reads the CSV exactly as distributed.
- `deduplicate()` drops records identical in every column except `id`, keeping the first.
- `split_ids()` draws the stratified 80/20 split of patient ids (seed 42) from the deduplicated raw
  data, so every cleaning strategy shares it.

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

**Cell 8 — Cleaning code** (111 lines):

- `bmi(df)`: weight / height².
- `audit(df)`: one True/False column per check (each value out of range; diastolic ≥ systolic; BMI
  out of range).
- `repair_pressure(value, lo, hi)`: exactly one rule may explain a reading:
  - **sign**: −120 → 120 (only if the result is in range);
  - **cmHg**: 12 → 120, only if the result is in range;
  - **extra digits**: drop trailing digits until in range, so 1000 → 100, 14020 → 140, 906 → 90;
  - anything else (zero, a fraction above range, a sign flip that would also need a unit fix) is
    "unrecoverable".
- `repair(df)`: **first** swaps pairs typed the wrong way round (80/120 → 120/80), on the values as
  recorded and only when the swapped pair is plausible; then applies one rule to each remaining
  implausible reading; logs every change (id, field, before, after, rule).
- `clean(df, strategy)`:
  - `raw` = deduplicate only;
  - `drop` = deduplicate, then remove any record failing a check;
  - `repair` = deduplicate, repair, then remove what still fails.
- `cleaning_summary(df)`: records kept, repaired and removed per strategy.

**Cell 9 — The audit.** Prints "24 exact duplicates removed" and "1,395 records (2.0%) fail at least
one check", and, among training patients, the share with CVD: **74.3% of the failing records against
49.5% of the others**. It also shows how many fail each check: systolic out of range 229, diastolic
1,034, diastolic ≥ systolic 1,236, height 53, weight 7, BMI 44.

**Cell 10 — The most frequent impossible values**, for each pressure. For example diastolic 1000
(666 times) and 1100 (156), and systolic 12 (76), 14 (29), 11 (28). It also prints the **8 records
with a negative reading** (−100/80, −115/70, −100/70, −140/90, −120/80 twice, −150/80, 15/−70).

**Cell 11 — Swapped pairs.** "100 records form a plausible reading once their two values are swapped",
and the most common of them: 80/120 (24 records), 90/140 (9), 90/130 (8), 70/110 (7). Most of the
1,236 "diastolic ≥ systolic" records are *not* swaps: they hold an extra-digit diastolic (150/1000) or
a systolic in cmHg (12/80).

**Cell 12 — Figure 1.** Two scatter plots of every patient's blood-pressure pair:

- **left:** as recorded, on a log scale, with the failing records in red, scattered far from the
  real cluster;
- **right:** after repair, with the repaired patients in orange inside the normal cloud.

**Cell 13 — Contradictory records.** "12 groups of training records identical in every feature but
with different diagnoses". They are kept: no model can classify all of them correctly.

**Cell 14 — The three strategies.** A table of records per strategy:

| Strategy | Records | Notes |
|---|---|---|
| raw | 69,976 | duplicates removed only |
| drop | 68,581 | 1,395 removed, 98.0% kept |
| repair | 69,844 | 1,263 repaired, 132 removed, 99.8% kept |

**Cell 15 — The repair log.** Counts by field and rule: systolic cmHg 179, extra digits 38, sign 7,
unrecoverable 5; swapped 100; diastolic extra digits 950, cmHg 15, sign 1, unrecoverable 54; with three
examples of each rule.

**Cell 16 — What cleaning changes** (training patients). The correlation between systolic pressure
and the diagnosis is **0.053 raw, 0.427 dropped, 0.431 repaired**. The maximum systolic is 16,020 raw
and 240 cleaned. Fewer than 2% of the records nearly erase the strongest predictor's correlation.

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
- `train_test(strategy)`: the modelling table divided by the fixed ids.

**Cell 18 — The split.** Prints: training patients 54,859, test patients 13,722 (the drop version);
49.5% with CVD; age 30–65, median 54; 35% men.

**Cell 19 — Figure 2:** six bar charts of the share with CVD by group. The dashed line is the overall
rate.

| Factor | Disease rate from lowest to highest group |
|---|---|
| Age | 23% (<40) → 66% (60+) |
| Systolic pressure | about a quarter (<120) → 86% (160+) |
| BMI | about a quarter → about two thirds |
| Cholesterol | 44% → 76% |
| Glucose | 47% → 62% |

Lifestyle: smokers and drinkers have slightly lower rates; inactive patients slightly higher.

**Cell 20 — Figure 3, the smoking paradox.** Prints "share who smoke": women 1.8%, men 21.9% (so
about 87% of smokers are men). Then, for men only, the gap between smokers and non-smokers within each
age group, averaged with the group sizes as weights, with a bootstrap interval:

- **smokers: −3.1 points (95% CI −4.9 to −1.5)** — the paradox survives;
- **drinkers: −2.0 points (95% CI −4.3 to +0.3)** — inconclusive;
- smokers drink far more (31% vs 5%), which this comparison does not separate.

The text offers a possible explanation the data cannot test: diagnosis and answers come from the same
visit, so diagnosed patients may have quit (the "sick quitter"). Consequences: this is detection, not
prediction; effects are associations, not causes.

**Cell 21 — Figure 4, correlation heatmap.** The strongest pairs: systolic–diastolic 0.73,
weight–BMI 0.86, male–height 0.52, cholesterol–glucose 0.45. With CVD: systolic 0.43, diastolic 0.34,
age 0.24, cholesterol 0.22, BMI 0.19; sex and lifestyle about 0.

### §6 Feature engineering

**Cell 22 — Figure 5.** Disease rate by ACC/AHA stage (normal 22%, elevated 32%, stage 1 38%, stage
2 78%). Then it prints **the most common reading, 120/80 (34% of training patients)**, which is
stage 1 because a diastolic of 80 starts stage 1, and the share of readings that are multiples of 10
(**97%** of systolic and diastolic readings).

### §7 Models

**Cell 23 — Framingham code:**

- `FRAMINGHAM`: the published 2008 office-based coefficients for women and men (age, BMI, systolic,
  smoking, diabetes, baseline survival, mean).
- `diabetes_proxy`: glucose "well above normal", since the data has no diabetes field.
- `framingham_risk`: the published 10-year risk, with nothing fitted.
- `framingham_design`: the same terms per sex, so a logistic regression can refit the formula.

**Cell 24 — Model code** (159 lines):

- `model_specs()`: each model and its grid:

  | Model | Grid searched |
  |---|---|
  | Framingham form | — |
  | Logistic regression | C ∈ {0.01, 0.1, 1, 10} |
  | Random forest (200 trees) | leaf size {10, 30, 100} × features per split {2, √, 0.5} |
  | Gradient boosting | learning rate {0.01, 0.03, 0.1} × leaves {7, 15, 31} × leaf size {20, 50, 200}, early stopping, up to 1,000 trees |
  | MLP | layers {(32), (64, 32), (128, 64), (256, 128)} × weight decay {1e-4, 1e-2, 0.1, 1} |

- `SIMPLICITY`: the order fixed before any result (linear models, gradient boosting, random forest,
  the network).
- `classification_metrics`: AUC, accuracy, sensitivity, specificity.
- `accuracy_threshold`: the most accurate cut-off on training data.
- `compare_models`: scores the 3 baselines and runs a grid search for each model, all on the same
  5 folds; the SD over folds is the sample SD for every row.
- `choose_model`: the one-standard-error rule.
- `feature_set_comparison`, `learning_curve_auc`.

**Cell 25 — Running the comparison.** About 5.5 minutes on the VM. The table:

| Model | CV AUC (SD) | Accuracy |
|---|---|---|
| Gradient boosting | 0.8005 (0.0029) | 73.5% |
| MLP | 0.7996 (0.0027) | 73.4% |
| Random forest | 0.7992 (0.0031) | 73.3% |
| Logistic regression | 0.7902 (0.0029) | 72.7% |
| Framingham refitted | 0.7798 (0.0038) | 71.8% |
| Systolic alone | 0.7506 (0.0035) | 71.3% |
| Rule ≥ 130 | 0.7121 (0.0037) | 71.3% |
| Framingham published | 0.6962 (0.0031) | 65.2% |

Systolic pressure alone, with its most accurate cut-off, classifies exactly like the 130 mmHg rule
(identical accuracy, sensitivity and specificity): almost all readings are multiples of 10, so this is
a consistency check, not a discovery.

**Cell 26 — Figure 6.** The same results as dots with error bars, baselines in grey and models in blue.

**Cell 27 — How flat is each search?** All 27 gradient-boosting settings land between 0.7995 and
0.8005; random forest 0.7937–0.7992 (best: √ features, leaf 30); MLP 0.7969–0.7996 (best: (64, 32),
decay 0.1); logistic regression 0.7902 for every C. The best settings sit inside their grids. The text
also says when the grids were widened (Part 10).

**Cell 28 — Do derived features help?** For gradient boosting the AUC changes by +0.0003; for logistic
regression by −0.0005. No.

**Cell 29 — Figure 7, learning curve.** AUC against the number of training patients, from 438 to
43,887.

- Logistic regression levels off early: +0.003 between 2,000 and 44,000 patients.
- Gradient boosting overtakes it at about 2,000 and keeps improving slowly: +0.0015 for the last
  doubling.
- So more patients of the same kind would add little.

**Cell 30 — The choice.** Prints: best AUC is gradient boosting, 0.8005, fold SD 0.0029; one standard
error is 0.0013; within it: random forest, gradient boosting, MLP; chosen by the order fixed in
advance: **gradient boosting** (learning rate 0.01, 15 leaves, at least 50 patients per leaf). Model
size: 593 trees of at most 15 leaves, against the forest's 114,770 leaves.

### §8 The final model on the test set

**Cell 31 — Evaluation helpers:**

- `bootstrap`: resamples the test patients 2,000 times, the same resamples for all models.
- `interval`: the 2.5–97.5 percentiles (or another level).
- `out_of_fold`: each training patient's risk from a model that didn't see them.
- `screening_threshold`: the highest cut-off still catching at least 80% of the sick training patients.
- `threshold_at_specificity`: the lowest cut-off that clears at least a given share of healthy
  training patients.

**Cell 32 — Every candidate on the test set.** Each model is scored once, with its settings and cut-off
already fixed (also saved to `outputs/test_results.csv`). It prints the systolic cut-off chosen on all
training patients (130 mmHg) and the gain over the rule: **+2.3 points (95% CI +1.8 to +2.9)**.

| Model | Test AUC (95% CI) | CV AUC | Accuracy (95% CI) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Gradient boosting** | **0.804 (0.797–0.811)** | 0.801 | **73.6% (72.9–74.4%)** | 68.9% | 78.2% |
| Random forest | 0.803 (0.796–0.810) | 0.799 | 73.4% (72.6–74.1%) | 68.5% | 78.1% |
| MLP | 0.800 (0.793–0.808) | 0.800 | 73.1% (72.4–73.9%) | 70.6% | 75.6% |
| Logistic regression | 0.793 (0.785–0.800) | 0.790 | 72.6% (71.9–73.3%) | 66.8% | 78.2% |
| Framingham refitted | 0.780 (0.773–0.788) | 0.780 | 71.4% (70.6–72.1%) | 66.3% | 76.3% |
| Systolic alone | 0.754 (0.746–0.761) | 0.751 | 71.3% (70.6–72.0%) | 61.8% | 80.6% |
| Rule ≥ 130 | 0.712 (0.704–0.719) | 0.712 | 71.3% (70.6–72.0%) | 61.8% | 80.6% |
| Framingham published | 0.693 (0.684–0.701) | 0.696 | 64.6% (63.8–65.4%) | 74.0% | 55.5% |

Every test AUC is within 0.004 of its CV AUC: the cross-validation was a faithful guide.

**Cell 33 — Why Framingham fails here.** Mean published risk: women 7.0%, men 15.9%, while CVD is about
equally common (49.2% vs 49.9%). Within women it ranks as well as systolic alone (0.752 vs 0.756);
within men worse (0.695 vs 0.752). Switching the smoking term off raises the AUC from 0.693 to 0.709.
Its weights do not transfer; refitted, the same terms reach 0.780.

**Cell 34 — Figure 8, ROC curves** for gradient boosting, logistic regression, systolic alone and
Framingham, with the rule shown as one point.

**Cell 35 — The gain in patients.** The model's cut-off is chosen **on training patients** (out-of-fold
predictions), as the lowest one clearing as many healthy training patients as the rule (80.7%): 0.531.
On the test patients: rule 61.8% sensitivity / 80.6% specificity; model 65.8% / 81.0%. **Model minus
rule: +274 of the 6,785 diseased (95% CI +220 to +332), and −29 healthy flagged (95% CI −76 to +19,
no clear difference).**

**Cell 36 — How large are the differences?** Gradient boosting minus each other model, with 95% and
Bonferroni-adjusted 99% intervals:

| Comparison | AUC difference | 95% CI | 99% CI (adjusted) |
|---|---|---|---|
| vs random forest | +0.0015 | +0.0002 to +0.0028 | −0.0003 to +0.0032 |
| vs MLP | +0.0040 | +0.0024 to +0.0056 | +0.0020 to +0.0062 |
| vs logistic regression | +0.0119 | +0.0094 to +0.0144 | +0.0086 to +0.0151 |
| vs Framingham refitted | +0.0239 | +0.0205 to +0.0273 | +0.0196 to +0.0286 |
| vs systolic alone | +0.0504 | +0.0457 to +0.0551 | +0.0443 to +0.0569 |

The lead over random forest does not survive the adjustment; none of the first two would change a
clinical decision.

**Cell 37 — Figure 9, two cut-offs.** Confusion matrices:

| Cut-off | Sick missed | Healthy flagged | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|
| 0.50 | 2,112 | 1,510 | 68.9% | 78.2% | 73.6% |
| 0.38 (screening) | 1,347 | 2,476 | 80.1% | 64.3% | 72.1% |

The screening cut-off was chosen on training data and hits its 80% target on unseen patients.

**Cell 38 — Explanation code:**

- `FEATURE_GROUPS`: blood pressure (5 features), age, body size (height, weight, BMI), cholesterol,
  glucose, activity, smoking, alcohol, sex.
- `grouped_permutation_importance`: shuffles each group together, 10 times, and measures the AUC drop.
- `drop_group_importance`: retrains the model without each group and cross-validates it.
- `partial_dependence_curve`: sets one factor to each value for everyone (recomputing BMI and the
  pressures, and capping diastolic at systolic − 10, so no impossible combination appears) and
  averages the predicted risk.

**Cell 39 — Figure 10, what the model relies on.** AUC drop when each group is scrambled:
blood pressure **0.210**, age 0.036, cholesterol 0.030, body size 0.007, then activity, glucose,
smoking, alcohol and sex at 0.002 or less each.

**Cell 40 — Retraining without each factor** (cross-validated): without blood pressure a model still
reaches **0.702**; leaving out age costs 0.020, cholesterol 0.009, body size 0.002, every other factor
0.001 or less. The two views agree on the top four, in the same order.

**Cell 41 — Figure 11, partial dependence** for eight factors (3,000 test patients):

- **Systolic pressure works more like a switch than a slope:** about 31–39% from 100 to 120 mmHg, 54%
  at 130, 80% at 140, then a plateau near 84%. Rounded readings and the stage feature *may* sharpen
  the steps. A straight-line model can only approximate this, which may explain part of its gap.
- **Steady rises:** age 38% at 40 → 65% at 64; weight 41% at 50 kg → 58% at 120 kg; cholesterol
  "well above" adds about 25 points.
- **Flat or slightly lower:** glucose (about −1 point at the top level), smoking and alcohol (about
  −3); activity lowers it slightly too, as medicine expects. Associations, not causes.

**Cell 42 — `PointsScore` code** (the scorecard):

- **Bands:** age <45 / 45–49 / 50–54 / 55–59 / 60+; systolic <120 / 120–129 / 130–139 / 140–159 /
  160+; BMI <25 / 25–30 / 30–35 / 35+; three levels for cholesterol and glucose; yes/no for the
  lifestyle answers.
- **Fitting:** a logistic regression on the bands. Each coefficient is divided by 0.1 and rounded, so
  one point = 0.1 in log-odds.
- **Mapping:** total → the share of *training* patients with CVD at that total, smoothed so that more
  points always mean more risk (isotonic regression; a tiny nudge keeps the ranking identical to the
  totals).
- Refuses values outside its bands; saves as plain numbers (`to_dict` / `from_dict`).
- `CLINICAL_FACTORS` = age, systolic, BMI, cholesterol.

**Cell 43 — The full scorecard.** Systolic earns up to 27 points, age up to 11, cholesterol up to 11,
BMI up to 4. **Smoking −2, alcohol −2 and glucose "well above normal" −3** go against what medicine
expects; activity −2 is the expected direction.

**Cell 44 — The clinical scorecard** (the rule: keep measured factors whose training-fit points go the
way medicine expects):

| Factor | Points by band |
|---|---|
| Age | <45 0, 45–49 4, 50–54 5, 55–59 8, 60+ 11 |
| Systolic | <120 0, 120–129 5, 130–139 13, 140–159 25, 160+ 27 |
| BMI | <25 0, 25–30 2, 30–35 3, 35+ 4 |
| Cholesterol | normal 0, above 3, well above 9 |

Test results: full card AUC 0.795 (0.788–0.803), 73.0%; **clinical card AUC 0.793 (0.785–0.800),
72.4%**; logistic regression 0.793, 72.6%; gradient boosting minus the clinical card **+0.0117 (95% CI
+0.0094 to +0.0142)**. Risk by total: 0 → 8%, 10 → 27%, 20 → 55%, 30 → 81%, 40 → 84%.

**Cell 45 — Figure 12.** Total points against the observed share with CVD among test patients, next
to the risk the card assigns from training patients: they follow each other closely.

### §9 Reliability

**Cell 46 — Reliability code:**

- `strategy_comparison`: trains on raw, drop or repair, tests on the common test patients, with paired
  bootstrap intervals for the change against drop.
- `excluded_test_patients`: how the model does on the test patients that drop removes.
- `smote`: a minimal SMOTE, written out. It interpolates between a patient and one of its 5 nearest
  same-class neighbours.
- `leakage_experiments`: the correct pipeline plus three mistakes, with like-for-like comparisons.
- `subgroup_table`: metrics by sex and age group, with AUC intervals and systolic-alone AUCs.
- `ppv_at_prevalence`: the share of flagged patients who are really sick, at another disease rate.

**Cell 47 — Cleaning strategies:**

| Trained on | Gradient boosting AUC | change vs drop (95% CI) | Logistic regression AUC | change vs drop (95% CI) |
|---|---|---|---|---|
| raw (55,980) | 0.8036 | −0.0008 (−0.0014 to −0.0001) | **0.7756** | **−0.0169 (−0.0193 to −0.0144)** |
| drop (54,859) | 0.8044 | — | 0.7925 | — |
| repair (55,865) | 0.8044 | +0.0000 (−0.0006 to +0.0006) | 0.7925 | −0.0000 |

**Cell 48 — Why logistic regression suffers** (new in the audit). Its blood-pressure weights in
original units (change in log-odds), trained on raw and on clean data:

| Weight | Trained on raw | Trained on drop |
|---|---|---|
| systolic, per 10 mmHg | +0.065 | +0.184 |
| diastolic, per 10 mmHg | +0.004 | +0.161 |
| pulse pressure, per 10 mmHg | +0.023 | +0.270 |
| mean arterial pressure, per 10 mmHg | +0.033 | +0.218 |
| blood-pressure stage, per stage | +0.647 | +0.120 |

On raw data the pressure weights shrink to 3–35% of their clean size, and the stage, which caps any
reading at stage 3, takes over: that is why the raw-data model still reaches 0.776.

**Cell 49 — The excluded test patients.** Drop removes 274 test patients, 76.6% of them sick. Repair
recovers 257 (78.2% sick); on them the model reaches AUC 0.707 and 82.5% accuracy (high mostly because
most are sick). Added to the main test set: AUC 0.806, 73.8%, so Section 8 is, if anything, slightly
conservative.

**Cell 50 — Leakage experiments** (about 2.5 minutes):

| Pipeline | Model | "Reported" accuracy |
|---|---|---|
| Correct | gradient boosting | 73.6% |
| Mistake 1: accuracy on the training patients | fully grown random forest | **99.9%** (71.0% on unseen patients) |
| Mistake 2: SMOTE before the split | 1-nearest neighbour | **85.3%** (64.4% unseen; 64.0% with SMOTE done correctly) |
| Mistake 2: SMOTE before the split | random forest | **86.7%** (71.0% unseen; 70.5% with SMOTE done correctly) |
| Mistake 3: luckiest of 50 random splits | logistic regression | 73.4% (median 72.8%) |

**Cell 51 — Figure 13, calibration.** Predicted against observed risk in 10 groups. Gradient boosting
lies on the diagonal (Brier 0.179, the best); logistic regression bends around it (0.187); the
clinical card is close (0.184).

**Cell 52 — Subgroups:**

| Group | Patients | With CVD | Mean predicted | AUC (95% CI) | Systolic alone AUC | Sens. | Spec. |
|---|---|---|---|---|---|---|---|
| Women | 8,858 | 49.2% | 48.7% | 0.803 (0.794–0.812) | 0.756 | 68.0% | 78.6% |
| Men | 4,864 | 49.9% | 50.7% | 0.808 (0.796–0.819) | 0.752 | 70.4% | 77.6% |
| Under 50 | 4,361 | 36.8% | 36.3% | 0.835 (0.822–0.847) | 0.796 | 59.0% | 91.4% |
| 50–59 | 6,900 | 50.8% | 51.6% | 0.768 (0.756–0.779) | 0.730 | 64.2% | 77.7% |
| 60+ | 2,461 | 67.9% | 66.5% | 0.718 (0.696–0.740) | 0.681 | 88.1% | 34.7% |

Men minus women: +0.005 (95% CI −0.011 to +0.020), within the noise. The age drop appears for systolic
pressure alone too, so it reflects the information more than the model.

**Cell 53 — Positive predictive value at other disease rates:**

| Share with CVD | Accuracy cut-off | Screening cut-off |
|---|---|---|
| 50% (this data) | 76% | 69% |
| 30% | 58% | 49% |
| 20% | 44% | 36% |
| 10% | 26% | 20% |
| 5% | 14% | 11% |

### §10 Using the model

**Cell 54 — Code for checking a patient:**

- `FINAL_PARAMS`: the selected setting, for building the model without repeating the search.
- `build_final`: takes the fitted model, finds the screening cut-off on out-of-fold predictions, and
  fits the clinical scorecard.
- `save_final` and `load_final`: write and read `outputs/model.joblib` (the scorecard as plain
  numbers).
- `patient(...)`: checks a new examination with the same plausibility rules, requires real numbers and
  an age of 30–65, and builds its features. It raises `ValueError` otherwise.
- `assess(bundle, row)`: returns the risk, the verdict, the screening decision and the points
  breakdown.

**Cell 55 — Save the model.** Prints the selected setting (learning rate 0.01, 15 leaves, 50 per leaf),
saves it and prints "screening cut-off: 0.382".

**Cell 56 — Three invented patients:**

| Patient | Risk | Decision | Points (risk) |
|---|---|---|---|
| 62-year-old man, 165/100, BMI 33, cholesterol well above normal | 87% | likely, refer | 50 (85%) |
| 44-year-old woman, 115/75, BMI 22, all normal | 14% | unlikely, no referral | 0 (8%) |
| 55-year-old woman, 130/85, BMI 27, cholesterol above normal | 59% | likely, refer | 26 (68%) |

**Cell 57 — Rejections.** Systolic 1400 prints "rejected: implausible examination values: ap_hi out of
range"; age 80 prints "rejected: age 80 is outside the 30-65 years the model was trained on".

### §11 Conclusion — text only

- **The three answers:**
  - **Q1.** A real but small gain; the limit near 0.80 AUC seems to lie in the data.
  - **Q2.** Data handling can change reported performance far more than the algorithm.
  - **Q3.** Trustworthy, within limits.
- **Seven limitations:**
  1. Detection, not prediction.
  2. Unknown source and diagnosis definition. If hypertension counts as CVD, part of blood pressure's
     power is circular.
  3. No treatment data.
  4. A 50/50 sample.
  5. Records with entry errors are mostly from sick patients.
  6. One dataset, one split, one fit per model; intervals cover only the test sampling; the report was
     revised after its test results had been seen.
  7. Coarse measurements, and a diabetes stand-in.
- **12 references.**

---

## Part 5 — The results in one place

| Question | Answer | Evidence |
|---|---|---|
| How good is the model? | AUC 0.804 (0.797–0.811), accuracy 73.6% (72.9–74.4%) on 13,722 unseen patients | cell 32 |
| Better than the rule? | +2.3 points of accuracy (1.8–2.9); +274 of 6,785 sick found (220–332) at a training-matched specificity | cells 32, 35 |
| Better than Framingham? | Published Framingham does worse than blood pressure alone (0.693); refitted it reaches 0.780, still clearly below the models | cells 32, 33 |
| Which algorithm? | Gradient boosting, but random forest and the MLP are within 0.004 AUC; logistic regression is 0.012 behind | cells 25, 36 |
| Would more data help? | Little: the learning curves flatten | cell 29 |
| What drives it? | Blood pressure (AUC drop 0.210), then age and cholesterol; risk jumps between 120 and 140 mmHg | cells 39–41 |
| Simple alternative? | 4-factor scorecard: AUC 0.793, 72.4% | cell 44 |
| Does cleaning matter? | For logistic regression yes (0.776 → 0.793); for trees barely | cells 16, 47, 48 |
| How do 88–94% claims arise? | Leaky evaluation gives 85–100% from the same data | cell 50 |
| Are risks trustworthy? | Calibrated (Brier 0.179); similar for both sexes; weaker over 60 | cells 51–52 |
| In a real population? | At 20% prevalence, 44% of flagged patients are sick | cell 53 |

---

## Part 6 — The deliverables

### 6.1 The zip

`submission/Hadi-Ghazi_Patient-Sickness-Prediction.zip`:

| File | What it is |
|---|---|
| `report.ipynb` | the report and **all** the code, executed: 100 cells (57 code), 13 figures, no errors |
| `Cardio-Risk.pptx` | the presentation: 15 slides + 3 backup, speaker notes |
| `data/cardio.csv` | the original dataset, unchanged (all cleaning is in code) |
| `requirements.txt` | numpy 2.2.4, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.15.3, joblib 1.4.2, threadpoolctl 3.5.0, matplotlib 3.10.0, nbformat 5.10.4, nbconvert 7.16.6, ipykernel 6.29.5, pytest 8.3.5 |

- **No `.py` file.** The brief accepts either format, and the notebook contains all the code.
- **To re-run it:** open `report.ipynb` with `data/cardio.csv` beside it (or in a parent folder) and run
  all. It takes about 13 minutes on four cores; the model comparison and the leakage experiments take
  most of that.

### 6.2 The presentation, slide by slide

10 minutes. Morph transitions throughout; the bars of slide 7 grow into the leakage bars of slide 11.
The spoken script (about 1,180 words, 8.4 minutes) is in the speaker notes and in
`presentation/script.md`.

| # | Title | What it shows / what you say |
|---|---|---|
| 1 | Detecting Cardiovascular Disease from a Routine Check-up | your name, Lebanese University; the question |
| 2 | Why this matters | 19.8 million deaths in 2022 (WHO); the objective; Q1–Q3 |
| 3 | The data: 70,000 check-ups | 70,000 / 11 / 50-50; measured, lab and self-reported values; source undocumented |
| 4 | 2% of the records are impossible | Figure 1; the typing-error patterns; three in four from sick patients; r = 0.05 → 0.43 |
| 5 | What the clean data shows | Figure 2: blood pressure, age, weight, cholesterol |
| 6 | The smoking paradox | Figure 3; −3.1 points at the same age (CI 1.5 to 4.9); a possible reason, the sick quitter |
| 7 | What machine learning adds | test accuracy and AUC of every candidate on one axis |
| 8 | On patients the model has never seen | 0.804, 73.6%, +274 (with CIs); ROC curves |
| 9 | What drives the prediction | importance bars; retrained without blood pressure 0.70; blood pressure as a switch |
| 10 | A score a doctor can add up | the 4-factor scorecard and its points-to-risk line |
| 11 | How 90% accuracy can happen | leakage bars; the published 88% and 94.5% marked, without claiming what those papers did |
| 12 | Can the predictions be trusted? | calibration; women vs men; older patients; 44% at 20% prevalence |
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
  patients but flags more healthy ones. For a calibrated model, 0.5 maximises expected accuracy.
- **Train/test split, stratified.** Learn on 80%, judge on 20% never seen; stratified keeps the 50/50
  balance in both parts.
- **5-fold cross-validation.** Split the training data into 5 parts and train on 4 / test on 1, five
  times. It is used for choosing without touching the test set.
- **Out-of-fold predictions.** Each training patient's risk from the fold model that did not see
  them; used to choose cut-offs on training data honestly.
- **Grid search.** Try every combination of settings and keep the best cross-validated one.
- **One-standard-error rule.** Prefer the simplest model whose score is within one standard error
  (fold SD / √5) of the best.
- **Bootstrap confidence interval.** Recompute the metric on 2,000 resamples of the test patients and
  keep the middle 95%. Paired: every model on the same resamples, so differences get intervals too.
- **Bonferroni adjustment.** With five comparisons at once, each interval is widened to 99% so that
  the five together keep about a 95% guarantee.
- **Calibration and the Brier score.** Calibrated means predicted 70% comes true about 70% of the time.
  The Brier score is the mean squared error of the risks (lower is better; always 50% gives 0.25).
  Calibration-in-the-large: the mean predicted risk equals the observed share.
- **Isotonic regression.** A fit that may only go up: used to turn points into risks that never fall
  as points rise.
- **Grouped permutation importance.** Scramble a group of features and measure the AUC lost; related
  features are scrambled together.
- **Retraining without a factor.** Build the model again without it and cross-validate: what a model
  can do without that information.
- **Partial dependence.** Set one factor to a value for everyone and average the predictions; it shows
  the shape of an effect.
- **Logistic regression.** A weighted sum of features turned into a probability. Simple and readable,
  but each feature can only push in a straight line (in log-odds).
- **Random forest.** Many deep decision trees on random subsets of patients and features, averaged.
- **Gradient boosting.** Small trees added one at a time, each correcting the previous errors.
- **MLP (neural network).** Layers of weighted sums with non-linear activations.
- **Early stopping.** Boosting stops adding trees when a held-out 10% stops improving.
- **SMOTE.** Synthetic patients interpolated between similar ones, used for imbalanced classes. Done
  before the split, it plants near-copies of training patients in the test set: leakage.
- **Leakage.** Any route by which test information reaches training or model choice. It inflates
  reported results without improving the model.
- **Positive predictive value.** The share of flagged patients who are really sick; it depends on how
  common the disease is.
- **Case mix.** The kinds of patients in a group (how sick, how varied). AUCs of groups with different
  case mixes are not strictly comparable.
- **Sick-quitter effect.** People change habits after diagnosis, so habits can look protective in data
  collected afterwards.
- **Framingham risk score.** A standard 2008 equation for the 10-year risk of a first CVD event, from
  age, sex, BMI, systolic pressure, smoking and diabetes.
- **ACC/AHA 2017 stages.** The US guideline's blood-pressure categories: normal, elevated, stage 1
  (130/80+), stage 2 (140/90+).

---

## Part 8 — Questions you may be asked

**Why only 73.6%?** Careful work on this dataset gets about the same, very different algorithms land
within 0.012 AUC, the searches are flat and the learning curves flatten. So the limit seems to lie in
what the 11 values can tell. It is a reasoned reading, not a proof.

**Then why use ML?** At a cut-off chosen on training data to match the rule's specificity, it finds
274 more of the 6,785 sick test patients (95% CI 220 to 332), with no clear difference in false alarms,
and it gives a calibrated risk instead of a yes/no. The project also produced a 4-factor scorecard
nearly as good.

**Why gradient boosting?** The rule fixed in advance chose it: the simplest model within one standard
error of the best, in an order set before any result. Random forest and the MLP are practically tied,
so either would do.

**Why is the network last in that order?** Its weights cannot be read the way a coefficient or a tree
split can, so it is usually considered the hardest to inspect. The order was set before any result,
so it cannot have been tuned to pick a winner.

**Did you look at the test set before choosing?** No: the model, its setting and the cut-offs were all
chosen on training data. But be ready for this one: **the report was revised after test results had
been seen** (a review of an earlier version), and Section 2 lists what changed: the repair rules, the
random-forest and network grids, the operating point against the rule (now chosen on training data,
which turned +301 into +274), the scorecard's risk mapping, more intervals, and the written rule for
the clinical card, and the sample SD in the one-standard-error rule. None of it touched the split, the
checks, the gradient-boosting grid or the order of simplicity; the chosen model and its results are
the same.

**Why does the clinical card leave out glucose?** Its training-fit points go the wrong way (−3 at
"well above normal"), probably through its overlap with cholesterol. The report also admits that an
earlier version had chosen the same four factors partly from test-set importances; the training data
alone point the same way, and the clinical card scores slightly *below* the full card, so this did not
flatter it.

**Why not SMOTE?** The classes are already balanced. Done correctly it made the models slightly worse
(64.0% vs 64.4%, 70.5% vs 71.0%); done before the split it inflated accuracy to 85–87%.

**Did cleaning the test set leak anything?** No. The rules are fixed per record, learn nothing and never
look at the diagnosis, and all strategies are scored on the same test patients. A real system would
reject impossible inputs, which is what `patient()` does. The report also shows how the model does on
the removed test patients once repaired.

**Why does smoking lower the risk?** It doesn't, biologically. A possible reason the data cannot test:
diagnosed patients may have quit (the sick quitter). The model describes associations, and the
clinical scorecard leaves smoking out.

**Is the model fair?** Women and men get similar AUCs (difference +0.005, CI −0.011 to +0.020) and
calibration. It is weaker over 60, as is blood pressure alone; that is reported as a limitation.

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
| `build_report.py` | builds and executes the notebook without `cardio.py`; checks figures and errors; `--reuse-outputs` |
| `tests/` | 69 unit tests: cleaning and repair rules, features, Framingham, metrics and cut-offs, scorecard, patient check, notebook build |
| `Dockerfile`, `requirements.txt` | the reproducible environment |
| `.github/workflows/ci.yml` | tests on every push |
| `.github/workflows/vm.yml` | runs the report on dblp-vm on demand (Actions → VM → Run workflow → `run`, `report`) |
| `presentation/` | `build.py` (rebuilds the deck from the saved model), the template, figures, `script.md`, the deck |
| `brief/` | the two course documents |
| `outputs/`, `submission/` | local VM outputs and the zip (not committed) |

**Visibility.** The repository is public, so the brief and a complete solution can be found by anyone,
including other students. Make it private again when the work is final; GitHub Actions minutes are
only needed for another VM run.

---

## Part 10 — The audit: what was checked and what changed

You asked for an audit of every phase: right decisions, nothing overclaimed, no inflated accuracy.
Three independent reviews (code and pipeline, statistics, claims) and a final re-check of every
sentence against the outputs and the git history found the following, and all of it was fixed.

**What was verified and held.** No leakage in the real pipeline: one split drawn once, cleaning that
learns nothing and never sees the diagnosis, every model choice and cut-off on training data, the test
set opened after the choice. The headline (AUC 0.804, accuracy 73.6%) did not change and reproduces
exactly across runs.

**Fixed in the method:**

- **Repair rules.** Swaps are now tested first on the recorded values (some, like 90/160, were being
  lost), and a reading gets at most one guess.
- **The operating point against the rule.** It was read off the test ROC curve (+301); it is now chosen
  on out-of-fold training predictions (+274, CI 220–332).
- **The scorecard.** Points map to risk through the training patients' observed rates (isotonic); the
  card refuses values outside its bands and saves as plain numbers.
- **Statistics.** Sample SD everywhere for the one-standard-error rule; intervals for every candidate's
  accuracy, the gain over the rule, the strategy differences and the subgroups; Bonferroni-adjusted
  intervals for the five model comparisons; an exact screening cut-off.
- **Like-for-like leakage comparisons**: each leaky number now sits next to the same model scored
  honestly, with and without correct SMOTE.
- **New checks:** retraining without each factor; how the model does on the removed test patients; the
  systolic-alone AUC in each subgroup; the logistic regression's weights on raw data.
- **Grids.** The random-forest and network grids were widened because their best settings sat at the
  edges. This happened after test results had been seen; neither model was chosen.
- **The patient check** rejects ages outside 30–65 and non-numbers.

**Fixed in the claims:** "equally well" became "similar" with the interval; "ceiling set by the data"
became "seems to lie in the data"; the smoking paradox is adjusted for age with an interval and its
explanation is "possible", not "likely"; alcohol's gap is "inconclusive"; glucose and the Framingham
refit are described by their numbers; the protocol now says exactly what the test set was used for;
the references were corrected (WHO 19.8 million in 2022; Auvee et al.; Eltawil et al.; the IEEE Access
correction notice).

**Disclosed rather than hidden:** the report's Section 2 lists every change made after test results
had been seen, Section 7.2 says when the grids were widened, and Section 8.5 says the clinical card's
four factors were first picked partly from test-set importances.

**What remains (and is stated as limitations):** detection, not prediction; undocumented source and
diagnosis definition; no treatment data; a 50/50 sample; entry errors concentrated in sick patients;
one dataset and one split; coarse measurements.
