# Project guide: Patient's Sickness Prediction (cardio)

A complete account of the project: what is in the submission, what every file does, what was done
and in what order, why each decision was taken, and the concepts you need to explain it. Every number
here comes from the final executed report (VM run 40122319735).

---

## 1. The project in one paragraph

The course asked for one of five topics, done alone, as "practical research" that applies AI to a
public dataset. The deliverable is a zip with the report as a notebook (`.ipynb`), a Python file
(`.py`) and a presentation. We chose **topic 2, Patient's Sickness Prediction**. It uses `cardio.csv`,
70,000 patients with 11 values recorded at a medical check-up and a yes/no label for cardiovascular
disease (CVD).

- **What we built.** We cleaned the data, explored it, and compared clinical baselines with five
  machine-learning models. The final model is **gradient boosting: 73.6% accuracy and ROC AUC 0.804
  on 13,722 patients it never saw.** A doctor's simple rule (systolic pressure ≥ 130 mmHg) gets
  71.3%.
- **The tool.** The last section of the notebook packages the model as a check of a new patient.
- **The honest framing.** The report also shows what the model adds, how data handling changes the
  results (including how some published papers could claim 88–94.5%), and how far its predictions can
  be trusted.

---

## 2. The submission zip, file by file

`submission/Hadi-Ghazi_Patient-Sickness-Prediction.zip`:

| File | Required by the brief? | What it is |
|---|---|---|
| `report.ipynb` | yes: the report "as a .ipynb file" | the full report with **all** the code inside it, already executed: text, code, tables, 13 figures |
| `Cardio-Risk.pptx` | yes: "a final presentation" | 15 slides + 3 backup, speaker notes on every slide |
| `data/cardio.csv` | no, added | the dataset, so the notebook can be re-run |
| `requirements.txt` | no, added | exact library versions, so the results reproduce |

**Why there is no `.py` file.** The brief lists ".ipynb" and ".py" because a project can be
delivered either way, and we chose the notebook. Everything is in it: every function, model and
check appears as visible code in the section where it is first used. The notebook needs only
`data/cardio.csv` next to it, or in a parent folder.

### 2.1 `report.ipynb`: the report

Read it top to bottom; it is written as a paper with code inside it. It has 91 cells, 52 of them
code, all executed with outputs saved, and 13 embedded figures. It was executed on the VM in a Docker
container in 536 seconds.

**§1 Introduction.**

- **Why CVD matters.** It causes 17.9 million deaths a year (WHO).
- **The objective.** Build a model that detects CVD from one examination, and evaluate it honestly.
- **Why the data is tricky.** It contains impossible values. Careful studies get about 73%, while
  some papers claim 88% and 94.5%.
- **Three guiding questions:**
  - **Q1.** What does machine learning add over what doctors already have (a blood-pressure rule and
    the Framingham risk score)?
  - **Q2.** How much do data-handling choices (cleaning, leakage) change the measured performance?
  - **Q3.** Can the predictions be trusted and explained?

**§2 Evaluation protocol.** The rules were fixed before any model was trained:

1. Remove the 24 exact duplicate records first, so the same patient cannot be in both train and test.
2. Make one stratified 80/20 train/test split by patient id (seed 42). Everything later uses these ids.
3. Cleaning uses fixed per-record rules and learns nothing from the data, so it cannot leak.
4. Choose models with 5-fold cross-validation on training data only, and score the test set once.
5. When comparing cleaning strategies, score all of them on the same test patients.
6. Metrics: ROC AUC is primary; accuracy, sensitivity and specificity are reported at stated cut-offs;
   Brier score and calibration measure the risks. Every test number gets a 95% bootstrap interval.
7. The leakage experiments are deliberately wrong pipelines, kept apart and never used for model choice.

**§3 The data.** 70,000 records with 13 columns: id, age (in days), gender (coded 1/2), height,
weight, systolic `ap_hi` and diastolic `ap_lo` pressure, cholesterol and glucose (1/2/3), smoke,
alco, active (0/1), and the target `cardio`. The classes are 50/50 and no cells are empty. Gender
code 2 is taken as male because it is 8.5 cm taller on average. Age is converted to years (30–65).

**§4 Data quality.**

- **4.1 Plausibility audit.** Wide physiological bounds: systolic 70–250, diastolic 40–150, height
  120–220 cm, weight 30–200 kg, BMI 12–70, and diastolic must be below systolic. 1,395 records (2.0%)
  fail at least one check, almost all on blood pressure. The errors follow patterns:
  - an **extra zero**: 1000 should be 100 (over 800 records)
  - **cmHg typed instead of mmHg**: 12 should be 120
  - a **stray minus sign**: −120 should be 120
  - the two readings **swapped**: 80/120 should be 120/80

  Figure 1 shows the raw scatter (log scale) next to the repaired data. 17 groups of records are
  identical in every feature but have different diagnoses; they are kept, as honest noise.
- **4.2 Three cleaning strategies:**
  - `raw`: duplicates removed only; 69,976 records.
  - `drop`: every implausible record removed; 68,581 records (98.0%). Used for the main analysis.
  - `repair`: fixes blood pressures with a recognisable error (sign, then unit, then extra digits,
    then swap), then removes whatever is still implausible. 69,838 records, 1,257 of them repaired,
    each repair logged.
- **4.3 What cleaning changes.** The correlation of systolic pressure with the diagnosis is **0.054
  raw and 0.428 after cleaning**. Fewer than 2% of the records hide the strongest predictor from any
  linear analysis.

**§5 Exploratory analysis**, on the 54,859 training patients only, so the test patients influence
nothing:

- **5.1 Disease rate by factor** (Figure 2). It rises steadily with systolic pressure (23% below 120,
  86% at 160+), age (23% → 66%), BMI (26% → 68%) and cholesterol (44% → 76%). Glucose is weaker. The
  lifestyle answers barely matter, and smokers and drinkers have slightly *less* disease.
- **5.2 The smoking paradox** (Figure 3).
  - Almost only men smoke here (22% of men, 1.8% of women).
  - Among men, smokers have less CVD in every age group.
  - The likely explanation is the "sick quitter": diagnosis and answers come from the same visit, so
    patients already diagnosed have usually quit.
  - Two consequences: this is **detection of existing disease, not prediction of future disease**,
    and feature effects are **associations, not causes**.
- **5.3 Correlations** (Figure 4). The two pressures correlate at 0.73, weight and BMI at 0.86. Trees
  don't care; logistic regression coefficients must be read with care.
- **5.4 Implications.** The effects are monotonic, so a linear model should get most of the signal.
  There is no class imbalance, so SMOTE (oversampling) has nothing to fix.

**§6 Feature engineering.** Four standard clinical quantities are added:

- **BMI**: weight / height².
- **Pulse pressure**: systolic − diastolic.
- **Mean arterial pressure**: (systolic + 2 × diastolic) / 3.
- **ACC/AHA 2017 blood-pressure stage**: 0 normal, 1 elevated, 2 stage 1, 3 stage 2. Figure 5 shows
  the disease rate rising across the stages.

They are fixed formulas, so they cannot leak. §7 shows they don't improve the AUC, but they make
explanations readable.

**§7 Models** (5-fold CV on training patients):

- **The candidates:**
  - Three baselines that fit nothing: the rule systolic ≥ 130; systolic pressure alone with a cut-off
    chosen on training folds; the Framingham 2008 office-based score as published.
  - Five models: Framingham's formula refitted on our data, logistic regression, random forest,
    gradient boosting (HistGradientBoosting), and a neural network (MLP). Each model's
    hyperparameters are searched by grid.
- **Results** (Figure 6). Gradient boosting has CV AUC 0.8005 and 73.5% accuracy. Random forest
  (0.7992) and the MLP (0.7986) are essentially tied with it; logistic regression is 0.7902. Below
  the models: Framingham refitted 0.780, systolic pressure alone 0.751, the rule 0.712, Framingham
  published 0.696.
- **Notable findings:**
  - The data-chosen cut-off for systolic pressure is exactly **130 mmHg** in all five folds, the
    guideline's own value.
  - **Published Framingham ranks worse than blood pressure alone.** It predicts a first event over 10
    years, a different question in a different population. Refitted, it reaches 0.780: the formula is
    fine, but the weights don't transfer.
- **7.3 Derived features.** They change the AUC by under 0.001.
- **7.4 Learning curve** (Figure 7). Logistic regression stops improving at about 2,000 patients.
  Gradient boosting overtakes it, but the last doubling (22k → 44k patients) adds only 0.0015. **The
  ~0.80 AUC ceiling comes from the information in the 11 recorded values, not from too little data.**
  Widening the search grids changed nothing beyond the 4th decimal.
- **7.5 Choice.** The one-standard-error rule, fixed in advance, picks **gradient boosting**
  (learning rate 0.01, 15 leaves per tree, at least 50 patients per leaf).

**§8 The final model on the test set** (13,722 patients, used once):

- **8.1 Test results.** Gradient boosting reaches **AUC 0.804 (95% CI 0.797–0.811), 73.6% accuracy,
  68.9% sensitivity, 78.2% specificity**. Every model's test AUC is within 0.004 of its CV AUC, so
  nothing overfitted. ROC curves are in Figure 8.
- **At the rule's own specificity** (80.6%), the model detects 66.2% of diseased patients instead of
  61.8%: **301 more of 6,785**.
- **8.2 Paired bootstrap differences.**
  - The lead over random forest and the MLP is under 0.002: statistically detectable, clinically
    meaningless.
  - Over logistic regression: +0.012. Over systolic pressure alone: +0.050.
- **8.3 Thresholds** (Figure 9).
  - At 0.50: 2,112 sick patients missed, 1,510 healthy patients flagged.
  - Screening cut-off (0.38), chosen on training predictions to catch 80% of the sick: test
    sensitivity 80.1%, specificity 64.3%; 1,347 missed, 2,476 flagged.
- **8.4 What drives it** (Figures 10–11).
  - **Grouped permutation importance**: scrambling the 5 blood-pressure features costs 0.210 AUC; age
    0.036, cholesterol 0.030, body size 0.007; everything else ≈ 0.
  - **Partial dependence**: predicted risk sits around 31–39% below 125 mmHg, jumps to about 84% by
    150 mmHg, then stays flat. **It acts like a switch, not a slope**, which explains why trees beat
    logistic regression. Glucose, smoking and alcohol come out flat or slightly protective (reverse
    causation again).
- **8.5 A points score** (Figure 12).
  - **Full scorecard**: each factor's band earns whole points. AUC 0.795, accuracy 72.5%.
  - **The problem**: smoking, alcohol, activity and high glucose earn *negative* points.
  - **The fix**: a **clinical version** with only age, systolic pressure, BMI and cholesterol reaches
    **AUC 0.793 and 72.3% accuracy**. That equals full logistic regression and is only 0.011 below
    gradient boosting.

**§9 Reliability:**

- **9.1 Cleaning strategy.**
  - Gradient boosting is unaffected: 0.804 trained on raw, dropped or repaired data. Trees split on
    the order of values.
  - Logistic regression falls from 0.793 to **0.776 on raw data**.
  - Repair and drop give the same results.
- **9.2 Leakage experiments**, the same data with three mistakes:

  | Pipeline | Reported accuracy |
  |---|---|
  | Correct (gradient boosting, unseen patients) | **73.6%** |
  | Scored on the training patients (fully grown random forest) | **99.9%** (71.0% on unseen patients) |
  | SMOTE before the split, then 1-nearest-neighbour | **85.3%** |
  | SMOTE before the split, then random forest | **86.7%** |
  | SMOTE done correctly (training data only) | 64.0–70.5% |
  | Luckiest of 50 random splits | 73.4% (median split: 72.8%) |

  Conclusion: accuracies far above 74% on this dataset should be read as signs of leakage. We do not
  claim to know what the 88% or 94.5% papers actually did.
- **9.3 Calibration** (Figure 13). Gradient boosting's risks sit on the diagonal (Brier 0.179, the
  lowest). Logistic regression bends around the diagonal.
- **9.4 Subgroups.**
  - Women and men get similar AUCs: 0.803 and 0.808.
  - Discrimination falls with age: 0.835 under 50, 0.768 at 50–59, 0.718 at 60+.
  - Risks stay calibrated in every group. At 60+, the 0.5 cut-off flags 88% of the sick but clears
    only 35% of the healthy.
- **9.5 Outside this dataset.** Here 76% of flagged patients really have CVD. In a population where
  20% have CVD it would be 44% (36% at the screening cut-off), so a positive result means "examine
  further", not a diagnosis.

**§10 Using the model.** The notebook saves the final model with its two cut-offs and the clinical
scorecard to `outputs/model.joblib`, and assesses three invented patients:

| Patient | Predicted risk |
|---|---|
| 62-year-old man, 165/100 mmHg, BMI 33, high cholesterol | 87% |
| 44-year-old woman, normal results | 14% |
| 55-year-old woman, 130/85 mmHg | 59% |

It also shows that an impossible reading (1400 mmHg) is rejected.

**§11 Conclusion and limitations.**

- **Answers to the three questions:**
  - **Q1.** A real but small gain; the ceiling is set by the data.
  - **Q2.** Handling matters more than the algorithm.
  - **Q3.** The predictions are trustworthy, within limits.
- **Limitations:**
  1. It detects existing disease; it does not forecast.
  2. The source and the diagnosis definition are undocumented. If the label includes hypertension,
     part of blood pressure's power is circular.
  3. Treatment is not recorded.
  4. The 50/50 sample makes predictive values and risks sample-specific.
  5. One dataset, no external validation.
  6. Coarse lab values; diabetes had to be approximated from glucose.

It ends with 12 references.

### 2.2 The code inside the notebook

The notebook contains all the code, as 52 code cells of about 1,200 lines. The functions are
defined in the section where they are first used, and the cells after them call them by name, for
example `clean(raw, "repair")` or `compare_models(train)`:

| Code cell (where in the notebook) | Main functions | What they do |
|---|---|---|
| Setup (§1) | `BOUNDS`, `FEATURES`, `SEED=42`, `TEST_SIZE=0.2` | the plausibility bounds, the 15 model features, the fixed split |
| Loading (§3) | `load`, `deduplicate` | read the CSV; drop exact duplicates (ignoring `id`) |
| Data quality (§4.1) | `audit`, `repair_pressure`, `repair`, `clean`, `cleaning_summary` | flag implausible values; repair one blood pressure (sign → cmHg → extra digits), swap transposed pairs, log every change; build the raw/drop/repair datasets |
| Features (§5) | `prepare`, `bp_stage`, `add_features`, `dataset`, `split_ids`, `train_test` | age in years, sex as `male`; BMI, pulse pressure, mean arterial pressure, ACC/AHA stage; the fixed patient-id split shared by every strategy |
| Clinical baselines (§7.2) | `framingham_risk`, `framingham_design`, `diabetes_proxy` | the published Framingham 2008 equations (coefficients checked against two sources); the same terms per sex for refitting; glucose "well above normal" as the diabetes stand-in |
| Models (§7.2) | `model_specs`, `compare_models`, `choose_model`, `feature_set_comparison`, `learning_curve_auc` | the five models and their grids; cross-validation of baselines and models on the same folds; the one-standard-error choice; the derived-feature check; the learning curve |
| Evaluation (§8.1) | `classification_metrics`, `accuracy_threshold`, `bootstrap`, `interval`, `out_of_fold`, `screening_threshold` | AUC, accuracy, sensitivity, specificity; cut-offs chosen on training data; paired bootstrap confidence intervals |
| Points score (§8.5) | `PointsScore`, `CLINICAL_FACTORS` | bands the factors, fits a logistic regression, rounds coefficients to whole points (1 point = 0.1 log-odds), maps the total back to a risk; the 4-factor clinical version |
| Explanation (§8.4) | `grouped_permutation_importance`, `partial_dependence_curve` | the AUC drop when a group of related features is scrambled together; the average risk as one factor varies, with derived features recomputed |
| Reliability (§9.1) | `strategy_comparison`, `smote`, `leakage_experiments`, `subgroup_table`, `ppv_at_prevalence` | the three cleaning strategies on one test set; a minimal SMOTE written out; the three deliberate mistakes; metrics by sex and age; predictive value at other prevalences |
| Using the model (§10) | `final_estimator`, `build_final`, `save_final`, `load_final`, `patient`, `assess` | fit and save the model with both cut-offs and the clinical card; validate a new examination with the cleaning rules; return the risk, verdict, screening decision and points breakdown |

Checking a new patient takes two lines (Section 10):

```python
bundle = load_final()     # the saved model, both cut-offs and the clinical scorecard
assess(bundle, patient(age=62, sex="male", height=172, weight=98, systolic=165, diastolic=100, cholesterol=3))
```

This returns a risk of 87%, the verdict "cardiovascular disease likely", "refer for examination" at
the 0.38 screening cut-off, and 50 clinical points (+11 age 60+, +27 systolic 160+, +3 BMI 30–34.9,
+9 cholesterol well above normal). An impossible reading such as `systolic=1400` raises
`ValueError: implausible examination values: ap_hi out of range`.

**How the notebook is built.** In the repository the code lives in one file, `cardio.py`, which the
unit tests and the presentation build also use. `build_report.py` copies each section of that file
into the notebook, at the markers in `report.md`. The notebook is then executed in a folder that does
*not* contain `cardio.py`, so any hidden dependency on it would fail. This keeps one copy of the
code while the submitted notebook stands alone. `cardio.py` also still offers a command line
(`python cardio.py predict ...`), but that is not part of the submission.

### 2.3 `Cardio-Risk.pptx`: the presentation

A 10-minute talk: 15 main slides plus 3 backup slides.

- **Template:** built by `presentation/build.py` from the Slidesgo "Child Meningitis Case Study"
  template (the same one as your questionnaire defense), with its green recoloured to a cardiac red.
- **Transitions:** Morph on every slide. On slide 11 the model bars from slide 7 grow into the
  inflated "leakage" bars.
- **Script:** speaker notes on every slide hold the spoken script. The same text is in
  `presentation/script.md`: 1,147 words, about 8 minutes at 140 words a minute.

| # | Slide | Message |
|---|---|---|
| 1 | Title | name, Lebanese University (course and instructor left out: the brief doesn't name them) |
| 2 | Why this matters | 17.9 million deaths a year; the objective; Q1–Q3 |
| 3 | The data | 70,000 patients, 11 values, 50/50; measured / lab / self-reported |
| 4 | 2% of the records are impossible | the four error patterns; r = 0.05 → 0.43 |
| 5 | What the clean data shows | disease rate by factor (Figure 2) |
| 6 | The smoking paradox | the sick quitter; detection not prediction; associations not causes |
| 7 | What machine learning adds | test accuracy of every candidate on one axis |
| 8 | On patients the model has never seen | AUC 0.804, 73.6%, +301 patients; ROC curves |
| 9 | What drives the prediction | importance bars; blood pressure as a switch |
| 10 | A score a doctor can add up | the 4-factor scorecard and points → risk |
| 11 | How 90% accuracy happens | the leakage bars; the published 88% and 94.5% marked |
| 12 | Can the predictions be trusted? | calibration; sexes; over 60; 44% at 20% prevalence |
| 13 | Checking a new patient | the notebook call and its real output; a rejected reading |
| 14 | Conclusions | Q1–Q3 answers and limitations |
| 15 | Thank you | questions; template credit |
| 16–18 | Backup | full test table; cleaning strategies; subgroups |

### 2.4 `data/cardio.csv`

The dataset exactly as distributed in the course zip (Kaggle, Ulianova 2019): 70,000 rows and
13 columns, unchanged. All cleaning happens in code, so the original file is never edited.

### 2.5 `requirements.txt`

These are the exact versions the results were produced with. scipy, joblib and threadpoolctl are
pinned because a newer scipy printed a warning on every logistic regression fit.

| Library | Version |
|---|---|
| numpy | 2.2.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.15.3 |
| joblib | 1.4.2 |
| threadpoolctl | 3.5.0 |
| matplotlib | 3.10.0 |
| nbformat | 5.10.4 |
| nbconvert | 7.16.6 |
| ipykernel | 6.29.5 |
| pytest | 8.3.5 |

---

## 3. What we did, in order

### Step 1: reading the brief and choosing the dataset

We profiled all five datasets with pandas before choosing:

| Dataset | Finding | Verdict |
|---|---|---|
| Water potability (3,276) | no signal: class means identical, every correlation with the label below 0.035 | rejected: a model can barely beat guessing |
| **Cardio (70,000)** | real but modest signal, real data errors to fix, medical meaning | **chosen** |
| Email spam (5,172 × 3,000 word counts) | Enron emails; words like "enron" and "hpl" never appear in spam (a shortcut); 541 duplicate rows; 97%+ is easy | rejected: too easy, no raw text |
| Loan eligibility (614) | the single rule "credit history = 1" is already 80.9% accurate | rejected: tiny, one column decides |
| Credit-card segmentation (8,950) | clustering only, no labels | runner-up: hard to defend without a correct answer |

### Step 2: validating the idea against the literature

- **What careful studies get.** About 73% on this dataset (CatBoost 0.734, arXiv 2507.21898), close
  to our 71.3% rule. So the realistic gain from ML is small.
- **What some papers claim.** 88.09% (Omkari and Shaik, IEEE Access 2024) and 94.5% (Zafar et al.,
  Systems and Soft Computing 2026). That is very hard to reconcile with ~73%, which suggests leakage.
- **A precedent for the check.** A published comment (Eltawil et al., JCDD 2026) showed a 99% claim
  on another CVD dataset fall to 77% once SMOTE was moved after the split.
- **Validity issues found:**
  - The task is detection, not prediction.
  - Patients may already be on treatment.
  - The 50/50 balance is not real-world prevalence.
  - The source is undocumented.
- **Result.** The plan grew from "train models" into the three questions Q1–Q3.

### Step 3: keeping it the project the brief asks for

You asked that the result still read as the project described: a model that detects the disease. So
the report follows the conventional structure (data → cleaning → EDA → features → models → evaluation
→ use → conclusion). The research questions live inside the evaluation and reliability sections,
and §10 gives a working tool.

### Step 4: infrastructure, with all computation on dblp-vm

- **Repository.** Private, `hadigghazi/cardio-risk`.
- **Docker image.** Python 3.12 with the pinned libraries.
- **Tests.** 43 unit tests (26 test functions, some parametrised).
- **CI on GitHub.** On every push, GitHub's runner builds the image and runs the tests (about 50 s).
  On pushes to main it then calls the VM workflow.
- **The VM workflow** (`vm.yml`), modelled on the dblp project:
  - It SSHes into dblp-vm with three repo secrets.
  - It copies the commit there, builds the image on the VM, and runs the report in a container
    capped at 4 GB memory and 4 CPUs. The VM has no swap and also serves the dblp and AutoML sites,
    so a runaway job must die alone.
  - It sends the outputs back as a downloadable artifact.
- **Access setup.**
  - A dedicated SSH key (`cardio-risk-deploy`) was created. Auto mode blocked me from writing secrets,
    so you set them with `gh`.
  - First attempt: the pasted commands ran into ssh-keygen's passphrase prompt, so nothing was set.
  - Second attempt: the key's fingerprint was pasted into `authorized_keys` instead of the key.
  - Third attempt: the real key line was added, and the connection worked.
  - Lines 6–7 of `authorized_keys` are harmless leftovers from those attempts.

### Step 5: cleaning

We inspected the impossible values and found the four typing-error patterns. We wrote the rules,
`drop` and `repair`, and tested them. Then we confirmed the effect on the correlation: 0.054 → 0.428.

### Step 6: exploration and features

The training-only analysis found the four strong factors, the weak lifestyle answers, and the
smoking paradox. We then added the four derived features.

### Step 7: models

- **Verifying the Framingham coefficients.** We checked them against two sources and by hand: a
  50-year-old man with BMI 25 and 120 mmHg comes out at 8.48%.
- **The first VM run.** It placed the best settings of gradient boosting and the MLP at the edge of
  their search grids. A best value at the edge means a better one might lie beyond it.
- **Widening the grids.** The results moved by under 0.0005, so the models had reached the ceiling.

### Step 8: test evaluation, explanation, scorecard

We scored every candidate once on the test patients, with bootstrap intervals. We added the screening
cut-off, the grouped permutation importance and the partial dependence.

The points score earned negative points for lifestyle factors, so the clinical 4-factor card was
added. It turned out to lose almost nothing.

### Step 9: reliability

We trained on the three cleaning strategies, ran the three leakage experiments, and checked
calibration, subgroups and predictive value at realistic prevalence.

### Step 10: the tool

`train` and `predict` were added. The tool uses the clinical card, prints readable labels, and gives
a one-line rejection for impossible values instead of a crash.

### Step 11: the presentation

- **Format.** You chose PowerPoint, about 10 minutes.
- **How it's built.** The build script follows your earlier decks: template layouts, the template's
  own medical icons, Morph, and notes timed at 140 words a minute.
- **Numbers that can't drift.** The build reads the scorecard and the patient example from the
  saved model. It refuses an outdated model unless run with `--draft`.
- **QA.** Every slide was rendered through PowerPoint and inspected, and the obvious problems were
  fixed: two titles that wrapped, overlapping labels on the leakage slide, a chart too small to read
  (redrawn as native bars), and a duplicated credit line. The file passed the PowerPoint validator,
  with no leftover template text.

### Step 12: bugs caught before submission

1. **No figures in the notebook.** The Docker image set `MPLBACKEND=Agg`, which silently stopped
   Jupyter embedding plots. The figure files existed, but the report had no charts. We removed the
   setting, and the build now fails if fewer figures are embedded than the report draws. The final
   notebook has all 13.
2. **Scipy warnings.** A newer scipy than scikit-learn 1.6.1 expects warned on every fit; it is now
   pinned.
3. **A rough command-line tool.** It printed raw column names and crashed on bad input; both fixed.

### Step 13: one notebook instead of a notebook and a .py file

You confirmed the brief means *either* format, so we submit the notebook alone. Every section of
`cardio.py` now appears as code inside the notebook, and the command line was dropped from the
submission. Section 10 and slide 13 now show the check inside the notebook.

At the same time, the account's GitHub Actions allowance ran out ("spending limit"), so no job could
start. Following the rule from the groundtruth project, CI now runs only the tests on push, and the
report runs on demand. You then made the repository public, which makes Actions free, and the
notebook was re-run on the VM.

### Step 14: packaging

The final VM run's notebook was checked: 79 cells, 41 code cells executed, 13 figures, 0 errors.
Its test results match the run the deck was built from. It was committed to the repo, and the zip
was assembled.

---

## 4. Concepts you may need to explain

- **ROC AUC.** The probability that the model gives a randomly chosen sick patient a higher risk than
  a randomly chosen healthy one. 0.5 is chance and 1.0 is perfect. It doesn't depend on any cut-off,
  which is why it is the primary metric.
- **Accuracy, sensitivity, specificity.**
  - Accuracy: the share classified correctly.
  - Sensitivity: the share of sick patients detected.
  - Specificity: the share of healthy patients correctly cleared.
  - The last two depend on the cut-off.
- **Train/test split, and stratified.** The model learns from 80% of patients and is judged on 20% it
  has never seen. Stratified means both parts keep the 50/50 balance.
- **5-fold cross-validation.** The training data is cut into 5 parts. Each model trains on 4 and is
  scored on the 5th, five times over. It is used to choose models and settings without touching the
  test set.
- **Grid search.** Trying every combination of hyperparameter values and keeping the one with the
  best cross-validated score.
- **The one-standard-error rule.** Choose the simplest model whose score is within one standard error
  of the best. Differences smaller than the noise should not buy complexity.
- **Bootstrap confidence interval.** Resample the test patients with replacement 2,000 times and
  recompute the metric each time; the middle 95% is the interval. Resampling the same patients for
  every model also gives intervals for the *differences* between models.
- **Calibration and the Brier score.**
  - Calibrated means a predicted 70% comes true about 70% of the time.
  - The Brier score is the mean squared error of the predicted risks: lower is better, and always
    predicting 50% scores 0.25.
- **Permutation importance, grouped.** Shuffle a feature across patients and measure how much the AUC
  drops. Related features (the 5 blood-pressure ones) are shuffled together, so they cannot stand in
  for each other.
- **Partial dependence.** Set one factor to a value for everyone and average the predicted risk. It
  shows the *shape* of an effect.
- **Gradient boosting.** Many small decision trees, each correcting the errors of the previous ones.
  Trees handle thresholds and interactions naturally, which is why it beats a straight-line model
  here.
- **Logistic regression.** A weighted sum of the features passed through an S-curve. It is simple
  and interpretable, but each feature can only push the log-odds in a straight line.
- **SMOTE.** Creates synthetic patients by interpolating between a patient and a similar one.
  - **Done after the split** (training data only), it can help imbalanced data. Here the data is
    balanced, so it only hurt.
  - **Done before the split**, synthetic near-copies of training patients end up in the test set.
    That is leakage.
- **Leakage.** Any way information from the test patients reaches training or model choice. It
  inflates reported accuracy without making the model better.
- **Positive predictive value (PPV).** The share of flagged patients who are really sick. Unlike
  sensitivity and specificity, it depends on how common the disease is.
- **The sick-quitter effect.** People who are diagnosed change their habits, so in data collected
  after diagnosis, the habits look protective.
- **Framingham risk score.** A standard clinical equation (2008) estimating 10-year risk of a first
  CVD event from age, sex, BMI, systolic pressure, smoking and diabetes.

---

## 5. Questions you may be asked

**Why only 73.6%? Isn't that low?** It is the ceiling for these 11 values. Careful studies get the
same (~73%), the learning curve is flat, and five different algorithms all land within 0.01 AUC.
Higher published figures are reproducible only with leaky evaluation (§9.2).

**Then why use ML at all?** The gain over the rule is real: +301 sick patients found at the same
false-alarm rate. The model also gives a calibrated risk rather than a yes/no answer. And the work
produced a 4-factor scorecard nearly as good as the best model.

**Why gradient boosting and not random forest?** The rule fixed in advance chose it. Random forest was
0.0013 behind against a margin of 0.0012, a near tie, and gradient boosting is also faster. On the
test set they differ by under 0.002, so in practice either is fine.

**Did you clean the test set?** Cleaning rules are fixed per record and learn nothing, so applying
them to test records leaks nothing. Strategies are compared on the same test patients (those valid
under every strategy). A real system would reject impossible inputs at entry, which is what the tool
does.

**Why not SMOTE?** The classes are already 50/50, so there is nothing to balance. Applied correctly,
SMOTE made the models worse (64–71%). Applied before the split, it inflated accuracy to 85–87%.

**Why does smoking lower the risk?** It doesn't, biologically. In this data, diagnosed patients have
usually quit, so smokers look healthier (the sick-quitter effect). The model is told to describe
associations, and the clinical scorecard leaves smoking out.

**Is the model fair?** Women and men get the same AUC and calibration. It is weaker for patients
over 60, and a single cut-off doesn't suit every age: that is reported as a limitation.

**Can it predict who will get heart disease?** No. Diagnosis and features come from the same visit,
so it detects existing disease. Prediction would need patients followed over time.

**Why is there no .py file?** The brief accepts the work as a notebook or as a Python file; we chose
the notebook. It contains all the code (52 code cells), so it runs on its own with just the data
file.

**Why run it on a VM?** To keep the computation reproducible and off the laptop. Every result in the
notebook comes from a Docker container on dblp-vm, built from a specific commit. CI runs the tests on
every push and the full report on pushes to main.

---

## 6. What is in the repository but not in the zip

| Path | Purpose |
|---|---|
| `cardio.py` | the single source of all the code; the notebook build copies it in section by section; also a command line |
| `report.md` | the source of the report; `build_report.py` turns it into the notebook and executes it |
| `build_report.py` | Markdown → notebook converter: copies in the code sections, executes without `cardio.py`, fails if figures are missing |
| `tests/` | 46 unit tests (cleaning rules, features, Framingham, metrics, scorecard, patient check, notebook build) |
| `Dockerfile`, `.dockerignore` | the reproducible environment |
| `.github/workflows/ci.yml` | tests on every push (about a minute) |
| `.github/workflows/vm.yml` | run the report on dblp-vm on demand; also collect, log, stop, status |
| `presentation/` | `build.py`, the template, icons, figures, `script.md`, the deck |
| `brief/` | the two course documents |
| `outputs/` (not committed) | local copies of VM outputs, including `model.joblib` |
| `submission/` (not committed) | the zip |

Repository: https://github.com/hadigghazi/cardio-risk (private).
