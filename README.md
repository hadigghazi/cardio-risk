# Predicting cardiovascular disease from a routine examination

Course project, topic 2 (Patient's Sickness Prediction): `data/cardio.csv`, 70,000 patients.
The brief is in `brief/`. Deliverables: the report notebook, the Python file and a presentation.

| File | What it is |
|---|---|
| `cardio.py` | all project code: loading, cleaning, features, (later) models and evaluation; also a CLI |
| `report.md` | source of the report; `build_report.py` turns it into `outputs/report.ipynb` and executes it |
| `tests/` | unit tests for the cleaning rules and features |
| `.github/workflows/ci.yml` | every push: tests + report built on GitHub's runner (artifact `report-ci`) |
| `.github/workflows/vm.yml` | manual: run a job on dblp-vm in a capped container, collect its outputs |

## Report plan

The report follows the brief: build a model that detects whether a patient has cardiovascular
disease, then evaluate it honestly. Three questions guide the evaluation: **Q1**, what ML adds over
clinical baselines (the systolic ≥ 130 rule and the Framingham office-based score); **Q2**, how much
data handling (cleaning, leakage) changes the measured performance; **Q3**, whether the predictions
can be trusted and explained.

| § | Section | Status |
|---|---|---|
| 1-2 | Introduction, evaluation protocol | done |
| 3-4 | Data, data quality and cleaning | done |
| 5 | Exploratory analysis (training patients only) | done |
| 6 | Feature engineering | done |
| 7 | Models: baselines, LR, RF, gradient boosting, MLP; 5-fold CV (Q1) | next, on the VM |
| 8 | Final model on the test set; explanation; points score | |
| 9 | Reliability: cleaning strategies, reproduced leakage, calibration, subgroups (Q2, Q3) | |
| 10 | Using the model on a new patient | |
| 11 | Conclusion and limitations | |
| - | Presentation | after the report |

## Running

On the VM (Actions → VM → Run workflow): `action=run`, `job=report`. The executed notebook,
figures and tables come back as the run's artifact; a job that outlives `wait` keeps running and is
fetched later with `action=collect` and its `run_id`.

Locally, with pandas, scikit-learn, matplotlib and nbconvert available:

```
python -m pytest
python cardio.py clean
python build_report.py
```
