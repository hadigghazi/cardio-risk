# Predicting cardiovascular disease from a routine examination

Course project, topic 2 (Patient's Sickness Prediction): `data/cardio.csv`, 70,000 patients.
The brief is in `brief/`. Deliverables: the report notebook (with all the code inside it) and a presentation.

| File | What it is |
|---|---|
| `cardio.py` | single source of all project code; `build_report.py` copies its sections into the notebook; also a CLI |
| `report.md` | source of the report; `build_report.py` turns it into `outputs/report.ipynb` and executes it |
| `report.ipynb` | the executed report from the latest VM run (the deliverable) |
| `PROJECT-GUIDE.md` | the whole project explained: every decision, every notebook cell, the results and why they can be trusted |
| `tests/` | unit tests: cleaning and repair rules, features, Framingham, metrics and cut-offs, scorecard, patient checks, notebook build |
| `.github/workflows/ci.yml` | every push: tests on GitHub's runner (the report runs on demand, see below) |
| `.github/workflows/vm.yml` | runs the report on dblp-vm on demand, over SSH (secrets VM_HOST, VM_USER, VM_SSH_KEY), in a capped container |

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
| 7 | Models: baselines, LR, RF, gradient boosting, MLP; 5-fold CV (Q1) | done |
| 8 | Final model on the test set; explanation; points score | done |
| 9 | Reliability: cleaning strategies, reproduced leakage, calibration, subgroups (Q2, Q3) | done |
| 10 | Using the model on a new patient (`assess(load_final(), patient(...))`) | done |
| 11 | Conclusion, limitations, references | done |
| - | Presentation (`presentation/`) | done |

## Presentation

`presentation/build.py` builds `presentation/Cardio-Risk.pptx` (15 slides + 3 backup, Morph
transitions, speaker notes) from the Slidesgo "Child Meningitis Case Study" template recoloured red. It
reads the scorecard and the patient example from `outputs/model.joblib`, so fetch the latest VM run's
model first. `presentation/script.md` is the spoken script (~8 minutes at 140 words a minute).

## Submission

The zip holds `report.ipynb` (executed on the VM, all code inside it), `Cardio-Risk.pptx`, plus
`data/cardio.csv` and `requirements.txt` so the notebook can be re-run. No `.py` file: the brief accepts
either format, and the notebook is self-contained.

## Running

On the VM (Actions → VM → Run workflow): `action=run`, `job=report`, about 15 minutes. When only
the text of `report.md` changes afterwards, `python build_report.py --reuse-outputs <executed.ipynb>`
rebuilds the notebook with that run's outputs; it refuses if any code cell differs. The executed notebook,
figures and tables come back as the run's artifact; a job that outlives `wait` keeps running and is
fetched later with `action=collect` and its `run_id`.

Locally, with pandas, scikit-learn, matplotlib and nbconvert available:

```
python -m pytest
python cardio.py clean
python build_report.py
```
