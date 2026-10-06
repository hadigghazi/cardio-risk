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
