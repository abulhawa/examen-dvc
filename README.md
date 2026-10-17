# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.

---

## Project implementation guide

This project predicts `silica_concentrate` using a reproducible DVC pipeline.

- GitHub: https://github.com/abulhawa/examen-dvc
- DagsHub: https://dagshub.com/abulhawa/examen-dvc

## Setup

Run commands from the repository root. The working environment uses Python 3.14.7 on Ubuntu under WSL2; other Python versions have not been verified.

```bash
git clone https://github.com/abulhawa/examen-dvc.git
cd examen-dvc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, with uv and Python 3.14 available:

```bash
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The requirements pin the main packages, including DVC and its S3 plugin. Transitive dependencies are not fully locked.

## Retrieve data and models

The tracked `.dvc/config` points to DagsHub S3 storage. Authenticate with your own DagsHub account and token with access to this repository. Credentials belong in the ignored `.dvc/config.local`, never in Git.


```bash
dvc pull
dvc status
```

`dvc pull` retrieves the raw data, split and scaled datasets, best parameters, trained model, and predictions. The small evaluation scores file is tracked directly in Git.

## Pipeline

| Stage | Script | Main outputs |
| --- | --- | --- |
| split | `src/data/split_data.py` | Training/test features and targets in `data/processed_data/` |
| normalize | `src/data/normalize_data.py` | Scaled training/test features |
| GridSearch | `src/models/grid_search.py` | `models/best_params.pkl` |
| Train | `src/models/train.py` | `models/model.pkl` |
| Evaluate | `src/models/evaluate.py` | `data/predictions.csv`, `metrics/scores.json` |

The split removes the text `date` column and uses `silica_concentrate` as the target, with a 20% test split and `random_state=42`. StandardScaler is fitted only on training data and then applied to both feature sets.

GridSearchCV compares six Random Forest configurations using five shuffled folds on the training set, selecting by mean squared error. Training fits a RandomForestRegressor using the selected parameters and `random_state=42`. Evaluation uses the held-out test set of 364 rows.

```bash
# Run changed stages and their prerequisites.
dvc repro

# Inspect the pipeline and saved metrics.
dvc dag
dvc metrics show

# Force all stages to run again after retrieving the data.
dvc repro --force
```

Keep the virtual environment active when running DVC: the stage commands use `python`.

## Recorded test results

| Metric | Value |
| --- | ---: |
| MSE | 0.7715305612 |
| R2 | 0.2291450220 |

These results describe the current held-out test predictions. The goal of this evaluation is a reproducible DVC workflow; predictive performance is modest. The scaler is fitted before the internal GridSearch cross-validation, so internal validation folds share training-wide scaling statistics. Random Forest does not require scaling; the separate normalization stage follows the exam workflow.

## Save a new checkpoint

After making changes, run `dvc repro`, review the changes, and commit the scripts, pipeline definitions, lock file, ignore files, and metrics to Git. Upload cached outputs using `dvc push`, then push the Git commit to GitHub. DagsHub mirrors the GitHub repository.

Do not add datasets, model binaries, `.venv/`, or `.dvc/config.local` directly to Git. Only load pickle files from trusted sources.

## Exam submission

Grant `licence.pedago` read-only collaborator access on DagsHub. Submit a ZIP containing a Markdown file with your last name, first name, email address, and the DagsHub repository link. Keep personal submission details outside this public repository.
