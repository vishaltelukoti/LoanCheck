# LoanCheck

LoanCheck is an MLOps project for triaging home-loan applications. It gives loan officers a predicted approval probability so they can prioritize straightforward cases and spend more time on ambiguous applications. It is a decision-support tool—not an automated approval system—and a human remains responsible for the final lending decision.

## Key Features

- Schema-validated raw-data pipeline using Pandera contracts.
- DVC-versioned data, processed features, imputation statistics, and model artifacts.
- MLflow experiment tracking and model registry with `challenger` and `champion` aliases.
- FastAPI serving that imputes supported missing application fields using statistics fitted on the training split.
- Multi-stage Docker image with a non-root runtime user and health check.
- GitHub Actions CI with six jobs: `lint`, `typecheck`, `test`, `data-check`, `pipeline`, and `docker`.
- Git, DVC, and MLflow lineage tags to connect code, data, parameters, and registered models.

## Architecture Overview

```text
Raw Kaggle loan data
        |
        v
+-----------------------------------------------+
| DVC pipeline                                  |
| ingest -> validate -> featurize -> train      |
|                         -> evaluate           |
+-----------------------------------------------+
        |                         |
        |                         v
        |                 reports/metrics.json
        v
MLflow experiment tracking and registry
        |
        v
loancheck@champion
        |
        v
FastAPI container
  GET  /health
  POST /predict
        |
        v
Loan officers use approval probability
for application triage
```

## Project Structure

```text
loancheck/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── charter.md
│   ├── definition_of_done.md
│   └── model_card.md
├── models/
├── reports/
├── scripts/
├── src/
│   └── loancheck/
│       ├── api/
│       ├── data/
│       ├── features/
│       ├── models/
│       └── config.py
├── tests/
├── dvc.yaml
├── params.yaml
├── pyproject.toml
└── Dockerfile
```

## Machine Learning Pipeline

The pipeline compares logistic regression, decision tree, and random forest classifiers. The current quality bars in `params.yaml` are accuracy ≥ 0.78 and ROC-AUC ≥ 0.75.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8618 | 0.8400 | 0.9882 | 0.9081 | 0.8601 |
| Decision tree | 0.7236 | 0.8493 | 0.7294 | 0.7848 | 0.7200 |
| Random forest | 0.8537 | 0.8681 | 0.9294 | 0.8977 | 0.8605 |

Logistic regression is the current champion because it has the highest accuracy while exceeding both quality bars. Its recall and F1 score are also the strongest among the candidates, and its simpler, interpretable form makes it a suitable baseline for this project.

The five DVC stages are:

1. `ingest`
2. `validate`
3. `featurize`
4. `train`
5. `evaluate`

The latest evaluated champion metrics in `reports/metrics.json` are:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8617886178861789 |
| Precision | 0.84 |
| Recall | 0.9882352941176471 |
| F1 | 0.9081081081081082 |
| ROC-AUC | 0.860061919504644 |

## Data Validation

Raw loan applications are validated with a Pandera schema covering identifiers, categorical values, numeric ranges, and the optional training-only `Loan_Status` target.

Missing values are checked against the `missingness_limit_pct` of 10 in `params.yaml` before imputation:

| Strategy | Columns |
| --- | --- |
| Mode | `Credit_History`, `Self_Employed`, `Gender`, `Loan_Amount_Term` |
| Median | `LoanAmount` |

`LoanAmount` uses the median because loan amounts are skewed by outliers. Categorical and binary fields use their mode.

The pipeline also computes Population Stability Index (PSI) for `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount`.

| PSI status | Threshold |
| --- | ---: |
| Pass | PSI < 0.1 |
| Warn | PSI < 0.25 |
| Fail | PSI ≥ 0.25 |

The current `reports/data_quality.json` status is `warn`:

| Column | PSI | Status |
| --- | ---: | --- |
| `ApplicantIncome` | 0.245673571714364 | Warn |
| `CoapplicantIncome` | 0.15188320839950026 | Warn |
| `LoanAmount` | 0.09462698522474738 | Pass |

The income PSI warnings are expected for this static Kaggle train/test split because income is skewed and the check uses a small bin count.

For fairness, `Gender` is validated for data quality but is excluded from model features. `Loan_ID` and `Loan_Status` are also excluded from features to prevent identifier and target leakage.

## API Endpoints

Start the API locally:

```bash
uv run uvicorn loancheck.api.main:app --host 0.0.0.0 --port 8000
```

### `GET /health`

Returns a liveness response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Returns the current MLflow champion model’s approval probability and a thresholded approval decision.

Required request fields:

- `Education`
- `ApplicantIncome`
- `CoapplicantIncome`
- `Property_Area`

Optional request fields, because they have observed missingness in training data and are imputed with training-fitted statistics:

- `Married`
- `Dependents`
- `Self_Employed`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`

`Gender` is intentionally not accepted by the API because it is excluded from model features.

Example request:

```json
{
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500.0,
  "LoanAmount": 128.0,
  "Loan_Amount_Term": 360.0,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

Response shape:

```json
{
  "approval_probability": <float>,
  "approved": <boolean>
}
```

`approved` is `true` when `approval_probability` meets the configured `approval_threshold` of 0.5.

## Local Setup

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd loancheck
uv sync
```

LoanCheck requires Python 3.12 or later.

### 2. Start MLflow

In a separate terminal:

```bash
uv run python scripts/start_mlflow_server.py
```

This starts MLflow on port 5000 using the project-local SQLite metadata store and artifact location.

### 3. Reproduce the DVC pipeline

```bash
uv run dvc repro
```

The pipeline uses `random_seed: 42` and `test_size: 0.2` from `params.yaml`.

### 4. Train and register candidate models

With the MLflow server running:

```bash
uv run python scripts/run_train.py
```

This logs the three candidate models to MLflow, registers the best qualifying model as `challenger`, and promotes it to `champion` when it beats the current champion under the configured rule.

### 5. Run the API

```bash
uv run uvicorn loancheck.api.main:app --host 0.0.0.0 --port 8000
```

## Running with Docker

The Docker image expects `models/missing_value_stats.json`, while model weights are loaded from the MLflow registry at API startup.

Build the image:

```bash
docker build -t loancheck-api:local .
```

Run it locally on host port 8001:

**Linux / Mac**

```bash
docker run --rm -p 8001:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  loancheck-api:local
```

**Windows PowerShell**

```powershell
docker run --rm -p 8001:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  loancheck-api:local
```

On Windows and Docker Desktop, `host.docker.internal` lets the container reach the MLflow server running on the host machine.

Send a full application payload:

**Linux / Mac**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500.0,
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }'
```

**Windows PowerShell**

```powershell
$payload = @{
  Married = "Yes"
  Dependents = "1"
  Education = "Graduate"
  Self_Employed = "No"
  ApplicantIncome = 5000
  CoapplicantIncome = 1500.0
  LoanAmount = 128.0
  Loan_Amount_Term = 360.0
  Credit_History = 1
  Property_Area = "Urban"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8001/predict" `
  -ContentType "application/json" `
  -Body $payload
```

Send an incomplete application to exercise the imputation path. Both examples deliberately omit `Credit_History`:

**Linux / Mac**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500.0,
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Property_Area": "Urban"
  }'
```

**Windows PowerShell**

```powershell
$payload = @{
  Married = "Yes"
  Dependents = "1"
  Education = "Graduate"
  Self_Employed = "No"
  ApplicantIncome = 5000
  CoapplicantIncome = 1500.0
  LoanAmount = 128.0
  Loan_Amount_Term = 360.0
  Property_Area = "Urban"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8001/predict" `
  -ContentType "application/json" `
  -Body $payload
```

## Running Tests

Run the complete test suite:

```bash
uv run pytest
```

The suite covers configuration loading, raw-data schema expectations, missing-value handling, feature construction, drift calculations, model training, lineage helpers, and API validation and imputation behavior.

For static analysis:

```bash
uv run ruff check .
uv run mypy src/
```

## CI/CD

GitHub Actions runs these jobs on pull requests to `main` and pushes to `main`:

| Job | Purpose |
| --- | --- |
| `lint` | Runs Ruff checks. |
| `typecheck` | Runs strict mypy checks for `src/`. |
| `test` | Runs pytest with coverage output. |
| `data-check` | Attempts to pull DVC data, then runs raw-data quality checks when the data is available. |
| `pipeline` | Attempts to pull DVC data, then reproduces the DVC pipeline only when the raw training artifact is available. |
| `docker` | Attempts to retrieve DVC-tracked model artifacts, then builds, scans, generates an SBOM for, and conditionally pushes the image. |

## Technologies Used

| Technology | Use in LoanCheck |
| --- | --- |
| Python ≥ 3.12 | Project runtime |
| uv | Dependency and virtual-environment management |
| FastAPI | HTTP API serving |
| Uvicorn | ASGI server |
| scikit-learn | Preprocessing pipelines, classifiers, and evaluation |
| Pandera | Data schema validation |
| DVC | Data, artifact, and pipeline versioning |
| MLflow | Experiment tracking, lineage, registry, and model loading |
| pandas | Tabular data processing |
| NumPy | Numerical feature operations |
| Matplotlib | Training diagnostic plots |
| PyYAML | Pipeline and parameter configuration parsing |
| Docker | Containerized API deployment |
| GitHub Actions | Continuous integration |
| Ruff | Linting and formatting checks |
| mypy | Strict static type checking |

## Known Limitations

The Docker CI job builds, scans, generates an SBOM, and pushes the image, but does not run a live smoke test because API startup requires an MLflow champion model and hosted CI cannot reach the project's local MLflow server. The smoke test is run locally instead. These Docker steps are also skipped when hosted CI cannot retrieve the required model artifact from the project's local DVC remote; configure a cloud remote to enable them.

The `pipeline` and `data-check` CI jobs also depend on a local DVC remote path that is not reachable from hosted GitHub runners. They degrade gracefully until a cloud remote such as S3, Azure, or Google Drive is configured.

The current champion was promoted manually to version 2 because it tied version 1 on all metrics but was trained from a clean, non-dirty Git commit. The automated promotion rule does not account for that lineage distinction.

The project uses a static Kaggle train/test split. Its documented income PSI warnings are expected for that split and should not be interpreted as an uninvestigated production-data drift alert.

## Author

Vishal Telukoti