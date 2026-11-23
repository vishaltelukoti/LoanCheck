# LoanCheck — Definition of Done

Every change (every pull request, from Task 3 onward) must satisfy all of the following before it is merged into `main`. This checklist is also copied into `.github/pull_request_template.md` so it's checked on every PR.

## Code quality
- [ ] `ruff` passes with no errors (style + lint)
- [ ] `mypy` passes with no errors (type hints on all new/changed functions)
- [ ] No hard-coded paths, numbers, or secrets in the code — settings come from `configs/*.yaml` via `config.py`
- [ ] No real secrets committed — only `.env.example` with empty values

## Tests
- [ ] All `pytest` tests pass
- [ ] Coverage stays at or above 80% (`pytest --cov`)
- [ ] New logic (features, training, prediction) has at least one corresponding test

## Data quality (once Task 4 is in place)
- [ ] `pandera` schema contract passes (column names, types, allowed values, ranges, unique `Loan_ID`)
- [ ] No column has more than 10% missing values without an explicit, documented fill strategy
- [ ] Drift check against the scoring file passes (PSI < 0.1 = pass, 0.1–0.25 = warn, ≥ 0.25 = fail)
- [ ] Leakage test passes — `Loan_ID` and `Loan_Status` are never used as features, and train/test split happens before any imputation or scaling
- [ ] `reports/data_quality.json` is generated and attached as a CI artifact

## Model quality (once Task 6 is in place)
- [ ] Model is logged to MLflow with params, metrics, and artifacts (confusion matrix, feature importance)
- [ ] Run is tagged with the Git commit ID and DVC data version
- [ ] Model meets the quality bar: accuracy ≥ 0.78 and ROC-AUC ≥ 0.75
- [ ] Model is only promoted to `@champion` if it beats the current champion on these metrics
- [ ] `docs/model_card.md` is updated if the champion model changed

## Pipeline reproducibility (once Task 5 is in place)
- [ ] `dvc repro` runs clean from a fresh clone after `dvc pull`
- [ ] `dvc.lock` is committed and matches the latest `dvc repro` output
- [ ] Changing a parameter in `params.yaml` and re-running produces a visibly different, explainable result in `dvc metrics diff`

## Containers (once Task 7 is in place)
- [ ] Docker image builds successfully via the multi-stage `Dockerfile`
- [ ] Image runs as a non-root user and defines a `HEALTHCHECK`
- [ ] Trivy scan has no CRITICAL vulnerabilities with an available fix
- [ ] SBOM is generated (Syft) and attached as a CI artifact
- [ ] Smoke test passes: container starts, `/health` returns `ok`, `/predict` returns a sane response

## Process
- [ ] Change was made on a short-lived branch (`feat/...`, `fix/...`), not directly on `main`
- [ ] Commit messages follow Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`)
- [ ] PR was reviewed by at least one other person (trainer/colleague) where required by Task 3
- [ ] Relevant docs (`README.md`, `docs/*.md`) are updated if behavior changed
- [ ] CI is green on all jobs: lint, typecheck, test, data-check, pipeline, docker
