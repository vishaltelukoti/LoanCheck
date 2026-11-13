# LoanCheck — Project Charter

## 1. The problem

A housing finance company receives many home loan applications every day. Each application is currently reviewed by hand, which is slow and inconsistent. The company needs a system that looks at an applicant's income, loan amount, loan term, credit history, property area, and education, and quickly flags each application as **"likely to be approved"** or **"needs a closer look."** This lets staff spend their time on the genuinely difficult cases instead of re-checking every application from scratch.

## 2. Who uses the predictions

**Loan officers** are the primary users. They will see the model's predicted approval probability (e.g., 0.91) alongside each application and use it as a triage signal — not a final decision. A human always makes the actual approve/reject call; the model's job is to speed up the easy cases and surface the ambiguous ones.

## 3. How we measure success

- At least **80% accuracy** on held-out applications.
- Very few **bad loans marked as approved** — i.e., false positives (predicting approval for an application that should be rejected) are tracked and minimized specifically, not just overall accuracy.
- The model, once deployed, is **reproducible**: given the same code version, data version, and parameters, it produces the same result every time.
- Every prediction can be **traced back** to the exact code, data, and model version that produced it.

## 4. Which mistake costs more?

**Approving a loan that should be rejected costs more than rejecting a good applicant.** A bad approval can lead to a loan default — direct financial loss, plus the cost of collections or write-off. A wrongly rejected good applicant is also a real cost (lost business, reputational harm, and the applicant may go to a competitor), but it's recoverable: the applicant can appeal, reapply, or be caught by a human reviewer. Because of this asymmetry, LoanCheck is tuned to be cautious about approvals — the quality bar in Task 6 should weight **false positives (bad approvals)** more heavily than false negatives when comparing models, and borderline cases should default to "needs a closer look" rather than "approved."

## 5. The ML lifecycle and who owns each step

| Step | What happens | Owner |
|---|---|---|
| Get data | Pull the raw loan applications (Kaggle dataset / production source) | Data engineer |
| Check data | Validate schema, ranges, missing values, drift | Data engineer / ML engineer |
| Prepare features | Fill missing values, engineer features (e.g., total income, loan-to-income ratio) | Data scientist |
| Train | Fit candidate models (Logistic Regression, Decision Tree, Random Forest) | Data scientist |
| Evaluate | Score models against the quality bar (accuracy ≥ 0.78, ROC-AUC ≥ 0.75) | Data scientist |
| Register model | Promote a model to `challenger`, then `champion` if it beats the current best | ML engineer |
| Package | Build the Docker image containing the API and the champion model | ML engineer |
| Serve | Run the API in production, answering `/predict` requests | ML engineer |
| Monitor | Watch for data drift, prediction drift, and model degradation over time | ML engineer / reviewer |

On a one-person project, I (Vishal) hold all of these roles, but I'm keeping the separation explicit — it's the structure that scales to a real team and it's what the CI pipeline and PR reviews are standing in for.

## 6. Lineage

Lineage means being able to answer, for any prediction or any saved model: *which exact code, data, and training run produced this?* It's done in three layers that are tagged together:

1. **Git** records which version of the code (features, training script, API) was used — identified by a commit hash.
2. **DVC** records which version of the data (raw → interim → processed) was used — identified by a data hash tracked alongside the Git commit.
3. **MLflow** records the training run itself — the parameters, metrics, and resulting model artifact — and each run is tagged with the Git commit ID and the DVC data version.

So a model in the registry is never a loose file; it always points back to the exact commit and data snapshot that created it, which means any prediction can be fully explained and reproduced later.

## 7. Environments

| | Dev | Staging | Prod |
|---|---|---|---|
| **Data source** | Small local sample / synthetic subset of the Kaggle dataset | Full dataset, held in the shared DVC remote | Live application data pulled from the production system |
| **Settings** (`configs/*.yaml`) | Relaxed quality bar, verbose logging, local MLflow tracking URI | Same quality bar as prod, points at staging MLflow/registry | Full quality bar enforced, points at production MLflow registry and `@champion` alias |
| **Log level** | `DEBUG` — full tracebacks and request/response bodies for troubleshooting | `INFO` — enough to debug issues without leaking applicant PII | `WARNING`/`ERROR` only — no applicant data in logs, for privacy and noise reduction |
| **Docker image** | Built locally, not scanned | Built and scanned by CI, pushed to registry with a `-staging` tag | Same image promoted from staging (not rebuilt), tagged with Git commit + version |
