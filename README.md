# LoanCheck
MLOps assessment in which we build a model that looks at the loan applicant's details and predicts the approval.

The Docker CI job builds, scans, generates an SBOM, and pushes the image, but does not
run a live smoke test because API startup requires an MLflow champion model and hosted
CI cannot reach the project's local MLflow server. The smoke test is run locally
instead. These Docker steps are also skipped when hosted CI cannot retrieve the required
model artifact from the project's local DVC remote; configure a cloud remote to enable them.
