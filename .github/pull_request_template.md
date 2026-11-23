## Description

<!-- Describe what changed, why it changed, and any notable implementation details. -->

## Checklist

### Code quality

- [ ] `ruff` passes with no errors (style + lint)
- [ ] `mypy` passes with no errors (type hints on all new/changed functions)
- [ ] No hard-coded paths, numbers, or secrets in the code — settings come from `configs/*.yaml` via `config.py`
- [ ] No real secrets committed — only `.env.example` with empty values

### Tests

- [ ] All `pytest` tests pass
- [ ] Coverage stays at or above 80% (`pytest --cov`)
- [ ] New logic has at least one corresponding test

### Process

- [ ] Change was made on a short-lived branch (`feat/...`, `fix/...`), not directly on `main`
- [ ] Commit messages follow Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`)
- [ ] PR was reviewed by at least one other person (trainer/colleague) where required by Task 3
- [ ] Relevant docs (`README.md`, `docs/*.md`) are updated if behavior changed
- [ ] CI is green on all current jobs: lint, typecheck, test

<!-- Data quality / Model quality / Container sections are added once Tasks 4, 6, and 7 land. -->
