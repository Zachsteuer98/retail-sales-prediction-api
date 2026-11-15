# Architecture decision records

## ADR-1: Keep Flask for the API
**Context:** The original deployment used Flask, and FastAPI would give validation and docs for free.
**Decision:** Keep Flask. Add Pydantic validation, JSON error handlers and gunicorn.
**Consequences:** Preserves the original API contract and shows a second framework alongside my FastAPI
projects. There is no auto-generated OpenAPI page; `/v1/schema` and the README document the contract.

## ADR-2: Validate inputs against what training covered
**Context:** The data contains 4 stores, and each store's attributes never vary. The one-hot encoder
uses `handle_unknown="ignore"`, so unseen categories (e.g. `Supermarket Type3`) and unseen store
combinations produced plausible-looking numbers with no warning.
**Decision:** Learn an input schema at training time (known stores, categories, numeric ranges,
product prefix ↔ sugar pairs), save it with the model, and reject violations with HTTP 422 and
specific messages.
**Consequences:** The API refuses some requests the original accepted, including the brief's batch file.
That is the honest behaviour: those rows describe stores the model knows nothing about.

## ADR-3: Select by cross-validation, score test once
**Context:** The original notebook fit preprocessing before the split and chose the champion on test
scores, which makes reported test performance optimistic.
**Decision:** Put preprocessing inside the pipeline, choose by 5-fold CV RMSE on the training split,
and report test metrics once. The production model is then refit on all rows with the chosen settings.
**Consequences:** Test metrics are an honest estimate. They turned out almost identical to the
original, which confirms the original choice.

## ADR-4: Two Hugging Face Docker Spaces, created by CI
**Context:** The demo needs free public hosting, and manual setup caused configuration typos last time.
**Decision:** The deploy workflow creates both Spaces and sets the frontend's `API_URL`. Only
`HF_TOKEN` is configured by hand.
**Consequences:** Setup is one secret. Free Spaces sleep when idle, so the UI uses long timeouts.
