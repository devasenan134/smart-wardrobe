# Contributing to Smart Wardrobe

Thanks for your interest! The project is early, so contributions of any size help: code, data, docs, reviews, or just a well-argued issue.

## Before you start

1. Read the [README](README.md) and [docs/ROADMAP.md](docs/ROADMAP.md) to see where the project is heading and why.
2. Look at the current phase notes (for now [docs/phase1-garment-understanding.md](docs/phase1-garment-understanding.md)).
3. **Open an issue first** for anything bigger than a small fix, so we can agree on the approach before you spend time on it.

## Development setup

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/devasenan134/smart-wardrobe.git
cd smart-wardrobe
git config core.hooksPath .githooks
```

**Backend**

```bash
cd services/api
uv run uvicorn app.main:app --reload
uv run pytest
```

The API uses a stub analyzer by default, so no GPU or model is needed.

**ML**

```bash
cd ml
uv sync
```

Datasets are never committed. They are downloaded into `ml/data/` (see [ml/data/README.md](ml/data/README.md)). Model weights (`*.pt`, `*.onnx`) and experiment runs are gitignored too.

## Ground rules

- **The contract is the boundary.** The backend and the ML only talk through [`services/api/app/ml/contract.py`](services/api/app/ml/contract.py). If you change it, update both sides and explain why in the PR.
- **Map datasets into the taxonomy, not the other way round.** [`ml/configs/taxonomy.yaml`](ml/configs/taxonomy.yaml) describes what the app needs. New datasets get a mapping config like `ml/configs/fashionpedia_map.yaml`.
- **Log every dataset and pretrained weight** in [docs/decisions.md](docs/decisions.md) with its source, version and license, and say whether commercial use is allowed. Non-commercial assets are fine for experiments, but they must be marked as such.
- **Measure against a baseline.** A new model PR should report its metrics next to the current baseline on the same evaluation set.
- **Privacy matters.** Never commit real user photos or personal data. Golden-set images must be ones you have the right to share.
- **Write down decisions.** If you make a non-obvious design choice, add a dated line to `docs/decisions.md`.

## Pull requests

- Keep PRs focused: one change per PR.
- Add or update tests for backend changes (`services/api/tests/`).
- Describe what changed, why, and how you checked it (tests, metrics, screenshots).
- By submitting a contribution you agree that it is licensed under the [Apache License 2.0](LICENSE), as described in section 5 of the license.

## Good first contributions

- Tests for API edge cases.
- EDA questions from the phase notes that are still open.
- Finding and license-checking datasets for shoes, menswear, swimwear or traditional wear.
- Fixes to the docs: typos, unclear explanations, broken links.

## Conduct

Be kind, assume good intent, and keep feedback about the work, not the person. Harassment or discrimination of any kind is not tolerated. Maintainers may remove comments or contributors that break this.
