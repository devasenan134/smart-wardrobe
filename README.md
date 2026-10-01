# Smart Wardrobe

**An open-source, privacy-first wardrobe assistant, with the ML built from scratch and in the open.**

Take photos of your clothes. Smart Wardrobe finds each garment, labels it (category, colour, pattern, sleeve length, fit…), turns it into a clean catalog image, suggests outfits that work together, picks what to wear today based on the weather and your calendar, and lets you chat with a stylist. Later, it will also show how an outfit looks on you with virtual try-on.

> **Status:** early. Phase 0 (API skeleton) is in place, and Phase 1 (garment understanding) is in progress. Now is a good time to get involved: big design decisions are still open, and you can help shape them.

---

## Why this project exists

Most "AI closet" apps send your photos to a third-party vision API and wrap the answer in a UI. Smart Wardrobe goes the other way:

- **We own the models.** Every step of the pipeline (detection, attributes, catalog image, outfit compatibility, ranking, try-on) is a model we train, evaluate and serve ourselves. Hosted APIs and zero-shot models are only the **baselines** our models have to beat on a golden test set.
- **Privacy by design.** Wardrobe and full-body photos stay on self-hosted models instead of being sent to someone else's API. Consent, encryption and deletion are built in from the start (designed with Quebec's Law 25 in mind).
- **No agent where none is needed.** The core features are fixed pipelines, served as plain endpoints. An LLM with tool calling is used **only** for the open-ended stylist chat, and its tools are those same endpoints.
- **Built in the open, decisions included.** Each phase has notes on what was planned, what was decided and what was learned, and every dataset license is logged. It works as a real product and also as a worked example of end-to-end ML engineering.

The full reasoning is in [docs/ROADMAP.md](docs/ROADMAP.md).

## What it will do

| Feature | How |
|---|---|
| **Garment understanding**: find every item in a photo, with mask, category, subtype and attributes | Instance segmentation + attribute classifier + embeddings |
| **Shop-style catalog image** from a messy photo | V1: mask and clean up. V2: diffusion "try-off" |
| **Outfit suggestions and occasion tags** | Compatibility model (Polyvore-style), occasion classifier |
| **Daily recommendation** from weather, schedule and feedback | Rules first, then a learned ranker |
| **Stylist chat** ("what goes with my green chinos?") | LLM with tool calls to the endpoints above |
| **Virtual try-on** | Diffusion VTON |

## Architecture

```
            ┌────────────── mobile app ──────────────┐
            │                                         │
            ▼                                         ▼
   services/api (FastAPI)  ───── stylist chat ───► LLM (tool calling)
            │                                         │
            │  contract.py (the only ML ↔ backend link)│ tools = same endpoints
            ▼                                         ▼
   ┌─────────────┬───────────────┬──────────────┬─────────────┬──────────┐
   │ /v1/analyze │ catalog image │ compatibility│ recommender │  try-on  │
   └─────────────┴───────────────┴──────────────┴─────────────┴──────────┘
                     one model per endpoint, trained in ml/
```

The backend and the ML only meet in [`services/api/app/ml/contract.py`](services/api/app/ml/contract.py). Until a real model is ready, the API runs against a stub analyzer, so you can work on the backend or the app without a GPU.

## Roadmap

| Phase | Feature | Key metrics | Status |
|---|---|---|---|
| 0 | Foundations: repo, API skeleton, tracking | — | in progress |
| 1 | Garment understanding | mask mAP, per-attribute macro-F1, recall@k | **in progress** |
| 2 | Shop-style catalog image | human preference, FID | planned |
| 3 | Outfits + occasion | compatibility AUC, FITB accuracy, occasion F1 | planned |
| 4 | Daily recommendation | NDCG@k, hit rate, acceptance rate | planned |
| 5 | Stylist chat | task success on scripted eval | planned |
| 6 | Virtual try-on | human eval, FID/KID | planned |

Phase 1 so far: a [v1 garment taxonomy](ml/configs/taxonomy.yaml) (9 detector classes, subtypes and closed-vocabulary attributes), a [Fashionpedia → taxonomy mapping](ml/configs/fashionpedia_map.yaml), loaders and an EDA notebook. Progress notes are in [docs/phase1-garment-understanding.md](docs/phase1-garment-understanding.md).

## Ways to contribute

You don't need to be an ML expert. Some open areas:

- **ML / CV**: baselines (zero-shot detector + FashionCLIP), training the Phase 1 detector, attribute heads, evaluation scripts, model cards.
- **Data**: building the golden test set (flat-lay, hanging and screenshot photos), finding commercially usable datasets for shoe types, menswear, traditional wear and swimwear, which Fashionpedia barely covers.
- **Backend**: FastAPI/SQLModel endpoints, storage, auth, tests.
- **Mobile app**: the client platform is **not decided yet**. Open an issue if you have a strong case.
- **Privacy and infra**: encryption at rest, deletion flows, model serving, monitoring.
- **Docs**: explaining the ML decisions so others can learn from them.

Start with [CONTRIBUTING.md](CONTRIBUTING.md), then open an issue to say what you'd like to pick up.

## Repository layout

| Path | What |
|---|---|
| `ml/` | Taxonomy and dataset configs, data loaders, notebooks, training, evaluation, serving |
| `services/api/` | Backend (FastAPI, SQLModel) |
| `app/` | Mobile client (platform not decided yet) |
| `docs/` | Roadmap, phase notes, decisions and dataset licenses |

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git config core.hooksPath .githooks
```

```bash
cd services/api && uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs. Set `WARDROBE_ANALYZER=http` to use the real model service instead of the stub.

```bash
cd services/api && uv run pytest
```

ML environment and data download steps are in [ml/README.md](ml/README.md) and [ml/data/README.md](ml/data/README.md).

## License

Code and documentation are licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).

Datasets and pretrained weights are **not** covered by this license and keep their own terms. Some are non-commercial or have per-image licenses. Each one is logged in [docs/decisions.md](docs/decisions.md).
