# Smart Wardrobe

Photograph your clothes and get a clean catalog, outfit combinations, daily suggestions based on weather and your schedule, a stylist chat, and virtual try-on.

## How it works

Each ML capability (garment detection and attributes, catalog images, outfit compatibility, recommendations, try-on) is a model served behind its own endpoint. The backend calls those endpoints directly. An LLM with tool calling is used only for the stylist chat, with the endpoints as its tools.

The only link between the backend and the ML is the contract in [`services/api/app/ml/contract.py`](services/api/app/ml/contract.py).

## Repository layout

| Path | What |
|---|---|
| `ml/` | Data, training, evaluation, model serving |
| `services/api/` | Backend (FastAPI, SQLModel) |
| `app/` | Mobile client (platform not decided yet) |
| `docs/` | Roadmap, phase notes, decisions and licenses |

## Run the API

```bash
cd services/api && uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs. Set `WARDROBE_ANALYZER=http` to use the real model service instead of the stub.

## Setup after cloning

Enable the repository's git hooks once:

```bash
git config core.hooksPath .githooks
```

## License

Copyright (c) 2026 Devasenan Murugan. All rights reserved. See [LICENSE](LICENSE).
