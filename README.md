# Smart Wardrobe

Photograph your clothes and get a clean catalog, outfit combinations, daily suggestions based on weather and your schedule, a stylist chat, and virtual try-on.

| Path | What | Owner |
|---|---|---|
| `ml/` | Data, training, evaluation, model serving | Devadas |
| `services/api/` | App backend (FastAPI, SQLModel) | Devadas + Claude |
| `app/` | Mobile client (platform not decided yet) | Claude |
| `docs/` | Roadmap, phase briefs, decisions and licenses | both |

The only link between app and ML is the contract in [`services/api/app/ml/contract.py`](services/api/app/ml/contract.py).

## Run the API

```bash
cd services/api && uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs. Set `WARDROBE_ANALYZER=http` to use the real model service instead of the stub.

## Git hooks

After cloning, run `git config core.hooksPath .githooks` once. The hooks keep Claude attribution out of any commit that touches `ml/`: `commit-msg` strips it, and `pre-push` blocks it if it slipped through.
