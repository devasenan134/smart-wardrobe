# Roadmap

## Architecture

Each ML capability is a model served behind its own endpoint. The app calls the endpoints directly. An LLM with tool calling is used **only** for the stylist chat (and possibly calendar-event parsing).

### Why I own the models (build vs. buy)

Every step in the core pipeline (photo → garments → attributes → catalog image → outfits) can either be **bought** or **built**. Buying means calling a hosted vision LLM or API with a fixed prompt and a structured output. Building means training and serving my own model.

**When buying wins:** early on and at low volume. There's almost no fixed cost, it launches in days, and the quality is decent on day one. A company building an MVP would buy most steps to find out whether people want the product at all.

**When building wins:** at high, ongoing volume, and on steps that set the product apart. Once there are millions of uploads, embeddings for every item and recommendations for every user every day, the per-call bill costs more than the engineers and GPUs. A fine-tuned model is also better at narrow, pixel-level work (masks, tight boxes, sleeve length) than a general vision LLM. A company switches when the numbers say so: API cost vs. own-model cost, and quality measured on a golden test set.

**Why I'm building anyway:**
- Learning ML engineering end to end is the goal of this project, and buying would skip exactly that work.
- Privacy: users' photos, including full-body ones for try-on, stay on models I host instead of going to a third party (Quebec Law 25).
- It still follows the company logic: each bought option becomes the **baseline** my model has to beat. In Phase 1 that's the zero-shot detector + FashionCLIP. Where an LLM does well (for example occasion classification), I'll use it as a baseline too.

The trade-off I'm accepting: much more upfront work (data, labelling, training, serving) than prompting a model.

### Not an agent

The core features are fixed pipelines, so there's nothing for an LLM to plan. It would only add cost, latency and non-determinism. The LLM with tool calling is used only for the stylist chat, which is open-ended, and its tools are my endpoints.

## Phases

| Phase | Feature | ML I build | Key metrics | Status |
|---|---|---|---|---|
| 0 | Foundations | Repo, API skeleton, tracking | — | in progress |
| 1 | Garment understanding | Detection/segmentation, attribute classifier, embeddings | mask mAP, per-attribute macro-F1, retrieval recall@k | next |
| 2 | Shop-style catalog image | V1: mask + cleanup; V2: diffusion "try-off" | human preference, FID | |
| 3 | Outfits + occasion | Compatibility model (Polyvore), occasion classifier | compatibility AUC, FITB accuracy, occasion F1 | |
| 4 | Daily recommendation | Rules → learned ranker from feedback logs | NDCG@k, hit rate, online acceptance rate | |
| 5 | Stylist chat | LLM tool use over phase 1–4 endpoints | task success on scripted eval | |
| 6 | Virtual try-on | Diffusion VTON (cloud GPU) | human eval, FID/KID | |

Cross-cutting: experiment tracking, a model card per model, a golden test set built from real user-style photos, latency budgets, monitoring and drift, privacy (consent, encryption, deletion — Quebec Law 25).

Hardware: RTX 4060 Laptop with 8 GB is enough for phases 1, 3 and 4. Phases 2-V2 and 6 need cloud GPU time.
