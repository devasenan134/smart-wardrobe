# Decisions and licenses

Every dataset, pretrained weight and major design choice I use goes here. I track **commercial use** because I may monetize the app: a non-commercial asset is fine for learning and the portfolio, but I'll have to replace it before charging money.

## Datasets and weights

| Asset | Used for | License | Commercial OK? | Checked |
|---|---|---|---|---|
| Fashionpedia dataset | Phase 1 seg + attributes | CC BY 4.0 (annotations); images are Flickr, each with its own license | verify per image | |

## Design decisions

- 2026-09-25 — I serve each ML model as an endpoint and use the LLM only for chat, not as an agent orchestrating everything. Reason: the core features have fixed inputs and outputs, and owning the models is the point of the project. Full reasoning in [ROADMAP.md](ROADMAP.md#why-i-own-the-models-build-vs-buy).
