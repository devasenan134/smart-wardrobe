# Phase 1 — Garment understanding

**Goal:** given a photo (worn, hanging, on a bed, or an online-store screenshot), return every garment in it with a category, a mask and attributes, served behind `POST /v1/analyze` as defined in `services/api/app/ml/contract.py`.

**Done when:** golden-set metrics beat the zero-shot baseline, a model card exists, and a photo uploaded through the API gets real labels.

Each step below has a plan, then the decisions I made and what I learned while doing it.

---

## 1.1 Taxonomy

**Plan:** decide what the app needs before looking at any dataset. The taxonomy lives in `ml/configs/taxonomy.yaml`, and every dataset gets mapped *into* it, not the other way round.

Starting ideas:
- Categories: top, shirt, t-shirt, sweater, outerwear, dress, trousers, jeans, shorts, skirt, shoes, bag, hat, accessory (belt, watch, jewellery, scarf)
- Attributes: colour (primary + secondary), pattern, sleeve length, fit, material, formality (1–5), season warmth (1–5)

**Decisions:**

**What I learned:**

---

## 1.2 Data acquisition + EDA

**Plan:** get the candidate datasets, log each license in `docs/decisions.md`, and explore them after mapping to my taxonomy.

| Dataset | Why | Watch out |
|---|---|---|
| Fashionpedia | ~48k images, masks and fine-grained attributes | mostly worn/street photos |
| DeepFashion2 | consumer vs shop image pairs, useful for the photo vs screenshot gap | non-commercial; access by request form |
| ModaNet | extra street photos with polygons | non-commercial |

Questions my EDA has to answer:
1. Class distribution after mapping. Which classes are rare, and which have no data at all?
2. Objects per image, and mask area as a share of the image (small accessories behave differently).
3. Image resolution and aspect ratios, to choose the training image size.
4. Attribute coverage: how many instances actually have each attribute labelled?
5. How different are these images from what users will upload?

**Findings:**

---

## 1.3 Golden test set

**Plan:** 150–300 images that look like real uploads: my own clothes hanging, on a bed, worn in mirror selfies, plus screenshots of real order pages. Labelled with my taxonomy in Label Studio or CVAT. Never trained on. This is the set that decides whether a model ships.

**Decisions:**

**Label stats:**

---

## 1.4 Zero-shot baseline

**Plan:** measure what I get without training: Grounding DINO or YOLO-World + SAM 2 for detection and masks, FashionCLIP for category and attributes on crops. Score it on the golden set. Every trained model has to beat it.

**Results:**

---

## 1.5 Fine-tune detection + segmentation

**Plan:** YOLO11-seg or YOLOv8-seg (s or m) on Fashionpedia mapped to my taxonomy, imgsz 640, batch 8–16 with AMP on the 8 GB GPU. Track every run (config, data version, metrics, sample predictions). Report mask mAP50-95 overall and per class, on Fashionpedia val and on the golden set.

Experiments to try: image size, class rebalancing, adding DeepFashion2 shop images, augmentation for phone-photo conditions.

**Experiments:**

| Run | Change | mAP50-95 (val) | mAP50-95 (golden) | Notes |
|---|---|---|---|---|

**Error analysis:**

---

## 1.6 Attribute model + embeddings

**Plan:** crop each garment with its mask; one shared backbone with several heads (colour, pattern, sleeve…), compared against fine-tuning FashionCLIP or SigLIP. Metric: macro-F1 per attribute. Keep the embedding vector for search, deduplication and outfit scoring in later phases.

**Results:**

**What I learned:**

---

## 1.7 Serving

**Plan:** `ml/serve/` FastAPI app exposing `POST /v1/analyze` that returns `AnalyzeResponse`. Export to ONNX or TensorRT and measure p50/p95 latency. Then switch the API to `WARDROBE_ANALYZER=http`.

**Latency:**

**What I learned:**
