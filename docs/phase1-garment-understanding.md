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
- referred the Fashionpedia dataset, to get an idea of what kind of labels are available and categories are covered.
- picked the important items what made sense to me (a user would want to pick), and organized them in to proper categories.
- Decided to split the general category Top into tops and shirts.
- and decided to have the general bottom into pants and skirts.
- And have a separate category for dress - to cover all the single piece dresses

    Taxonomy v1: design decisions

    1. Principle: the taxonomy is based on what the app needs, not on any dataset. Datasets are mapped into it (ml/configs/fashionpedia_map.yaml), matched by label name, not id.

    2. Three levels per garment
        - Category (9 classes): what the detector predicts. Kept broad on purpose (shirt, top, skirt, pants, dress, outerwear, shoes, accessory, bag). Fewer classes means more examples each, and less confusion between similar items like blazer and jacket.
        - Subtype: the specific item inside a category (jeans, hoodie, blazer, gown, pencil_skirt, watch). Predicted from the garment crop, not by the detector. It gets a default from the source category (shorts → pants/shorts) and a more specific value from Fashionpedia nicknames (jacket + blazer → blazer).
        - Attributes: details about the item: pattern, fit, length, silhouette, waist_rise, sleeve_length, neckline, color, material, formality, warmth. Each attribute has a closed list of values and an applies_to list of categories (no sleeve length on shoes).

    3. Key choices
        - Merged items that look alike or that the data doesn't separate: jeans and shorts are pants subtypes, jumpsuit is a dress subtype, and small items (belt, watch, glasses, socks…) are accessory subtypes.
        - Sweater, cardigan, hoodie and sweatshirt are all tops. outerwear only holds things worn over everything else (jackets, coats, blazers).
        - Outfit roles (base_top, mid_top, bottom, one_piece, outer, shoes, accessory, bag) are derived by rule from category + subtype, so the recommender can layer items correctly. They're not predicted.
        - Fashionpedia's fine-grained labels are collapsed into smaller value sets. 294 attributes → 103 mapped, and the other 161 are listed in dropped_attributes. Garment parts, closures and decorations are dropped as objects.
        - Sleeve length and neckline come from part objects. Fashionpedia labels them on separate sleeve and neckline objects, which get matched to their garment by box overlap.

    4. Known gaps
        - No colour, formality or warmth labels, and almost no material labels. Colour will be computed from mask pixels. Material, formality and warmth need zero-shot models, rules or my own labels (still being researched).
        - No shoe types (sneaker, boot, heel…). This matters a lot for outfits and needs another source.
        - Photo domain: Fashionpedia is worn, street-style photos, but users will upload flat-lay, hanging and screenshot photos. The golden set has to measure this. 
        - Coverage: no suits, swimwear, sleepwear or traditional wear. The data is skewed towards women's fashion.

    5. Versioning: the taxonomy has version: 1. Stored wardrobe items should record the version they were labelled with, so later changes can be migrated.



**What I learned:**
- I searched about the common terms used in fashion, got to know about the categories
- There is also other datasets like ModaNet and DeepFashion2, but they are not as diverse on the categories and attributes like the Fashionpedia dataset
- so went with referring to Fashionpedia set and to use it in the project
- im understanding the attributes is going to be useful while making creating recommendations for style and outfits.

---

## 1.2 Data acquisition + EDA

**Plan:** get the candidate datasets, log each license in `docs/decisions.md`, and explore them after mapping to my taxonomy.

| Dataset | Why | Watch out |
|---|---|---|
| Fashionpedia | ~48k images, masks and fine-grained attributes | mostly worn/street photos |
| DeepFashion2 | consumer vs shop image pairs, useful for the photo vs screenshot gap | non-commercial; access by request form |
| ModaNet | extra street photos with polygons | non-commercial |

For now decided to not use these 2 sets, as they are non-commercial

Questions my EDA has to answer:
1. Class distribution after mapping. Which classes are rare, and which have no data at all?
2. Objects per image, and mask area as a share of the image (small accessories behave differently).
3. Image resolution and aspect ratios, to choose the training image size.
4. Attribute coverage: how many instances actually have each attribute labelled?
5. How different are these images from what users will upload?

**Findings:**
- Some of our attributes like color, material, formality and warmth are not in fashionpedia
- have to figure some workaround to fill this gap (as these are some imp. attributes for our app goal)
- Fashionpedia has only a few material labels (jeans → denim, suede, leather types), not enough to train a material model.
- Also the Sleeve length is labelled on separate sleeve objects, not on the garments itself.
  
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
