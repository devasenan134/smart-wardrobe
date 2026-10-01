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

**Findings:** (Fashionpedia train split, after mapping to my taxonomy. Notebook: `ml/notebooks/fashionpedia_eda.ipynb`)

After mapping, 333k Fashionpedia objects become 163k garments in 45.6k images. The rest were garment parts and decorations I dropped.

**1. Class distribution**
- All 9 of my categories have plenty of data. The smallest is skirt with 5k garments, the largest is shoes with 46k.
- But it's imbalanced: shoes are 28.5% of all garments, shirts 3.8% and skirts 3.1% (about 9:1). → I should try class rebalancing in 1.5, otherwise the model will be best at shoes and weakest at shirts and skirts.
- At subtype level, some of my subtypes have no data at all: shirt vs blouse, trousers, chinos, joggers, sweatshirt, and every shoe type (sneakers, formal shoes…). → A model trained on Fashionpedia can never predict these. They have to come from the golden set, a zero-shot model or user corrections.
- Some subtypes are very rare: cargo shorts 25, track pants 33, wrap skirt 53, cargo pants 66. → Probably too few to learn well.

**2. Objects per image and object size**
- Images have 3–4 garments on average (median 3, max 20), because these are full outfits.
- Shoes and accessories are tiny: their median mask covers only 0.3% of the image, and about 70% of shoes and 60% of accessories cover less than 0.5%. Dresses are the biggest (median 12.5%).
- → Small objects will be the hardest part for the detector. Shoes and accessories will probably have the lowest mAP.

**3. Image resolution**
- Every image has its longest side at 1024 px. Most are 682×1024, and 80% are portrait.
- A shoe's box is about 56×86 px at full size. At YOLO's default 640, that shrinks to about 35×54 px.
- → I should compare imgsz 640 vs 1024 in 1.5 because of the small objects. 1024 is slower and needs more GPU memory (8 GB), so batch size will have to go down.

**4. Attribute coverage** (share of garments with a label, only counting the categories each attribute applies to)

| attribute | coverage | notes |
|---|---|---|
| subtype | 89% | inflated: shoes, dresses and accessories get it from the category default. Shirts 0%, pants 58%, tops 68% |
| length | 85% | |
| waist_rise | 66% | |
| neckline | 65% | from neckline objects |
| pattern | 63% | 87% of labels are `solid`. Camouflage only 141 |
| fit | 61% | baggy only 72, oversized 659 |
| silhouette | 59% | wide_leg only 869 |
| sleeve_length | 53% | from sleeve objects. `sleeveless` only 167 |
| material | 2% | only denim (from jeans), leather and suede |
| colour, formality, warmth | 0% | not in Fashionpedia at all |

- → Pattern is very imbalanced (mostly solid), so I need to look at per-value F1, not just accuracy, in 1.6.
- → Colour, material, formality and warmth have no usable labels. Colour I can compute from the mask pixels. The others need zero-shot (FashionCLIP), rules or my own labels.
- → Sleeve matching by box overlap misses layered outfits: 40% of shirts and 46% of tops get no sleeves matched, often because the jacket's box grabs the sleeves. Sleeve length numbers are an undercount. Using masks instead of boxes could fix this later.

**5. Difference from what users will upload**
- Fashionpedia photos are professional or street-style: people wearing full outfits, good lighting, clean backgrounds.
- My users will upload single items lying on a bed, hanging on a hanger, mirror selfies in a normal room, and shop screenshots.
- → Big domain gap. The golden set has to be mostly these user-style photos, so I can measure how badly the model drops on them. Later I may need flat-lay or hanging training data too (e.g. other images from the Grigorev clothing dataset, never the ones in my golden set), or augmentations.

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

Experiments to try: image size, class rebalancing, adding DeepFashion2 shop images, augmentation for phone-photo conditions, and class granularity: our 9 coarse classes vs ~24 Fashionpedia-level classes (jacket, coat, sweater, shorts…) grouped back into our 9 after detection. Tests whether finer detection beats predicting the fine type as `subtype` from the crop.

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
