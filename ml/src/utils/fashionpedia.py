"""Load Fashionpedia annotations and map them into our taxonomy.

build_garments() returns one row per garment (our 9 categories only), with
subtype, outfit role and every mapped attribute as its own column.
"""

import json
from collections import Counter, defaultdict

import pandas as pd
import yaml

# Attributes that Fashionpedia labels on separate part objects, not on the garment.
# part category name -> our attribute (same name as the mapping section)
PART_ATTRIBUTES = {"sleeve": "sleeve_length", "neckline": "neckline"}

# A part belongs to a garment if at least this share of the part's box lies inside the garment's box.
MIN_PART_OVERLAP = 0.5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_images(raw):
    """One row per image: id, file name, size."""
    df = pd.DataFrame(raw["images"])[["id", "file_name", "width", "height"]]
    df = df.rename(columns={"id": "image_id"})
    df["aspect"] = df["width"] / df["height"]
    return df


def build_garments(raw, mapping, taxonomy):
    """One row per garment, mapped into our taxonomy."""
    cat_name = {c["id"]: c["name"] for c in raw["categories"]}
    attr_name = {a["id"]: a["name"] for a in raw["attributes"]}
    images = {img["id"]: img for img in raw["images"]}
    applies_to = {k: set(v["applies_to"]) for k, v in taxonomy["attributes"].items()}

    rows, parts = [], []
    for ann in raw["annotations"]:
        fp_cat = cat_name[ann["category_id"]]
        names = [attr_name[i] for i in ann["attribute_ids"]]

        if fp_cat in PART_ATTRIBUTES:
            attr = PART_ATTRIBUTES[fp_cat]
            values = [mapping[attr][n] for n in names if n in mapping[attr]]
            parts.append({"image_id": ann["image_id"], "attr": attr, "bbox": ann["bbox"],
                          "value": values[0] if values else None})
            continue

        category = mapping["categories"][fp_cat]
        if category == "drop":
            continue

        img = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        row = {
            "image_id": ann["image_id"],
            "ann_id": ann["id"],
            "fp_category": fp_cat,
            "category": category,
            "subtype": mapping["category_subtypes"].get(fp_cat),  # default, a nickname can override
            "area": ann["area"],
            "area_frac": ann["area"] / (img["width"] * img["height"]),
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "img_w": img["width"],
            "img_h": img["height"],
        }
        for name in names:
            mapped = mapping["attributes"].get(name)  # None for dropped / part attributes
            if not mapped:
                continue
            for attr, value in mapped.items():
                if category not in applies_to[attr]:
                    continue
                if attr == "subtype" or attr not in row:  # nickname wins over the default subtype
                    row[attr] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _assign_parts(df, parts, applies_to)
    df["outfit_role"] = _outfit_roles(df, taxonomy["outfit_roles"])
    return df


def _overlap(part, garment):
    """Share of the part's box that lies inside the garment's box."""
    px, py, pw, ph = part
    gx, gy, gw, gh = garment
    iw = max(0, min(px + pw, gx + gw) - max(px, gx))
    ih = max(0, min(py + ph, gy + gh) - max(py, gy))
    return iw * ih / (pw * ph) if pw * ph else 0


def _assign_parts(df, parts, applies_to):
    """Match each sleeve / neckline to the garment it sits on and copy its value over.

    Also adds n_sleeves and n_necklines: how many part objects were matched to each garment.
    """
    by_image = defaultdict(list)
    for i, g in enumerate(df[["image_id", "category", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].itertuples(index=False)):
        by_image[g.image_id].append((i, g.category, (g.bbox_x, g.bbox_y, g.bbox_w, g.bbox_h)))

    votes = {attr: defaultdict(Counter) for attr in PART_ATTRIBUTES.values()}
    counts = {attr: Counter() for attr in PART_ATTRIBUTES.values()}
    for p in parts:
        candidates = [(_overlap(p["bbox"], box), i) for i, cat, box in by_image[p["image_id"]]
                      if cat in applies_to[p["attr"]]]
        if not candidates:
            continue
        best, i = max(candidates)
        if best < MIN_PART_OVERLAP:
            continue
        counts[p["attr"]][i] += 1
        if p["value"]:
            votes[p["attr"]][i][p["value"]] += 1

    for attr in PART_ATTRIBUTES.values():
        # two sleeves on one shirt usually agree; if not, take the most common value
        df[attr] = pd.Series({i: c.most_common(1)[0][0] for i, c in votes[attr].items()}, dtype=object)
    df["n_sleeves"] = pd.Series(counts["sleeve_length"]).reindex(df.index, fill_value=0)
    df["n_necklines"] = pd.Series(counts["neckline"]).reindex(df.index, fill_value=0)
    return df


def _outfit_roles(df, roles):
    by_cat = df["category"].map(roles["by_category"])
    by_sub = df["subtype"].map(roles["by_subtype"])
    return by_sub.fillna(by_cat)
