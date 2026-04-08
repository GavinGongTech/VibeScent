"""
VibeScent — Merge & Deduplicate Fragrance Datasets
Combines Parfumo, rdemarqui, Rawanalqarni, Fragrantica-clean, and Ayush
into a single unified CSV with only the fields the pipeline needs.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────
def clean(text):
    """Lowercase, strip whitespace, collapse spaces."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())

def make_key(name, brand):
    """Dedup key from name + brand."""
    return clean(name) + " | " + clean(brand)

# ── schema we want ───────────────────────────────────────────────────
UNIFIED_COLS = [
    "name", "brand", "top_notes", "middle_notes", "base_notes",
    "main_accords", "gender", "concentration", "rating_value",
    "rating_count", "longevity", "sillage", "year", "category",
    "source"   # track provenance
]

def empty_unified():
    return {c: np.nan for c in UNIFIED_COLS}

# ── loaders: each returns a list[dict] in unified schema ─────────────

def load_parfumo(path):
    """Parfumo CSV from TidyTuesday."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        row = empty_unified()
        row["name"]          = r.get("Name")
        row["brand"]         = r.get("Brand")
        row["top_notes"]     = r.get("Top_Notes")
        row["middle_notes"]  = r.get("Middle_Notes")
        row["base_notes"]    = r.get("Base_Notes")
        row["main_accords"]  = r.get("Main_Accords")
        row["concentration"] = r.get("Concentration")
        row["rating_value"]  = r.get("Rating_Value")
        row["rating_count"]  = r.get("Rating_Count")
        row["year"]          = r.get("Release_Year")
        row["source"]        = "parfumo"
        rows.append(row)
    return rows

def load_rdemarqui(path):
    """rdemarqui perfume_database CSV."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        row = empty_unified()
        row["name"]         = r.get("perfume")
        row["brand"]        = r.get("brand")
        # single "Notes" column — put in top_notes, leave mid/base empty
        row["top_notes"]    = r.get("Notes")
        row["main_accords"] = r.get("Main_accords")
        row["longevity"]    = r.get("Longevity")
        row["sillage"]      = r.get("Sillage")
        row["year"]         = r.get("Launch_year")
        row["source"]       = "rdemarqui"
        rows.append(row)
    return rows

def load_rawanalqarni(path):
    """Rawanalqarni Perfume_Dataaset CSV."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        row = empty_unified()
        row["name"]          = r.get("Name")
        row["brand"]         = r.get("Brand")
        row["top_notes"]     = r.get("Top_note")
        row["middle_notes"]  = r.get("Middle_note")
        row["base_notes"]    = r.get("Base_note")
        row["gender"]        = r.get("Gender")
        row["concentration"] = r.get("Concentration")
        row["rating_value"]  = r.get("Rate")
        row["rating_count"]  = r.get("Rating_count")
        row["year"]          = r.get("Year")
        row["category"]      = r.get("Fragrance_Family")
        row["source"]        = "rawanalqarni"
        rows.append(row)
    return rows

def load_fragrantica_clean(path):
    """Fragrantica cleaned dataset from Kaggle (olgagmiufana1)."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        row = empty_unified()
        row["name"]          = r.get("Perfume")
        row["brand"]         = r.get("Brand")
        row["top_notes"]     = r.get("Top Notes")
        row["middle_notes"]  = r.get("Middle Notes")
        row["base_notes"]    = r.get("Base Notes")
        # combine up to 5 accord columns
        accords = [str(r.get(f"Main Accord {i}", "")) for i in range(1, 6)]
        accords = [a for a in accords if a and a != "nan"]
        row["main_accords"]  = ", ".join(accords) if accords else np.nan
        row["gender"]        = r.get("Gender")
        row["rating_value"]  = r.get("Rating Value")
        row["rating_count"]  = r.get("Rating Count")
        row["year"]          = r.get("Year")
        row["source"]        = "fragrantica_clean"
        rows.append(row)
    return rows

def load_ayush(path):
    """Ayush perfume dataset from Kaggle."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        row = empty_unified()
        row["name"]          = r.get("perfume")
        row["brand"]         = r.get("brand")
        row["concentration"] = r.get("type")
        row["gender"]        = r.get("target_audience")
        row["longevity"]     = r.get("longevity")
        row["category"]      = r.get("category")
        row["source"]        = "ayush"
        rows.append(row)
    return rows

# ── merge logic ──────────────────────────────────────────────────────

def merge_rows(group_df):
    """
    Given multiple rows for the same fragrance (from different sources),
    produce one merged row by picking the first non-null value per column.
    Priority order is controlled by SOURCE_PRIORITY.
    """
    SOURCE_PRIORITY = [
        "fragrantica_clean",  # best structured fields
        "parfumo",            # good notes split
        "rawanalqarni",       # has gender, category
        "rdemarqui",          # has longevity/sillage
        "ayush",              # has category/longevity
    ]
    # sort by priority
    priority_map = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    group_df = group_df.copy()
    group_df["_priority"] = group_df["source"].map(
        lambda s: priority_map.get(s, 99)
    )
    group_df = group_df.sort_values("_priority")

    merged = empty_unified()
    sources_used = []
    for _, r in group_df.iterrows():
        sources_used.append(r["source"])
        for col in UNIFIED_COLS:
            if col == "source":
                continue
            if pd.isna(merged[col]) and pd.notna(r.get(col)):
                merged[col] = r[col]
    merged["source"] = " + ".join(dict.fromkeys(sources_used))  # deduped order
    return merged


def build_unified_dataset(dataset_paths: dict) -> pd.DataFrame:
    """
    dataset_paths: dict mapping loader name -> local file path, e.g.
        {"parfumo": "data/parfumo.csv", "rdemarqui": "data/rde.csv", ...}
    Only include keys for datasets you actually have downloaded.
    """
    LOADERS = {
        "parfumo":           load_parfumo,
        "rdemarqui":         load_rdemarqui,
        "rawanalqarni":      load_rawanalqarni,
        "fragrantica_clean": load_fragrantica_clean,
        "ayush":             load_ayush,
    }

    all_rows = []
    for key, path in dataset_paths.items():
        if key not in LOADERS:
            print(f"⚠ No loader for '{key}', skipping")
            continue
        print(f"Loading {key} from {path} ...")
        rows = LOADERS[key](path)
        print(f"  → {len(rows)} rows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=UNIFIED_COLS)
    print(f"\nTotal raw rows: {len(df)}")

    # ── dedup key ────────────────────────────────────────────────────
    df["_key"] = df.apply(lambda r: make_key(r["name"], r["brand"]), axis=1)

    # drop rows where name is missing entirely
    df = df[df["_key"].str.strip() != "|"].copy()

    # ── group & merge ────────────────────────────────────────────────
    merged_rows = []
    for key, group in df.groupby("_key"):
        merged_rows.append(merge_rows(group))

    unified = pd.DataFrame(merged_rows, columns=UNIFIED_COLS)

    # ── quality filter: keep only rows with at least *some* notes ────
    has_notes = unified[["top_notes", "middle_notes", "base_notes"]].notna().any(axis=1)
    print(f"Rows with at least one notes field: {has_notes.sum()} / {len(unified)}")
    unified_with_notes = unified[has_notes].copy()

    # ── build the text field for embedding ───────────────────────────
    def build_text(r):
        parts = []
        for col in ["top_notes", "middle_notes", "base_notes"]:
            if pd.notna(r[col]):
                parts.append(str(r[col]))
        text = " | ".join(parts)
        if pd.notna(r["main_accords"]):
            text += f" | Accords: {r['main_accords']}"
        if pd.notna(r["gender"]):
            text += f" | {r['gender']}"
        if pd.notna(r["concentration"]):
            text += f" | {r['concentration']}"
        return text

    unified_with_notes["embedding_text"] = unified_with_notes.apply(build_text, axis=1)

    unified_with_notes = unified_with_notes.reset_index(drop=True)
    return unified_with_notes


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # *** EDIT THESE PATHS to wherever you downloaded each CSV ***
    dataset_paths = {
        "parfumo":           "data/parfumo_data_clean.csv",
        "rdemarqui":         "data/perfume_database.csv",
        "rawanalqarni":      "data/Perfume_Dataaset.csv",
        "fragrantica_clean": "data/fragrantica_clean.csv",
        "ayush":             "data/perfume_dataset.csv",
    }

    # only load what you actually have on disk
    dataset_paths = {k: v for k, v in dataset_paths.items() if Path(v).exists()}
    if not dataset_paths:
        print("No datasets found! Update the paths in __main__.")
    else:
        df = build_unified_dataset(dataset_paths)
        out = "data/vibescent_unified.csv"
        df.to_csv(out, index=False)
        print(f"\n✅ Saved {len(df)} fragrances → {out}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample embedding_text:\n{df['embedding_text'].iloc[0]}")