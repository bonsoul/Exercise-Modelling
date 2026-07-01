"""Data loading utilities for the Exercise Data Log dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def load_summary() -> dict:
    """Load the aggregate dataset summary (category/equipment/target totals)."""
    with open(DATA_DIR / "exercises.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_rows() -> pd.DataFrame:
    """Load the per-exercise row-level table used by the explorer."""
    with open(DATA_DIR / "exercises_rows.json", encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_model_results() -> dict:
    """Load model performance / confusion-matrix / heatmap data."""
    with open(DATA_DIR / "model_results.json", encoding="utf-8") as f:
        return json.load(f)


def dict_list_to_df(items: list[dict]) -> pd.DataFrame:
    """Convert a list of {"name": ..., "value": ...} dicts into a DataFrame."""
    return pd.DataFrame(items)


def heatmap_df(model_results: dict) -> pd.DataFrame:
    """Pivot the equipment x category heatmap cells into a wide DataFrame."""
    cells = model_results["heatmap"]["cells"]
    df = pd.DataFrame(cells)
    pivot = df.pivot(index="equipment", columns="category", values="count").fillna(0)
    pivot = pivot.reindex(index=model_results["heatmap"]["equipment_list"],
                           columns=model_results["heatmap"]["category_list"])
    return pivot


def confusion_df(model_results: dict) -> pd.DataFrame:
    """Pivot the confusion-matrix cells into a wide DataFrame."""
    classes = model_results["confusion"]["classes"]
    cells = model_results["confusion"]["cells"]
    df = pd.DataFrame(cells)
    pivot = df.pivot(index="true", columns="pred", values="count").fillna(0)
    pivot = pivot.reindex(index=classes, columns=classes, fill_value=0)
    return pivot
