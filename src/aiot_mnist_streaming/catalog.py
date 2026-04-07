from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class CatalogBundle:
    catalog: pd.DataFrame
    model_lookup: dict[str, dict]
    all_model_names: list[str]
    strong_models: list[str]
    preview_model: str


def build_catalog_bundle(model_catalog: pd.DataFrame, preview_model: str) -> CatalogBundle:
    catalog = model_catalog.sort_values(["edge_mean_ms", "accuracy"]).reset_index(drop=True).copy()
    model_lookup = {row["model"]: row.to_dict() for _, row in catalog.iterrows()}
    all_model_names = list(catalog["model"].astype(str))
    strong_models = [name for name in all_model_names if name.startswith("full")]
    resolved_preview = preview_model if preview_model in all_model_names else ("tiny_quant" if "tiny_quant" in all_model_names else all_model_names[0])
    return CatalogBundle(
        catalog=catalog,
        model_lookup=model_lookup,
        all_model_names=all_model_names,
        strong_models=strong_models,
        preview_model=resolved_preview,
    )
