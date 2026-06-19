"""Offline scoring of prediction CSVs against the labeled columns in sample_claims.csv."""

from __future__ import annotations

import pandas as pd

EXACT_MATCH_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "object_part",
    "valid_image",
    "severity",
]

SET_OVERLAP_FIELDS = ["risk_flags", "supporting_image_ids"]


def _as_set(value: str) -> set[str]:
    value = (value or "none").strip()
    return {v.strip() for v in value.split(";") if v.strip()}


def _set_overlap_score(predicted: str, expected: str) -> float:
    pred_set, exp_set = _as_set(predicted), _as_set(expected)
    if not pred_set and not exp_set:
        return 1.0
    union = pred_set | exp_set
    if not union:
        return 1.0
    return len(pred_set & exp_set) / len(union)


def score_predictions(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    merged = predictions.merge(
        labels, on=["user_id", "image_paths", "claim_object"], suffixes=("_pred", "_true")
    )
    if merged.empty:
        raise ValueError("No matching rows between predictions and labels (check user_id/image_paths/claim_object).")

    field_scores: dict[str, float] = {}
    for field in EXACT_MATCH_FIELDS:
        matches = merged[f"{field}_pred"].astype(str).str.lower() == merged[f"{field}_true"].astype(str).str.lower()
        field_scores[field] = round(matches.mean(), 4)

    for field in SET_OVERLAP_FIELDS:
        scores = [
            _set_overlap_score(pred, true)
            for pred, true in zip(merged[f"{field}_pred"], merged[f"{field}_true"])
        ]
        field_scores[field] = round(sum(scores) / len(scores), 4)

    confusion = (
        merged.groupby(["claim_status_true", "claim_status_pred"]).size().rename("count").reset_index()
    )

    return {
        "num_rows": len(merged),
        "field_scores": field_scores,
        "claim_status_confusion": confusion.to_dict(orient="records"),
    }
