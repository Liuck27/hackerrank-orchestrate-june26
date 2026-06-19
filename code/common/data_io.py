"""CSV loading and claim-row helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ClaimRow:
    row_index: int
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str

    @property
    def image_abs_paths(self) -> list[Path]:
        return [REPO_ROOT / "dataset" / p.strip() for p in self.image_paths.split(";") if p.strip()]

    @property
    def image_ids(self) -> list[str]:
        return [Path(p.strip()).stem for p in self.image_paths.split(";") if p.strip()]


def load_claims(csv_path: Path, limit: int | None = None) -> list[ClaimRow]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if limit is not None:
        df = df.head(limit)
    rows = []
    for i, record in enumerate(df.to_dict(orient="records")):
        rows.append(
            ClaimRow(
                row_index=i,
                user_id=record["user_id"],
                image_paths=record["image_paths"],
                user_claim=record["user_claim"],
                claim_object=record["claim_object"],
            )
        )
    return rows


def load_user_history(csv_path: Path = REPO_ROOT / "dataset" / "user_history.csv") -> dict[str, dict]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    return {record["user_id"]: record for record in df.to_dict(orient="records")}


def load_evidence_requirements(
    csv_path: Path = REPO_ROOT / "dataset" / "evidence_requirements.csv",
) -> list[dict]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    return df.to_dict(orient="records")


def evidence_requirements_for(claim_object: str, requirements: list[dict]) -> list[dict]:
    return [r for r in requirements if r["claim_object"] in (claim_object, "all")]


def user_history_text(user_id: str, history: dict[str, dict]) -> str:
    record = history.get(user_id)
    if record is None:
        return "No prior claim history available for this user."
    return (
        f"past_claim_count={record['past_claim_count']}, "
        f"accept_claim={record['accept_claim']}, "
        f"manual_review_claim={record['manual_review_claim']}, "
        f"rejected_claim={record['rejected_claim']}, "
        f"last_90_days_claim_count={record['last_90_days_claim_count']}, "
        f"history_flags={record['history_flags']}, "
        f"history_summary={record['history_summary']}"
    )


def write_output_csv(rows: list[dict], output_path: Path) -> None:
    from common.schema import OUTPUT_COLUMNS

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
