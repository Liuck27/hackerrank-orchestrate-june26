"""Output schema and allowed enum values, mirroring problem_statement.md exactly."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClaimObject = Literal["car", "laptop", "package"]

ClaimStatus = Literal["supported", "contradicted", "not_enough_information"]

IssueType = Literal[
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
]

CarObjectPart = Literal[
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
]

LaptopObjectPart = Literal[
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
]

PackageObjectPart = Literal[
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
]

ObjectPart = Literal[
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "body",
    "unknown",
]

RiskFlag = Literal[
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]

Severity = Literal["none", "low", "medium", "high", "unknown"]

OBJECT_PARTS_BY_CLAIM_OBJECT: dict[str, list[str]] = {
    "car": list(CarObjectPart.__args__),
    "laptop": list(LaptopObjectPart.__args__),
    "package": list(PackageObjectPart.__args__),
}

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


class ImageObservation(BaseModel):
    """Per-image visual extraction produced by Stage A of the two-stage strategy."""

    image_id: str
    issue_type: IssueType
    object_part: ObjectPart
    image_quality_ok: bool = Field(
        description="False if the image is too blurry, dark, cropped, or obstructed to evaluate."
    )
    shows_claimed_object: bool = Field(
        description="False if the image does not show the claimed object type at all (wrong_object)."
    )
    observation_notes: str = Field(description="One short sentence describing what is visible.")


class ClaimDecision(BaseModel):
    """Final structured decision matching the required output schema."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[RiskFlag]
    issue_type: IssueType
    object_part: ObjectPart
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: Severity

    def to_output_row(self, user_id: str, image_paths: str, user_claim: str, claim_object: str) -> dict:
        risk_flags = self.risk_flags or ["none"]
        supporting_ids = self.supporting_image_ids or ["none"]
        return {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": str(self.evidence_standard_met).lower(),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": ";".join(risk_flags),
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": ";".join(supporting_ids),
            "valid_image": str(self.valid_image).lower(),
            "severity": self.severity,
        }
