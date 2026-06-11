"""Deterministic source-type tagging for extraction evidence."""

from __future__ import annotations

import re


SOURCE_KIND_LABELS = {
    "rfq": "RFQ",
    "scada_standard": "SCADA Standard",
    "workshop_note": "Workshop Note",
    "evaluation_memo": "Evaluation Memo",
    "appendix_form": "Appendix/Form",
    "background_reference": "Background Reference",
    "unknown": "Other Source",
}


def source_kind_label(kind: str) -> str:
    """Return a human-readable label for a source kind."""
    return SOURCE_KIND_LABELS.get(str(kind or "").strip(), SOURCE_KIND_LABELS["unknown"])


def infer_source_kind(*, text: str = "", filename: str = "") -> str:
    """Classify evidence into broad document-context buckets using deterministic rules."""
    hay = f"{filename}\n{text}".casefold()
    clean = re.sub(r"\s+", " ", hay)

    if _has_any(clean, (
        "rfq no.",
        "request for qualification",
        "request for qualifications",
        "city of tampa, florida - rfq",
        "selection & certification committee",
        "transmittal memorandum",
    )):
        return "rfq"

    standard_hits = _count_matches(clean, (
        "scada standards volume",
        "scada & pcs standards",
        "city standards",
        "standard details",
        "display definitions",
        "changeset files",
        "graphic symbols",
        "historian exchange",
    ))
    if standard_hits >= 1 and _has_any(clean, ("hmi", "oit", "scada", "pcs", "plc", "historian", "programming software")):
        return "scada_standard"

    if _has_any(clean, ("meeting minutes", "workshop no.", "discussion items:", "attendees:", "meeting agenda", "action items")):
        return "workshop_note"

    evaluation_hits = _count_matches(clean, (
        "evaluation",
        "report creator software",
        "hardware evaluation",
        "software platform",
        "ranking of",
        "scoring",
        "recommended platform",
        "comparison matrix",
    ))
    if evaluation_hits >= 1 or (
        _has_any(clean, ("ignition", "xlreport", "pivision", "factorytalk", "vtscada"))
        and _has_any(clean, ("evaluation", "ranking", "score", "comparison", "option"))
    ):
        return "evaluation_memo"

    appendix_hits = _count_matches(clean, (
        "good faith effort",
        "gfecp",
        "solicited subcontractors",
        "to-be-utilized sub-",
        "respondent certification",
        "non-collusion affidavit",
        "signature of respondent",
        "notary public",
        "sworn to and subscribed",
        "bid form",
        "proposal form",
    ))
    if appendix_hits >= 1 or (
        _has_any(clean, ("appendix", "form", "affidavit", "certification"))
        and _has_any(clean, ("signature", "respondent", "subcontractor", "notary", "acknowledged before me"))
    ):
        return "appendix_form"

    background_hits = _count_matches(clean, (
        "1.1 background",
        "master plan",
        "is permitted to treat",
        "type i two-stage",
        "tampa bay",
        "historically",
        "existing facility",
    ))
    if background_hits >= 1:
        return "background_reference"

    return "unknown"


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _count_matches(text: str, needles: tuple[str, ...]) -> int:
    return sum(1 for needle in needles if needle in text)
