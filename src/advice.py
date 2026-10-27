"""Phase 6: action advice per disease and stage (src/advice.json).

Draft guidance from university extension fact sheets; `expert_verified` is false
until a plant pathologist reviews advice.json. No chemical products or doses.
"""

from __future__ import annotations

import json
from pathlib import Path

ADVICE = json.loads((Path(__file__).resolve().parent / "advice.json").read_text(encoding="utf-8"))
STAGES = ["none", "early", "moderate", "severe", "critical"]


def advice_for(disease: str | None, stage: str | None) -> dict | None:
    """Advice for a reliable diagnosis; None when there is nothing trustworthy to advise on."""
    table = ADVICE["diseases"].get(disease or "")
    if table is None:
        return None
    if disease == "no_disease":
        actions, stage = table["none"], "none"
    else:
        stage = stage if stage in table else "early"
        actions = table[stage]
    return {
        "disease": disease,
        "stage": stage,
        "about": table.get("why"),
        # black rot and downy mildew look alike on leaves; the model confuses classic black rot
        # (yellow V-shapes from the edge) with downy mildew (outputs/final/WEB_PROBE_REPORT.md)
        "check_first": table.get("check"),
        "actions": actions,
        "general": ADVICE["general"] if disease not in ("no_disease", "insect_damage") else [],  # hygiene against pathogens
        "sources": ADVICE["sources"].get(disease, []),
        "expert_verified": ADVICE["expert_verified"],
        "disclaimer": ADVICE["note"],
    }
