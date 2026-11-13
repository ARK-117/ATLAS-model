"""Fail closed if the local paper profile enables live execution."""

import json
from pathlib import Path

from .json_inputs import reject_constant, unique_object

PAPER_PROFILE = Path(__file__).resolve().parents[1] / "configs" / "risk_profiles" / "paper.json"
DISABLED_FLAGS = (
    "live_trading_enabled",
    "live_production_mode",
    "broker_connection_configured",
    "allow_autonomous_trading",
)


def require_paper_lock() -> None:
    profile = json.loads(PAPER_PROFILE.read_text(encoding="utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
    if (
        not isinstance(profile, dict)
        or profile.get("permission_level") != "PAPER_TRADING"
        or any(profile.get(flag) is not False for flag in DISABLED_FLAGS)
    ):
        raise ValueError("the paper risk profile must keep live and autonomous execution locked")
