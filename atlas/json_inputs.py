"""Shared JSON hooks that reject ambiguous objects and nonstandard constants."""

from __future__ import annotations

from typing import NoReturn


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> NoReturn:
    raise ValueError(f"nonfinite JSON constant: {value}")
