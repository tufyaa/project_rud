"""Utilities for the demo client."""
from __future__ import annotations

from typing import Any
import json
import requests


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON and return JSON response."""
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def pretty_print(title: str, data: dict[str, Any]) -> None:
    """Pretty print JSON data."""
    print(f"\n=== {title} ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))
