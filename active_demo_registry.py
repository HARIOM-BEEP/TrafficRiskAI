"""Active Demo Junction Registry — 50-junction layer for the Streamlit control room.

Loads the active demo junction set from Supabase when available, otherwise falls
back to the embedded local JSON file. The master 705-junction registry in
``junction_registry.py`` is left untouched.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


DATA_DIR = Path(__file__).parent / "data"
FALLBACK_PATH = DATA_DIR / "active_demo_junctions.json"
SUPABASE_CACHE_TTL = 300  # seconds


def _load_from_supabase(force_refresh: bool = False) -> List[Dict[str, Any]] | None:
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    if not supabase_url or not supabase_key:
        return None

    cache_path = DATA_DIR / ".supabase_junctions_cache.json"
    cache_path_ts = DATA_DIR / ".supabase_junctions_cache_ts"

    if not force_refresh and cache_path.exists() and cache_path_ts.exists():
        try:
            cached_ts = float(cache_path_ts.read_text(encoding="utf-8").strip())
            if time.time() - cached_ts < SUPABASE_CACHE_TTL:
                with cache_path.open(encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

    for attempt in range(3):
        try:
            headers = {"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"}
            resp = requests.get(
                f"{supabase_url}/rest/v1/junctions?select=*&order=junction_id",
                headers=headers,
                timeout=(3, 10),
            )
            resp.raise_for_status()
            rows = resp.json()
            if not isinstance(rows, list):
                return None

            results: List[Dict[str, Any]] = []
            for row in rows:
                jid = str(row.get("junction_id", "")).strip()
                cam = str(row.get("camera_id", "")).strip()
                loc = str(row.get("location", "")).strip()
                lat = float(row.get("latitude") or 0.0)
                lon = float(row.get("longitude") or 0.0)

                if cam and not cam.startswith("CAM_"):
                    if cam.startswith("CAM"):
                        tail = cam[3:]
                        if tail.isdigit():
                            cam = f"CAM_{int(tail):03d}"

                record: Dict[str, Any] = {
                    "junction_id": jid,
                    "raw_junction_id": jid[1:] if jid.upper().startswith("J") and jid[1:].isdigit() else jid,
                    "camera_id": cam,
                    "location": loc or f"Junction {jid}",
                    "zone": str(row.get("zone") or "Nagpur Central"),
                    "latitude": lat,
                    "longitude": lon,
                    "risk_score": 0.0,
                    "raw_risk_score": 0.0,
                    "risk_category": "LOW",
                    "historical_risk_score": 0.0,
                    "historical_risk_category": "LOW",
                    "risk_reason": "Active demo junction",
                    "officer_present": 0,
                    "is_unmanned": 1,
                    "recommended_officers": 0,
                    "traffic_volume": 0.0,
                    "average_speed": 0.0,
                    "congestion_level": 0.0,
                    "accident_count": 0.0,
                }
                results.append(record)

            try:
                cache_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
                cache_path_ts.write_text(str(time.time()), encoding="utf-8")
            except Exception:
                pass

            return results
        except Exception:
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
            continue
    return None


def _load_from_fallback() -> List[Dict[str, Any]]:
    if not FALLBACK_PATH.exists():
        return []
    with FALLBACK_PATH.open(encoding="utf-8") as f:
        rows = json.load(f)
    results: List[Dict[str, Any]] = []
    for row in rows:
        jid = str(row.get("junction_id", "")).strip()
        cam = str(row.get("camera_id", "")).strip()
        loc = str(row.get("location", "")).strip()
        lat = float(row.get("latitude") or 0.0)
        lon = float(row.get("longitude") or 0.0)
        zone = str(row.get("zone") or "Nagpur Central")

        record: Dict[str, Any] = {
            "junction_id": jid,
            "raw_junction_id": jid[1:] if jid.upper().startswith("J") and jid[1:].isdigit() else jid,
            "camera_id": cam,
            "location": loc or f"Junction {jid}",
            "zone": zone,
            "latitude": lat,
            "longitude": lon,
            "risk_score": 0.0,
            "raw_risk_score": 0.0,
            "risk_category": "LOW",
            "historical_risk_score": 0.0,
            "historical_risk_category": "LOW",
            "risk_reason": "Active demo junction",
            "officer_present": 0,
            "is_unmanned": 1,
            "recommended_officers": 0,
            "traffic_volume": 0.0,
            "average_speed": 0.0,
            "congestion_level": 0.0,
            "accident_count": 0.0,
        }
        results.append(record)
    return results


def get_active_demo_junctions() -> List[Dict[str, Any]]:
    """Return the 50 active demo junctions.

    Source order:
    1. Supabase REST API (if credentials are present and reachable)
    2. Local fallback JSON file (``data/active_demo_junctions.json``)
    """
    supabase_junctions = _load_from_supabase()
    if supabase_junctions:
        return supabase_junctions
    return _load_from_fallback()
