"""
Junction Registry — Single Source of Truth for Nagpur CCTV Junction Data
========================================================================
Integrates and normalizes junction metadata from the Nagpur CCTV dataset:
  - Junction IDs: Standardized as 'J001', 'J002', etc. (with raw integer mappings).
  - Camera IDs: Standardized as 'CAM_001', 'CAM_002', etc.
  - Consistent schema: location, zone, lat/lon, risk_score, risk_category,
    risk_reason, officer_present, is_unmanned, and recommended_officers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


class JunctionRegistry:
    """Authoritative registry for all 705 Nagpur CCTV traffic junctions."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.junctions: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _open_csv(self, path: Path):
        for enc in ("utf-8", "cp1252", "latin1"):
            try:
                with path.open(encoding=enc, newline="") as f:
                    return list(csv.DictReader(f))
            except (UnicodeDecodeError, Exception):
                continue
        return []

    def _load_registry(self) -> None:
        risk_data_path = self.data_dir / "traffic_risk_data.csv"
        junctions_list_path = self.data_dir / "CCTV_Junctionslist-Nagpur-2017-2018.csv"

        # 1. Read zones and pole locations from the original CCTV list if available
        zone_map: Dict[str, str] = {}
        if junctions_list_path.exists():
            for row in self._open_csv(junctions_list_path):
                j_id = str(row.get("CCTV Junction No.", "")).strip()
                zone = str(row.get("Zone Name", "Nagpur Central")).strip()
                if j_id and j_id not in zone_map:
                    zone_map[j_id] = zone

        # 2. Read full risk data (705 junctions)
        if risk_data_path.exists():
            for row in self._open_csv(risk_data_path):
                raw_id = str(row.get("CCTV Junction No.", "")).strip()
                if not raw_id:
                    continue

                try:
                    num = int(raw_id)
                    std_j_id = f"J{num:03d}"
                    std_cam_id = f"CAM_{num:03d}"
                except ValueError:
                    std_j_id = f"J{raw_id}"
                    std_cam_id = f"CAM_{raw_id}"

                try:
                    lat = float(row.get("latitude") or 0.0)
                except ValueError:
                    lat = 21.1458
                try:
                    lon = float(row.get("longitude") or 0.0)
                except ValueError:
                    lon = 79.0882

                try:
                    raw_score = float(row.get("risk_score") or 0.0)
                    score_norm = round(raw_score / 100.0, 3) if raw_score > 1.0 else round(raw_score, 3)
                except ValueError:
                    raw_score = 0.0
                    score_norm = 0.0

                try:
                    officers = int(float(row.get("officer_present") or 0.0))
                except ValueError:
                    officers = 0

                # Consistently map risk category from risk score to align database & ML model
                if score_norm >= 0.85:
                    cat = "CRITICAL"
                elif score_norm >= 0.70:
                    cat = "HIGH"
                elif score_norm >= 0.40:
                    cat = "MEDIUM"
                else:
                    cat = "LOW"

                reason = str(row.get("risk_reason") or "Standard baseline traffic condition").strip()
                if reason.lower() in ("nan", ""):
                    reason = "Standard baseline traffic condition"

                # Align with allocation rules: HIGH/CRITICAL -> 2 officers, MEDIUM -> 1 officer, LOW -> 0 officers
                if cat in ("HIGH", "CRITICAL"):
                    rec_officers = 2
                elif cat == "MEDIUM":
                    rec_officers = 1
                else:
                    rec_officers = 0

                record = {
                    "junction_id": std_j_id,
                    "raw_junction_id": raw_id,
                    "camera_id": std_cam_id,
                    "location": str(row.get("location") or f"Junction {raw_id}").strip(),
                    "zone": zone_map.get(raw_id, "Nagpur Metro"),
                    "latitude": lat,
                    "longitude": lon,
                    "risk_score": score_norm,
                    "raw_risk_score": raw_score,
                    "risk_category": cat,
                    "historical_risk_score": score_norm,
                    "historical_risk_category": cat,
                    "risk_reason": reason,
                    "officer_present": officers,
                    "is_unmanned": 1 if officers == 0 else 0,
                    "recommended_officers": rec_officers,
                    "traffic_volume": float(row.get("traffic_volume") or 0.0),
                    "average_speed": float(row.get("average_speed") or 0.0),
                    "congestion_level": float(row.get("congestion_level") or 0.0),
                    "accident_count": float(row.get("accident_count") or 0.0),
                }
                self.junctions[raw_id] = record
                self.junctions[std_j_id] = record

    def get_junction(self, junction_id: Any) -> Optional[Dict[str, Any]]:
        """Lookup a junction by 'J001', '1', 'CAM_001', or location name."""
        if junction_id is None:
            return None
        query = str(junction_id).strip()

        # Direct key match
        if query in self.junctions:
            return self.junctions[query]

        # Try stripped J prefix
        if query.upper().startswith("J") and query[1:].isdigit():
            raw = str(int(query[1:]))
            if raw in self.junctions:
                return self.junctions[raw]

        # Try CAM prefix
        if query.upper().startswith("CAM_") or query.upper().startswith("CAM-"):
            tail = query[4:]
            if tail.isdigit():
                raw = str(int(tail))
                if raw in self.junctions:
                    return self.junctions[raw]

        # Case-insensitive check on junction_id or location
        q_lower = query.lower()
        for j in self.list_all():
            if j["junction_id"].lower() == q_lower or j["raw_junction_id"] == query:
                return j
            if q_lower in j["location"].lower():
                return j
        return None

    def list_all(self) -> List[Dict[str, Any]]:
        """Return unique list of all 705 junctions sorted numerically."""
        seen = set()
        unique = []
        for j in self.junctions.values():
            jid = j["junction_id"]
            if jid not in seen:
                seen.add(jid)
                unique.append(j)
        try:
            return sorted(unique, key=lambda x: int(x["raw_junction_id"]))
        except ValueError:
            return unique

    def get_high_risk(self) -> List[Dict[str, Any]]:
        return [j for j in self.list_all() if j["risk_category"] in ("HIGH", "CRITICAL") or j["risk_score"] >= 0.70]

    def get_unmanned_high_risk(self) -> List[Dict[str, Any]]:
        return [j for j in self.get_high_risk() if j["is_unmanned"] == 1]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = query.lower().strip()
        if not q:
            return []
        matches = []
        for j in self.list_all():
            if q in j["junction_id"].lower() or q in j["raw_junction_id"] or q in j["location"].lower() or q in j["zone"].lower():
                matches.append(j)
                if len(matches) >= limit:
                    break
        return matches
