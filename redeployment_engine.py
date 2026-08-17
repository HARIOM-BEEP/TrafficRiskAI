from datetime import datetime, timezone
import math
from typing import Any, Dict, List
from junction_registry import JunctionRegistry

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Radius of the Earth in km
    R = 6371.0
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class RedeploymentEngine:
    def __init__(self, registry: JunctionRegistry = None) -> None:
        self.registry = registry or JunctionRegistry()

    def process(self, risk_event: Dict[str, Any], response_event: Dict[str, Any]) -> Dict[str, Any]:
        prediction = risk_event.get("prediction", {})
        features = risk_event.get("features", {})
        response = response_event.get("response", {})
        
        raw_j_id = risk_event.get("junction_id") or features.get("junction_id") or risk_event.get("source_name")
        # Get registry entry
        junction = self.registry.get_junction(raw_j_id)
        
        junction_id = junction["junction_id"] if junction else (f"J{raw_j_id:03d}" if str(raw_j_id).isdigit() else str(raw_j_id))
        camera_id = junction["camera_id"] if junction else risk_event.get("camera_id", "CAM-01")
        latitude = junction["latitude"] if junction else float(risk_event.get("latitude") or 21.1458)
        longitude = junction["longitude"] if junction else float(risk_event.get("longitude") or 79.0882)
        
        risk_score = float(prediction.get("risk_score") or 0.0)
        risk_level = str(prediction.get("risk_level") or "LOW").upper()
        density = float(features.get("vehicle_density_per_100m") or 0.0)
        flow = float(features.get("vehicle_flow_per_hour") or 0.0)
        traffic_state = str(features.get("traffic_state") or "light").lower()
        
        # Determine recommended additional officers
        # LOW: 0, MEDIUM: 0-1, HIGH: 1-2, CRITICAL: 2-3
        additional_officers = 0
        priority = "LOW"
        action = "MONITOR"
        
        if risk_level == "MEDIUM":
            additional_officers = 1 if density >= 7 or flow >= 600 else 0
            priority = "MEDIUM"
            action = "ENHANCE_MONITORING"
        elif risk_level == "HIGH":
            additional_officers = 2 if density >= 12 or flow >= 800 else 1
            priority = "HIGH"
            action = "RECOMMEND_ADDITIONAL_COVERAGE"
        elif risk_level == "CRITICAL":
            additional_officers = 3 if density >= 15 or flow >= 1000 else 2
            priority = "URGENT"
            action = "RECOMMEND_IMMEDIATE_REDEPLOYMENT"
            
        reasons = list(prediction.get("explanation", []))
        if not reasons:
            reasons.append("Standard rule-based recommendation")
            
        trace = [
            f"Risk score {risk_score:.2f} maps to {risk_level} redeployment policy.",
            f"Observed density: {density:.1f} vehicles/100m, flow: {flow:.0f}/hr, state: {traffic_state}.",
        ]
        
        # Find nearby low-risk junctions that have officers historically present
        nearby_redeployments = []
        if additional_officers > 0 and junction:
            all_junctions = self.registry.list_all()
            candidates = []
            for other in all_junctions:
                if other["junction_id"] == junction_id:
                    continue
                dist = haversine_distance(latitude, longitude, other["latitude"], other["longitude"])
                # We want close junctions (e.g. within 15km in Nagpur)
                if dist <= 15.0:
                    candidates.append((dist, other))
                    
            # Sort by distance
            candidates.sort(key=lambda x: x[0])
            
            # Find low-risk candidates with officers historically present
            redeploy_sources = []
            for dist, other in candidates:
                if len(redeploy_sources) >= additional_officers:
                    break
                if other["historical_risk_category"] in ["LOW", "MEDIUM"] and other["officer_present"] > 0:
                    redeploy_sources.append((dist, other))
            
            # If we don't have enough low-risk with officers, just take the closest low-risk
            if len(redeploy_sources) < additional_officers:
                for dist, other in candidates:
                    if len(redeploy_sources) >= additional_officers:
                        break
                    if other not in [x[1] for x in redeploy_sources] and other["historical_risk_category"] in ["LOW", "MEDIUM"]:
                        redeploy_sources.append((dist, other))

            for dist, source in redeploy_sources:
                nearby_redeployments.append({
                    "from_junction_id": source["junction_id"],
                    "from_junction_name": source["location"],
                    "latitude": source["latitude"],
                    "longitude": source["longitude"],
                    "distance_km": round(dist, 2),
                    "officers_present": source["officer_present"],
                    "historical_risk": source["historical_risk_category"]
                })
                trace.append(
                    f"Recommend shifting 1 officer from {source['junction_id']} ({source['location']}, {dist:.2f} km away, low historical risk)."
                )

        return {
            "event_type": "traffic.redeployment_recommendation.v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "junction_id": junction_id,
            "camera_id": camera_id,
            "latitude": latitude,
            "longitude": longitude,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_additional_officers": additional_officers,
            "priority": priority,
            "action": action,
            "reason": reasons,
            "human_approval_required": True,
            "demo_notice": "Demo/rule-based recommendation — human approval required. No real police dispatch is occurring.",
            "decision_trace": trace,
            "nearby_redeployments": nearby_redeployments
        }
