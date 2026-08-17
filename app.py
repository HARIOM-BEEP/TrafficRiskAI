from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from active_demo_registry import get_active_demo_junctions
from allocation import allocate_officers, compare_deployment
from event_store import EventStore
from junction_registry import JunctionRegistry
from pipeline_contracts import UPCOMING_PIPELINE, build_downstream_event, build_redeployment_event, build_risk_event
from redeployment_engine import RedeploymentEngine
from response_engine import ResponseEngine
from risk_classifier import TrafficRiskClassifier
from risk_model import get_ranked_locations, get_unmanned_high_risk, load_risk_data
from vision_pipeline import YoloByteTrackPipeline


st.set_page_config(page_title="TrafficRisk AI | Vision Control", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 8% 2%, #183454 0, #08131f 35%, #040a11 100%); color: #eaf5ff; }
    [data-testid="stMetric"] { background: rgba(17,42,68,.72); border: 1px solid rgba(99,193,255,.22); border-radius: 14px; padding: 14px; }
    .stage-card { background: rgba(16,37,59,.78); border: 1px solid rgba(127,186,237,.20); border-radius: 12px; padding: 14px; min-height: 140px; }
    .stage-live { border-color: #35d2a4; box-shadow: 0 0 20px rgba(53,210,164,.12); }
    .eyebrow { color: #53d8ff; font-size: .83rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
    .risk-card { background: rgba(16,37,59,.78); border: 1px solid rgba(127,186,237,.20); border-radius: 12px; padding: 18px; }
    .risk-base { border-color: #53d8ff; }
    .risk-delta { border-color: #f4b942; }
    .risk-final { border-color: #35d2a4; box-shadow: 0 0 24px rgba(53,210,164,.15); }
    .feature-card { background: rgba(16,37,59,.78); border: 1px solid rgba(127,186,237,.20); border-radius: 12px; padding: 16px; }
    .evidence-step { background: rgba(16,37,59,.60); border: 1px solid rgba(127,186,237,.15); border-radius: 10px; padding: 10px 14px; margin: 4px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_risk_dashboard_data() -> pd.DataFrame:
    return compare_deployment(allocate_officers(load_risk_data(), total_officers=20))


@st.cache_resource(show_spinner=False)
def get_vision_pipeline(
    model_name: str,
    confidence: float,
    window_seconds: int,
    pixels_per_meter: float,
    road_segment_m: float,
) -> YoloByteTrackPipeline:
    return YoloByteTrackPipeline(
        model_name=model_name,
        confidence=confidence,
        window_seconds=window_seconds,
        pixels_per_meter=pixels_per_meter,
        road_segment_m=road_segment_m,
    )


@st.cache_resource(show_spinner=False)
def get_live_risk_classifier() -> TrafficRiskClassifier:
    return TrafficRiskClassifier()


@st.cache_resource(show_spinner=False)
def get_response_engine() -> ResponseEngine:
    return ResponseEngine()


@st.cache_resource(show_spinner=False)
def get_event_store() -> EventStore:
    return EventStore()


@st.cache_resource(show_spinner=False)
def get_junction_registry() -> JunctionRegistry:
    return JunctionRegistry()


@st.cache_resource(show_spinner=False)
def get_redeployment_engine() -> RedeploymentEngine:
    return RedeploymentEngine(registry=get_junction_registry())


def probe_video_file(path: str) -> dict[str, Any]:
    """Probe a video file using OpenCV and return basic metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"valid": False}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    cap.release()
    return {
        "valid": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "fourcc": fourcc_int,
        "fourcc_str": fourcc_str,
    }


def get_browser_preview_path(original_path: str) -> str | None:
    """Return a video only when its original codec is browser-playable.

    OpenCV's ``mp4v`` writer does not create H.264 video. Advertising that
    conversion as browser-compatible caused Chrome's repeated *video source
    error*. The real source stays untouched and is used by YOLOv8 regardless
    of whether a browser player can display it.
    """
    info = probe_video_file(original_path)
    codec = str(info.get("fourcc_str", "")).strip().lower()
    extension = Path(original_path).suffix.lower()
    browser_codecs = {"avc1", "h264", "vp09", "av01"}
    if info.get("valid") and extension in {".mp4", ".webm"} and codec in browser_codecs:
        return original_path
    return None


def get_video_format_label(path: str) -> str:
    """Return a short human-readable format string, e.g. 'MP4 / H.264'."""
    info = probe_video_file(path)
    if not info["valid"]:
        return "Unknown / Invalid"

    fourcc = info.get("fourcc_str", "????")
    codec_map = {
        "avc1": "H.264",
        "H264": "H.264",
        "av01": "AV1",
        "hev1": "H.265",
        "H265": "H.265",
        "vp09": "VP9",
        "mp4v": "MPEG-4",
        "XVID": "XVID",
        "divx": "DivX",
        "MJPG": "MJPEG",
    }
    codec_name = codec_map.get(fourcc, fourcc)

    ext = Path(path).suffix.lower().lstrip(".")
    container_map = {
        ".mp4": "MP4",
        ".avi": "AVI",
        ".mov": "MOV",
        ".mkv": "MKV",
        ".webm": "WebM",
    }
    container = container_map.get(f".{ext}", ext.upper())
    return f"{container} / {codec_name}"


def open_processed_video_writer(path: str, fps: float, frame_size: tuple[int, int]) -> tuple[cv2.VideoWriter | None, str | None]:
    """Open the best available MP4 writer without falsely promising H.264."""
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, frame_size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


def save_uploaded_video(uploaded_file) -> tuple[str, str | None]:
    """Save an uploaded file and expose its original browser-safe preview.

    Returns (original_path, preview_path). The original is always used for
    YOLOv8 / ByteTrack processing. The preview is used only for browser playback.
    """
    cache_key = f"temp_video_{uploaded_file.name}_{uploaded_file.size}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        if isinstance(cached, tuple) and len(cached) == 2:
            original_path, preview_path = cached
            if Path(original_path).exists():
                return original_path, preview_path

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(prefix="trafficrisk_", suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        original_path = temp_file.name

    preview_path = get_browser_preview_path(original_path)

    st.session_state[cache_key] = (original_path, preview_path)
    return original_path, preview_path


def render_pipeline() -> None:
    st.markdown("<p class='eyebrow'>Pipeline status</p>", unsafe_allow_html=True)
    ready_stages = [
        ("01", "CCTV / video input", "Ready", "Video upload or RTSP / CCTV URL"),
        ("02", "YOLOv8 detection", "Ready", "Vehicle-only COCO detection"),
        ("03", "ByteTrack tracking", "Ready", "Persistent vehicle IDs across frames"),
    ]
    columns = st.columns(7)
    for column, (number, title, status, note) in zip(columns, ready_stages):
        column.markdown(f"<div class='stage-card stage-live'><b>{number}</b><br><b>{title}</b><br><span style='color:#35d2a4'>{status}</span><br><small>{note}</small></div>", unsafe_allow_html=True)
    for column, stage in zip(columns[3:], UPCOMING_PIPELINE):
        column.markdown(f"<div class='stage-card stage-live'><b>→</b><br><b>{stage.name}</b><br><span style='color:#35d2a4'>{stage.status}</span><br><small>{stage.produces}</small></div>", unsafe_allow_html=True)


def post_event_to_api(event: dict) -> None:
    """POST a pipeline event to the FastAPI backend.

    Falls back to the local EventStore if the API server is unreachable,
    so processing continues without crashing.  A warning is shown once per
    session so the operator knows the API is down.
    """
    try:
        resp = requests.post("http://127.0.0.1:8502/events", json=event, timeout=1.5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        # API server unreachable — warn once per session then use local store
        if not st.session_state.get("_api_down_warned"):
            st.session_state["_api_down_warned"] = True
            st.warning(
                "⚠️ FastAPI backend (port 8502) is unreachable. "
                "Events are being stored locally only. "
                "Start the API with: `python -m uvicorn api_server:app --port 8502`"
            )
        try:
            get_event_store().append(event)
        except Exception:
            pass
    except Exception:
        # Other errors (timeouts, HTTP errors) — store locally, don't crash
        try:
            get_event_store().append(event)
        except Exception:
            pass


def post_status_to_api(
    junction_id: str, camera_id: str, session_id: str, status: str, details: dict
) -> None:
    event = {
        "event_type": "traffic.pipeline_status.v1",
        "source_name": "Pipeline Status",
        "junction_id": junction_id,
        "camera_id": camera_id,
        "session_id": session_id,
        "status": status,
        "details": details,
    }
    post_event_to_api(event)


def run_video_analysis(
    source: str,
    model_name: str,
    confidence: float,
    max_frames: int,
    source_label: str,
    window_seconds: int,
    pixels_per_meter: float,
    road_segment_m: float,
    junction_id: str = "1",
    camera_id: str = "CAM-01",
    session_id: str = "DEMO-LIVE",
    frame_skipping: int = 1,
    latitude: float = 21.1458,
    longitude: float = 79.0882,
    junction_name: str = "",
) -> None:
    pipeline = get_vision_pipeline(
        model_name, confidence, window_seconds, pixels_per_meter, road_segment_m
    )
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        st.error("Could not open this source. For CCTV, confirm the RTSP/HTTP URL is reachable from this machine.")
        return

    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    
    # Safely compute total processed frames accounting for frame skipping
    if reported_frames > 0:
        actual_video_frames = reported_frames // frame_skipping
        total = min(actual_video_frames, max_frames)
        duration_sec = reported_frames / fps
    else:
        total = max_frames
        duration_sec = 0.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_val = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)]) if fourcc_val > 0 else "UNKNOWN"

    # Format file size for display
    file_size_str = "N/A"
    is_uploaded_file = (source_label != "CCTV stream" and source_label != "")
    if is_uploaded_file:
        try:
            sz = Path(source).stat().st_size
            file_size_str = f"{sz / (1024 * 1024):.1f} MB"
        except Exception:
            pass

    # Print input video diagnostics block
    with st.expander("📊 Input Video Diagnostics Log", expanded=True):
        st.markdown(
            f"**File Name:** `{source_label}` | **Size:** `{file_size_str}`\n\n"
            f"- **FPS:** `{fps:.2f}`\n"
            f"- **Frame Count:** `{reported_frames}`\n"
            f"- **Duration:** `{duration_sec:.1f} seconds`\n"
            f"- **Resolution:** `{width} x {height}`\n"
            f"- **Codec:** `{codec_str}`"
        )

    video_writer = None
    output_temp_path = None
    output_codec = None

    preview_column, details_column = st.columns([3, 2])
    image_slot = preview_column.empty()
    metrics_slot = details_column.empty()
    features_slot = details_column.empty()
    risk_slot = details_column.empty()
    table_slot = details_column.empty()
    progress = st.progress(0, text="Connecting to the vision pipeline…")
    processed = 0
    frame_count = 0
    last_features: dict = {}
    completed_windows: list[dict] = []
    last_timestamp_s = 0.0
    latest_prediction: dict | None = None
    latest_window: dict | None = None
    latest_tracks: list = []

    # Post initial running status to clear the waiting states
    display_label = f"{junction_id} — {junction_name}" if junction_name else junction_id
    post_status_to_api(
        junction_id,
        camera_id,
        session_id,
        "RUNNING",
        {
            "cctv": "LIVE",
            "yolov8": "PROCESSING",
            "bytetrack": "TRACKING",
            "features": "COLLECTING",
            "risk": "READY",
            "response": "READY",
            "redeployment": "READY",
        },
    )

    try:
        with st.spinner("Loading YOLOv8 (the first run may download model weights)…"):
            while processed < max_frames:
                # Fast forward frame skipping using grab() (skips decoding)
                if frame_skipping > 1:
                    for _ in range(frame_skipping - 1):
                        capture.grab()
                        frame_count += 1

                ok, frame = capture.read()
                if not ok:
                    break
                frame_count += 1

                # Resize frame to speed up preprocessing and rendering if larger than 640 width
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))

                capture_timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                last_timestamp_s = capture_timestamp_s if capture_timestamp_s > 0 else frame_count / fps
                annotated, last_features, tracks, ready_windows = pipeline.process_frame(frame, last_timestamp_s)
                completed_windows.extend(ready_windows)
                
                # Record to the video writer
                if video_writer is None:
                    h_ann, w_ann = annotated.shape[:2]
                    suffix = ".mp4"
                    with tempfile.NamedTemporaryFile(prefix="processed_", suffix=suffix, delete=False) as tf:
                        output_temp_path = tf.name
                    video_writer, output_codec = open_processed_video_writer(
                        output_temp_path, fps / frame_skipping, (w_ann, h_ann)
                    )
                    if video_writer is None:
                        raise RuntimeError("Unable to create the processed-video evidence file.")
                
                video_writer.write(annotated)
                
                image_slot.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    caption=f"{source_label} · frame {processed + 1} (skipped {frame_count - processed - 1})",
                    width="stretch",
                )
                metrics_slot.markdown(
                    f"### Live tracking — {display_label}\n"
                    f"**Video source:** Uploaded CCTV Video — Demo Feed  \n"
                    f"**Vehicles in frame:** {last_features['vehicle_count']}  \n"
                    f"**Active track IDs:** {last_features['active_track_count']}  \n"
                    f"**Unique vehicles seen:** {last_features['unique_vehicles_seen']}  \n"
                    f"**New this frame:** {last_features['new_vehicles_this_frame']}  \n"
                    f"**Window traffic state:** {last_features['window_traffic_state'].title()}  \n"
                    f"**Window density:** {last_features['window_density_per_100m']:.2f} vehicles / 100 m  \n"
                    f"**Frame:** {processed + 1} (skip={frame_skipping})"
                )
                features_slot.markdown(
                    f"### 📡 Live Traffic Features\n"
                    f"**Heavy vehicles:** {last_features.get('heavy_vehicle_ratio', 0.0) * 100:.1f}%  \n"
                    f"**Stopped vehicles:** {last_features.get('stopped_vehicle_count', 0)}  \n"
                    f"**Stopped ratio:** {last_features.get('stopped_vehicle_ratio', 0.0) * 100:.1f}%  \n"
                    f"**Queue indicator:** {last_features.get('queue_indicator', 0.0) * 100:.1f}%  \n"
                    f"**Peak hour:** {'YES' if last_features.get('peak_hour_flag') == 1 else 'NO'}  \n"
                    f"**Speed reliable:** {'YES' if last_features.get('speed_reliable') else 'NO'}"
                )
                if tracks:
                    table_slot.dataframe(pd.DataFrame([{
                        "Track ID": track.track_id,
                        "Vehicle": track.label,
                        "Confidence": track.confidence,
                        "Box": track.bbox_xyxy,
                    } for track in tracks]), width="stretch", hide_index=True)
                else:
                    table_slot.info("No vehicles detected in this frame.")

                # Process ready windows instantly
                if ready_windows:
                    classifier = get_live_risk_classifier()
                    response_engine = get_response_engine()
                    for window in ready_windows:
                        # Post status: SCORING / DECIDING
                        post_status_to_api(
                            junction_id, camera_id, session_id, "RUNNING",
                            {"cctv": "LIVE", "yolov8": "PROCESSING", "bytetrack": "TRACKING",
                             "features": "COLLECTING", "risk": "SCORING", "response": "DECIDING", "redeployment": "DECIDING"},
                        )

                        prediction = classifier.predict(window).as_dict()
                        latest_prediction = prediction
                        latest_window = window
                        latest_tracks = tracks

                        st.session_state["latest_live_prediction"] = prediction
                        st.session_state["latest_live_window"] = window

                        f_event = build_downstream_event(window, source_label, junction_id, camera_id, session_id, latitude, longitude)
                        r_event = build_risk_event(window, prediction, source_label, junction_id, camera_id, session_id, latitude, longitude)
                        resp_event = response_engine.process(r_event)
                        redeploy_data = get_redeployment_engine().process(r_event, resp_event)
                        rd_event = build_redeployment_event(r_event, resp_event, redeploy_data)

                        post_event_to_api(f_event)
                        post_event_to_api(r_event)
                        post_event_to_api(resp_event)
                        post_event_to_api(rd_event)

                        # Revert status back to collecting
                        post_status_to_api(
                            junction_id, camera_id, session_id, "RUNNING",
                            {"cctv": "LIVE", "yolov8": "PROCESSING", "bytetrack": "TRACKING",
                             "features": "COLLECTING", "risk": "READY", "response": "READY", "redeployment": "READY"},
                        )

                    if latest_prediction:
                        risk_slot.markdown(
                            f"### 🧠 Hybrid Risk Engine\n"
                            f"**Historical Base Risk:** {latest_prediction.get('base_risk_score_percent', 0.0):.1f}%  \n"
                            f"**Live Traffic Adjustment:** +{latest_prediction.get('live_risk_delta', 0.0) * 100:.1f}%  \n"
                            f"**Final Risk:** {latest_prediction.get('risk_score_percent', 0.0):.1f}%  \n"
                            f"**Risk Level:** {latest_prediction.get('risk_level', 'LOW')}"
                        )

                processed += 1
                progress.progress(min(processed / max(total, 1), 1.0), text=f"Processed {processed} frame(s)")
    except Exception as exc:
        st.error(f"Vision pipeline stopped: {exc}")
        return
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
            if is_uploaded_file:
                st.session_state["processed_video_path"] = output_temp_path
                st.session_state["processed_video_codec"] = output_codec

    progress.empty()
    if last_features:
        st.success(f"Processed {processed} frame(s). YOLOv8 detection and ByteTrack tracking completed.")
        partial_window = pipeline.aggregator.flush(last_timestamp_s)
        if partial_window:
            completed_windows.append(partial_window)
            classifier = get_live_risk_classifier()
            response_engine = get_response_engine()
            
            # Post status: SCORING / DECIDING
            post_status_to_api(
                junction_id,
                camera_id,
                session_id,
                "RUNNING",
                {
                    "cctv": "LIVE",
                    "yolov8": "PROCESSING",
                    "bytetrack": "TRACKING",
                    "features": "COLLECTING",
                    "risk": "SCORING",
                    "response": "DECIDING",
                },
            )
            
            prediction = classifier.predict(partial_window).as_dict()
            f_event = build_downstream_event(partial_window, source_label, junction_id, camera_id, session_id, latitude, longitude)
            r_event = build_risk_event(partial_window, prediction, source_label, junction_id, camera_id, session_id, latitude, longitude)
            resp_event = response_engine.process(r_event)
            redeploy_data = get_redeployment_engine().process(r_event, resp_event)
            rd_event = build_redeployment_event(r_event, resp_event, redeploy_data)
            
            post_event_to_api(f_event)
            post_event_to_api(r_event)
            post_event_to_api(resp_event)
            post_event_to_api(rd_event)
            
        # Post IDLE status at the end
        post_status_to_api(
            junction_id, camera_id, session_id, "IDLE",
            {"cctv": "IDLE", "yolov8": "IDLE", "bytetrack": "IDLE",
             "features": "IDLE", "risk": "READY", "response": "READY", "redeployment": "READY"},
        )

        if completed_windows:
            st.subheader("Aggregated traffic feature windows")
            windows_df = pd.DataFrame(completed_windows)
            if "vehicle_flow_per_hour" in windows_df.columns:
                windows_df["estimated_vehicle_flow_per_hour"] = windows_df["vehicle_flow_per_hour"]
            st.dataframe(windows_df, width="stretch", hide_index=True)
            st.caption("vehicle_flow_per_hour is an estimated flow extrapolated from the aggregation window duration.")
            st.download_button(
                "Download feature windows (CSV)",
                windows_df.to_csv(index=False).encode("utf-8"),
                file_name="traffic_feature_windows.csv",
                mime="text/csv",
            )
            classifier = get_live_risk_classifier()
            predictions = [classifier.predict(window).as_dict() for window in completed_windows]
            response_engine = get_response_engine()
            risk_events = [
                build_risk_event(window, prediction, source_label, junction_id, camera_id, session_id, latitude, longitude)
                for window, prediction in zip(completed_windows, predictions)
            ]
            response_events = [response_engine.process(risk_event) for risk_event in risk_events]
            redeployment_engine = get_redeployment_engine()
            redeployment_events = [redeployment_engine.process(re, resp) for re, resp in zip(risk_events, response_events)]
            responses = [event["response"] for event in response_events]
            
            st.session_state["live_risk_predictions"] = predictions
            st.session_state["live_response_events"] = response_events
            st.session_state["live_redeployment_events"] = redeployment_events
            
            latest_prediction = predictions[-1]
            latest_window = completed_windows[-1]
            st.session_state["latest_live_prediction"] = latest_prediction
            st.session_state["latest_live_window"] = latest_window
            
            # Demo Evidence Section
            render_demo_evidence(latest_window, latest_prediction, latest_tracks or [])
            
            # Hybrid Risk Engine
            render_hybrid_risk_card(latest_prediction)
            
            # Live Traffic Features
            render_live_features_card(latest_window)
            
            st.markdown("---")
            
            # Risk Explanation
            render_risk_explanation(latest_prediction, latest_window)
            
            # High/Critical specific explanation
            render_high_risk_explanation(latest_prediction, latest_window)
            
            # Risk Contribution Breakdown
            render_risk_contributions(latest_prediction, latest_window)
            
            st.markdown("---")
            
            # Existing risk display (preserved)
            score_col, level_col, confidence_col = st.columns(3)
            score_col.metric("Risk score", f"{latest_prediction['risk_score']:.2f}")
            level_col.metric("Risk level", latest_prediction["risk_level"])
            confidence_col.metric("Model confidence", f"{latest_prediction['model_confidence']:.0%}")
            st.caption(" · ".join(latest_prediction["explanation"]))
            if latest_prediction["imputed_context"]:
                st.warning(
                    "Historical median defaults used for unavailable feeds: "
                    + ", ".join(latest_prediction["imputed_context"])
                )
            predictions_df = pd.DataFrame([{
                "risk_score": prediction["risk_score"],
                "risk_level": prediction["risk_level"],
                "model_confidence": prediction["model_confidence"],
                "base_risk_score": prediction.get("base_risk_score", ""),
                "base_risk_percent": prediction.get("base_risk_score_percent", ""),
                "live_risk_delta": prediction.get("live_risk_delta", ""),
                "final_risk_percent": prediction.get("risk_score_percent", ""),
                "explanation": "; ".join(prediction["explanation"]),
                "imputed_context": ", ".join(prediction["imputed_context"]),
            } for prediction in predictions])
            st.dataframe(predictions_df, width="stretch", hide_index=True)
            with st.expander("Risk event handed to the Response Engine"):
                st.json(risk_events[-1])
            render_response_engine(responses, response_events, redeployment_events)


def render_response_engine(responses: list[dict], response_events: list[dict], redeployment_events: list[dict] | None = None) -> None:
    """Display demo-only rules and recommendations for analyzed feature windows."""
    st.subheader("Response Engine - demo recommendations")
    st.caption("Transparent rule-based guidance only. This dashboard does not dispatch police or trigger real-world actions.")
    latest = responses[-1]
    priority_col, status_col, officers_col = st.columns(3)
    priority_col.metric("Latest alert priority", latest["alert_priority"])
    status_col.metric("Alert status", latest["alert_status"])
    officers_col.metric("Recommended officers", latest["recommended_officers"])
    st.info(f"{latest['alert_id']} - {latest['recommended_action']}")
    with st.expander("Decision explanation", expanded=True):
        st.markdown("**Reasons:** " + " | ".join(latest["reasons"]))
        for item in latest["decision_trace"]:
            st.markdown(f"- {item}")
    response_df = pd.DataFrame([{
        "alert_id": response["alert_id"], "timestamp": response["timestamp"],
        "risk_score": response["risk_score"], "risk_level": response["risk_level"],
        "priority": response["alert_priority"], "status": response["alert_status"],
        "recommended_officers": response["recommended_officers"],
        "recommended_action": response["recommended_action"],
    } for response in responses])
    st.dataframe(response_df, width="stretch", hide_index=True)
    st.download_button(
        "Download response recommendations (CSV)", response_df.to_csv(index=False).encode("utf-8"),
        file_name="traffic_response_recommendations.csv", mime="text/csv",
    )
    with st.expander("Latest response event JSON"):
        st.json(response_events[-1])
    
    # --- Redeployment Engine Section ---
    if redeployment_events:
        st.subheader("🚔 Dynamic Redeployment Recommendations")
        st.warning("⚠️ Decision Support Only — Human approval required. No real dispatch occurs.")
        latest_rd = redeployment_events[-1]
        rdcol1, rdcol2, rdcol3 = st.columns(3)
        rdcol1.metric("Junction", latest_rd.get("junction_id", "—"))
        rdcol2.metric("Recommended Additional Officers", latest_rd.get("recommended_additional_officers", 0))
        rdcol3.metric("Priority", latest_rd.get("priority", "—"))
        st.info(f"Action: {latest_rd.get('action', '—')}")
        st.caption(latest_rd.get("demo_notice", ""))
        if latest_rd.get("reason"):
            st.markdown("**Reasons:** " + " | ".join(latest_rd["reason"]))
        with st.expander("Decision Trace", expanded=True):
            for trace in latest_rd.get("decision_trace", []):
                st.markdown(f"- {trace}")
        nearby = latest_rd.get("nearby_redeployments", [])
        if nearby:
            st.markdown("**Suggested Redeployment Sources (nearby low-risk junctions):**")
            st.dataframe(pd.DataFrame(nearby), width="stretch", hide_index=True)
        rd_df = pd.DataFrame([{
            "junction_id": rd.get("junction_id"),
            "camera_id": rd.get("camera_id"),
            "risk_level": rd.get("risk_level"),
            "risk_score": round(rd.get("risk_score", 0), 3),
            "additional_officers": rd.get("recommended_additional_officers", 0),
            "priority": rd.get("priority"),
            "action": rd.get("action"),
            "human_approval_required": rd.get("human_approval_required", True),
        } for rd in redeployment_events])
        st.dataframe(rd_df, width="stretch", hide_index=True)
        with st.expander("Latest redeployment event JSON"):
            st.json(redeployment_events[-1])


def render_live_features_card(window: dict[str, Any]) -> None:
    st.markdown("### 📡 Live Traffic Features")
    col1, col2, col3 = st.columns(3)
    col1.metric("Heavy vehicles", f"{window.get('heavy_vehicle_ratio', 0.0) * 100:.1f}%")
    col1.caption("Bus + truck ratio")
    col2.metric("Stopped vehicles", f"{window.get('stopped_vehicle_count', 0)}")
    col2.caption(f"Ratio: {window.get('stopped_vehicle_ratio', 0.0) * 100:.1f}%")
    col3.metric("Queue indicator", f"{window.get('queue_indicator', 0.0) * 100:.1f}%")
    col3.caption("Estimated traffic queue")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Peak hour", "YES" if window.get("peak_hour_flag") == 1 else "NO")
    col_b.metric("Speed reliable", "YES" if window.get("speed_reliable") else "NO")
    col_c.metric("Traffic state", str(window.get("traffic_state", "unknown")).title())


def render_hybrid_risk_card(prediction: dict[str, Any]) -> None:
    st.markdown("### 🧠 Hybrid Risk Engine")
    base_col, delta_col, final_col = st.columns(3)
    base_col.markdown("<div class='risk-card risk-base'><b>Historical Base Risk</b><br>" + f"{prediction.get('base_risk_score_percent', 0.0):.1f}%</div>", unsafe_allow_html=True)
    delta_col.markdown("<div class='risk-card risk-delta'><b>Live Traffic Adjustment</b><br>" + f"+{prediction.get('live_risk_delta', 0.0) * 100:.1f}%</div>", unsafe_allow_html=True)
    final_score = prediction.get("risk_score_percent", 0.0)
    final_level = prediction.get("risk_level", "LOW")
    final_col.markdown(f"<div class='risk-card risk-final'><b>Final Risk</b><br>{final_score:.1f}%<br><small>{final_level}</small></div>", unsafe_allow_html=True)
    st.markdown("---")


def render_risk_explanation(prediction: dict[str, Any], window: dict[str, Any]) -> None:
    st.markdown("### 🔍 Why This Risk?")
    for line in prediction.get("risk_reasons", prediction.get("explanation", [])):
        st.markdown(f"- {line}")
    density = window.get("vehicle_density_per_100m", 0.0)
    flow = window.get("vehicle_flow_per_hour", 0.0)
    speed = window.get("average_speed_kmh")
    active = window.get("active_track_count", window.get("average_active_vehicles", 0))
    stopped = window.get("stopped_vehicle_count", 0)
    heavy = window.get("heavy_vehicle_ratio", 0.0)
    peak = window.get("peak_hour_flag", 0)
    details = []
    if density:
        details.append(f"Current density: {density:.2f} vehicles / 100m")
    if flow:
        details.append(f"Estimated flow: {flow:.0f} vehicles / hour")
    if speed is not None:
        details.append(f"Average speed: {speed:.1f} km/h")
    if active:
        details.append(f"Active vehicles: {active}")
    if stopped:
        details.append(f"Stopped vehicles: {stopped}")
    if heavy:
        details.append(f"Heavy vehicle ratio: {heavy * 100:.1f}%")
    details.append(f"Peak-hour condition: {'YES' if peak == 1 else 'NO'}")
    details.append(f"Historical base risk: {prediction.get('base_risk_score_percent', 0.0):.1f}%")
    details.append(f"Live adjustment: +{prediction.get('live_risk_delta', 0.0) * 100:.1f}%")
    final_pct = prediction.get("risk_score_percent", 0.0)
    details.append(f"Final risk: {final_pct:.1f}% — {prediction.get('risk_level', 'LOW')}")
    for d in details:
        st.markdown(f"- {d}")


def render_risk_contributions(prediction: dict[str, Any], window: dict[str, Any] | None = None) -> None:
    st.markdown("### 📊 Live Risk Contribution Breakdown")
    contributions = prediction.get("risk_contributions", {})
    if not contributions:
        st.caption("No live adjustment applied.")
        return
    rows = []
    weight_map = {
        "stopped_vehicle_ratio": ("Stopped vehicles", "0.08", "stopped_vehicle_ratio"),
        "queue_indicator": ("Queue indicator", "0.10", "queue_indicator"),
        "heavy_vehicle_ratio": ("Heavy vehicles", "0.05", "heavy_vehicle_ratio"),
        "peak_hour_flag": ("Peak hour", "0.03", "peak_hour_flag"),
    }
    for key, value in contributions.items():
        label, weight, window_key = weight_map.get(key, (key, "—", key))
        raw_value = 0.0
        if window and window_key in window:
            raw_value = window[window_key]
            if key == "peak_hour_flag":
                raw_value = "YES" if raw_value == 1 else "NO"
            else:
                raw_value = f"{float(raw_value) * 100:.1f}%"
        rows.append({"Factor": label, "Value": raw_value, "Weight": weight, "Contribution": f"{value:+.1f}%"})
    total = sum(v for v in contributions.values())
    rows.append({"Factor": "Live adjustment", "Value": "—", "Weight": "—", "Contribution": f"{total:+.1f}%"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_high_risk_explanation(prediction: dict[str, Any], window: dict[str, Any]) -> None:
    level = prediction.get("risk_level", "")
    if level not in ("HIGH", "CRITICAL"):
        return
    st.markdown(f"### ⚠️ Why {level} Risk")
    reasons = []
    density = window.get("vehicle_density_per_100m", 0.0)
    speed = window.get("average_speed_kmh")
    active = window.get("active_track_count", window.get("average_active_vehicles", 0))
    stopped = window.get("stopped_vehicle_count", 0)
    queue = window.get("queue_indicator", 0.0)
    heavy = window.get("heavy_vehicle_ratio", 0.0)
    if density >= 15:
        reasons.append(f"High density: {density:.1f} vehicles / 100m")
    if speed is not None and speed < 20:
        reasons.append(f"Low speed: {speed:.1f} km/h")
    if active and active > 10:
        reasons.append(f"High vehicle count: {active} active vehicles")
    if stopped:
        reasons.append(f"Stopped vehicles detected: {stopped}")
    if queue:
        reasons.append(f"Queue formation detected: {queue * 100:.1f}%")
    if heavy:
        reasons.append(f"Heavy vehicle ratio elevated: {heavy * 100:.1f}%")
    if not reasons:
        reasons.append("Historical base risk elevated by live traffic conditions")
    reasons.append(f"Live risk adjustment: +{prediction.get('live_risk_delta', 0.0) * 100:.1f}%")
    for r in reasons:
        st.markdown(f"- {r}")


def render_demo_evidence(window: dict[str, Any], prediction: dict[str, Any], tracks: list) -> None:
    st.markdown("### 🧾 Demo Evidence")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Video Input**")
        st.caption("Uploaded CCTV video — demo feed")
        st.markdown("**YOLOv8 Detection**")
        if tracks:
            st.dataframe(pd.DataFrame([{
                "Track ID": t.track_id,
                "Vehicle": t.label,
                "Confidence": f"{t.confidence:.2f}",
            } for t in tracks]), width="stretch", hide_index=True)
        else:
            st.caption("No vehicles detected in current frame")
    with col2:
        st.markdown("**ByteTrack IDs**")
        if tracks:
            st.caption(f"Persistent IDs: {', '.join(str(t.track_id) for t in tracks[:10])}")
        st.markdown("**Live Features**")
        st.caption(f"Density: {window.get('vehicle_density_per_100m', 0.0):.1f} / 100m | Estimated flow: {window.get('vehicle_flow_per_hour', 0.0):.0f}/hr | State: {window.get('traffic_state', 'unknown')}")
        st.markdown("**Hybrid Risk**")
        st.caption(f"Base: {prediction.get('base_risk_score_percent', 0.0):.1f}% | Delta: +{prediction.get('live_risk_delta', 0.0) * 100:.1f}% | Final: {prediction.get('risk_score_percent', 0.0):.1f}% ({prediction.get('risk_level', 'LOW')})")
        st.markdown("**Recommendation**")
        st.caption("See Response Engine section below")


def render_storage_api() -> None:
    st.subheader("Event store, API, and live updates")
    st.caption("Events are persisted locally in SQLite. The API and WebSocket are demo interfaces, not public or production services.")

    @st.fragment(run_every="5s")
    def live_event_view() -> None:
        store = get_event_store()
        events = store.recent(limit=25)
        total_col, latest_col, stream_col = st.columns(3)
        total_col.metric("Persisted events", store.count())
        latest_col.metric("Latest event", events[0]["event_type"] if events else "No events yet")
        stream_col.metric("Live refresh", "Every 5 seconds")
        if not events:
            st.info("Run a video analysis to persist feature, risk, and response events here.")
            return
        event_rows = [{
            "id": event["id"], "recorded_at": event["recorded_at"],
            "event_type": event["event_type"], "source_name": event["source_name"],
            "event_id": event["event_id"],
        } for event in events]
        st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)
        with st.expander("Latest persisted event"):
            st.json(events[0])

    live_event_view()
    st.markdown("**Local API endpoints**")
    st.code(
        "uvicorn api_server:app --host 127.0.0.1 --port 8502\n"
        "GET http://127.0.0.1:8502/health\n"
        "GET http://127.0.0.1:8502/events\n"
        "WebSocket ws://127.0.0.1:8502/ws/events",
        language="text",
    )
    st.info("The API server is optional and runs separately from Streamlit. It exposes only local demo event data.")


def render_vision_tab() -> None:
    st.subheader("Live vehicle detection & tracking")
    st.caption("Detects cars, motorcycles, buses, and trucks with YOLOv8; ByteTrack maintains IDs across frames.")
    
    st.markdown("### 🎬 Real Pipeline Demo Setup")
    active_junctions = get_active_demo_junctions()
    
    # Build select options with active demo junction metadata
    junction_options = [
        f"{j['junction_id']} — {j['location']}"
        for j in active_junctions
    ]
    
    setup_col1, setup_col2, setup_col3 = st.columns(3)
    
    with setup_col1:
        selected_idx = st.selectbox(
            "Target Junction (Nagpur CCTV)",
            range(len(junction_options)),
            format_func=lambda i: junction_options[i],
            index=0,
            help="Select one of the 50 active demo junctions for your uploaded video.",
        )
        selected_junction = active_junctions[selected_idx]
        junction_id = selected_junction["junction_id"]
        latitude = selected_junction["latitude"]
        longitude = selected_junction["longitude"]
        junction_name = selected_junction["location"]
        st.caption(f"📍 Monitoring Junction: {junction_name}")
        st.caption(f"Lat/Lon: {latitude:.5f}, {longitude:.5f}")
        st.caption("🟢 Demo Camera Mapping — uploaded video is mapped to this demo junction for display purposes")
        
    with setup_col2:
        camera_id = st.text_input(
            "Camera ID",
            value=selected_junction["camera_id"],
            help="Stable camera identifier used for live mapping.",
        )
        
    with setup_col3:
        default_session = f"DEMO-{datetime.now().strftime('%Y-%m-%d-%H%M')}"
        session_id = st.text_input(
            "Demo Session ID",
            value=default_session,
            help="Groups your live events. The React app can display only this session's events if desired.",
        )

    source_mode = st.radio("Input type", ["Upload video", "CCTV / RTSP URL"], horizontal=True)
    source = None
    source_label = ""
    upload_status = st.empty()
    
    if source_mode == "Upload video":
        uploaded = st.file_uploader("Upload a road-traffic video", type=["mp4", "avi", "mov", "mkv"], key="traffic_video_uploader")
        if uploaded:
            upload_status.success(f"Video uploaded ✓ ({uploaded.size / (1024*1024):.1f} MB)")
            source, preview = save_uploaded_video(uploaded)
            source_label = uploaded.name
            format_label = get_video_format_label(source)
            preview_available = preview is not None and Path(preview).exists()
            st.caption(f"📹 Video Source: Uploaded CCTV Video — Demo Feed ({uploaded.name})")
            st.caption(f"Video Format: {format_label}")
            if preview_available:
                st.caption("Preview: Browser compatible ✓")
                st.video(preview)
            else:
                st.caption("Preview: Browser preview unavailable — original video will be used for AI processing")
                st.warning("Video preview could not be decoded. Please upload an MP4/H.264, WebM, or AVI traffic video.")
        else:
            upload_status.info("Select a video file to upload")
    else:
        source = st.text_input("CCTV / RTSP URL", placeholder="rtsp://username:password@camera-host:554/stream")
        source_label = source or "CCTV stream"
        st.info("Live sources run for the selected frame limit so the Streamlit session remains responsive.")
        st.caption(f"📹 Video Source: {source_label}")

    # Show latest prediction if available from a previous run
    if "latest_live_prediction" in st.session_state and "latest_live_window" in st.session_state:
        prev_pred = st.session_state["latest_live_prediction"]
        prev_window = st.session_state["latest_live_window"]
        st.markdown("---")
        st.markdown("### 🧠 Previous Run — Latest Risk Snapshot")
        bcol, dcol, fcol = st.columns(3)
        bcol.metric("Base Risk", f"{prev_pred.get('base_risk_score_percent', 0.0):.1f}%")
        dcol.metric("Live Delta", f"+{prev_pred.get('live_risk_delta', 0.0) * 100:.1f}%")
        fcol.metric("Final Risk", f"{prev_pred.get('risk_score_percent', 0.0):.1f}% ({prev_pred.get('risk_level', 'LOW')})")
        st.caption(f"Estimated flow: {prev_window.get('vehicle_flow_per_hour', 0.0):.0f} vehicles / hour | Density: {prev_window.get('vehicle_density_per_100m', 0.0):.2f} / 100m | Speed: {prev_window.get('average_speed_kmh', 'N/A')} km/h")

    options, controls, skips = st.columns([2, 1, 1])
    with options:
        model_name = st.selectbox("YOLOv8 model", ["yolov8n.pt", "yolov8s.pt"], help="Nano is faster; Small can improve accuracy.")
        confidence = st.slider("Detection confidence", 0.15, 0.85, 0.35, 0.05)
    with controls:
        analyze_entire = False
        if source_mode == "Upload video":
            analyze_entire = st.checkbox(
                "Analyze entire video",
                value=False,
                help="Process all frames in the video instead of a limited count."
            )
        if analyze_entire:
            max_frames = 999999
        else:
            max_frames = st.number_input("Frames to analyze", min_value=1, max_value=10000, value=120, step=30)
        st.caption("Use a finite batch for testing; a future worker can run a 24/7 stream.")
    with skips:
        frame_skipping = st.slider("Frame skipping", 1, 5, 2, help="Process every N-th frame to speed up analysis.")

    with st.expander("Feature aggregation settings", expanded=True):
        setting_a, setting_b, setting_c = st.columns(3)
        window_seconds = setting_a.number_input(
            "Aggregation window (seconds)", min_value=5, max_value=900, value=10, step=5,
            help="Default is 10s for responsive judge demonstration. Adjust as needed.",
        )
        pixels_per_meter = setting_b.number_input(
            "Camera calibration (pixels / metre)", min_value=0.0, value=0.0, step=0.5,
            help="Set this from a known road measurement to calculate real speed. Leave at 0 to omit speed.",
        )
        road_segment_m = setting_c.number_input(
            "Observed road segment (metres)", min_value=10.0, value=100.0, step=10.0,
            help="Used to normalize vehicle density per 100 metres.",
        )
        st.caption("The traffic-state label is a density indicator only. Risk predictions feed transparent, demo-only response recommendations.")

    if st.button("Start YOLOv8 + ByteTrack", type="primary", width="stretch"):
        if not source:
            st.warning("Choose a video file or provide a CCTV / RTSP URL first.")
        else:
            for key in ["processed_video_path", "latest_live_prediction", "latest_live_window",
                        "live_risk_predictions", "live_response_events", "live_redeployment_events"]:
                if key in st.session_state:
                    del st.session_state[key]
            get_vision_pipeline.clear()
            run_video_analysis(
                source, model_name, confidence, int(max_frames), source_label,
                int(window_seconds), float(pixels_per_meter), float(road_segment_m),
                junction_id=junction_id, camera_id=camera_id, session_id=session_id,
                frame_skipping=int(frame_skipping), latitude=latitude, longitude=longitude,
                junction_name=junction_name,
            )

    # Render processed video player if output exists
    if "processed_video_path" in st.session_state and st.session_state["processed_video_path"]:
        st.subheader("🎥 Processed Video Output")
        st.video(st.session_state["processed_video_path"])


def render_risk_dashboard() -> None:
    df = get_risk_dashboard_data()
    live_predictions = st.session_state.get("live_risk_predictions", [])
    if live_predictions:
        latest = live_predictions[-1]
        st.info(
            f"Latest vision risk prediction: {latest['risk_level']} "
            f"({latest['risk_score']:.2f}, confidence {latest['model_confidence']:.0%})."
        )
    critical_risk = len(df[df["risk_category"] == "CRITICAL"])
    high_risk = len(df[df["risk_category"] == "HIGH"])
    medium_risk = len(df[df["risk_category"] == "MEDIUM"])
    unmanned_high = len(get_unmanned_high_risk(df))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Junctions", len(df))
    c2.metric("Critical / High Risk", critical_risk + high_risk, help=f"Critical: {critical_risk}, High: {high_risk}")
    c3.metric("Medium Risk", medium_risk)
    c4.metric("Unmanned High/Critical", unmanned_high)

    map_data = folium.Map(location=[21.1458, 79.0882], zoom_start=11)
    palette = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]], radius=7,
            color=palette.get(row["risk_category"], "blue"), fill=True,
            popup=folium.Popup(f"<b>Junction {row['CCTV Junction No.']}</b><br>Risk: {row['risk_score']}<br>Category: {row['risk_category']}<br>{row['risk_reason']}", max_width=300),
        ).add_to(map_data)
    st_folium(map_data, width=None, height=430)
    columns = ["CCTV Junction No.", "location", "risk_score", "risk_category", "recommended_officers", "risk_reason"]
    st.dataframe(get_ranked_locations(df)[columns].head(15), width="stretch", hide_index=True)


st.title("🚦 TrafficRisk AI Control Room")
st.caption("Live vision, feature aggregation, risk scoring, demo response recommendations, and local event persistence are active.")
render_pipeline()
vision_tab, dashboard_tab, storage_tab, roadmap_tab = st.tabs(["Vision console", "Risk dashboard", "Event store & API", "Pipeline roadmap"])
with vision_tab:
    render_vision_tab()
with dashboard_tab:
    render_risk_dashboard()
with storage_tab:
    render_storage_api()
with roadmap_tab:
    st.subheader("Extension-ready pipeline")
    st.caption("The active pipeline emits windowed traffic features and a trained risk prediction event.")
    for stage in UPCOMING_PIPELINE:
        st.markdown(f"**{stage.name} · {stage.status}** — accepts: `{stage.accepts}` → produces: `{stage.produces}`")
    st.info("Storage, local API access, and live dashboard updates are active. External integrations and non-vision context feeds remain intentionally out of scope.")
