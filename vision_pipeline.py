"""YOLOv8, ByteTrack, and windowed traffic-feature aggregation utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


VEHICLE_CLASS_IDS: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class TrafficFeatureAggregator:
    """Converts frame-level ByteTrack observations into traffic feature windows.

    Pixel motion is reported as km/h only when a camera calibration value is
    provided. This avoids presenting uncalibrated pixel movement as real speed.
    """

    def __init__(
        self,
        window_seconds: int = 300,
        pixels_per_meter: float | None = None,
        road_segment_m: float = 100.0,
        stopped_vehicle_threshold_px: float = 3.0,
        stopped_vehicle_min_frames: int = 3,
    ) -> None:
        self.window_seconds = window_seconds
        self.pixels_per_meter = pixels_per_meter if pixels_per_meter and pixels_per_meter > 0 else None
        self.road_segment_m = max(road_segment_m, 1.0)
        self.stopped_vehicle_threshold_px = stopped_vehicle_threshold_px
        self.stopped_vehicle_min_frames = stopped_vehicle_min_frames
        self._all_seen_ids: set[int] = set()
        self._last_positions: dict[int, tuple[tuple[float, float], float]] = {}
        self._window_start_s: float | None = None
        self._window_ids: set[int] = set()
        self._window_breakdown: Counter[str] = Counter()
        self._active_counts: list[int] = []
        self._speeds_kmh: list[float] = []
        self._frames_processed = 0
        self._consecutive_low_motion: dict[int, int] = {}
        self._stopped_track_ids: set[int] = set()

    def _reset_window(self, start_s: float) -> None:
        self._window_start_s = start_s
        self._window_ids = set()
        self._window_breakdown = Counter()
        self._active_counts = []
        self._speeds_kmh = []

    def _is_peak_hour(self, timestamp_s: float) -> int:
        dt_obj = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        hour = dt_obj.hour
        if (7 <= hour < 10) or (17 <= hour < 20):
            return 1
        return 0

    def _congestion_state(self, density_per_100m: float) -> str:
        if density_per_100m >= 15:
            return "high"
        if density_per_100m >= 7:
            return "moderate"
        return "light"

    def _compute_live_features(
        self,
        tracks: list[TrackObservation],
        timestamp_s: float,
        frame_ids: set[int],
    ) -> dict[str, Any]:
        elapsed_window = max(timestamp_s - (self._window_start_s or timestamp_s), 0.001)
        avg_active = round(float(np.mean(self._active_counts)), 2) if self._active_counts else 0.0
        density = round(avg_active / self.road_segment_m * 100, 2)

        vehicle_breakdown = dict(sorted(Counter(track.label for track in tracks).items()))
        total_vehicles = sum(vehicle_breakdown.values())
        heavy_vehicle_ratio = round(
            (vehicle_breakdown.get("bus", 0) + vehicle_breakdown.get("truck", 0)) / total_vehicles, 3
        ) if total_vehicles > 0 else 0.0

        active_count = len(frame_ids)
        stopped_count = len(self._stopped_track_ids)
        stopped_ratio = round(stopped_count / active_count, 3) if active_count > 0 else 0.0
        queue_indicator = round(stopped_ratio * min(density / 10.0, 1.0), 3)
        queue_indicator = max(0.0, min(queue_indicator, 1.0))

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "vehicle_count": len(tracks),
            "active_track_count": active_count,
            "unique_vehicles_seen": len(self._all_seen_ids),
            "new_vehicles_this_frame": len(frame_ids - self._all_seen_ids),
            "vehicle_breakdown": vehicle_breakdown,
            "window_elapsed_s": round(elapsed_window, 2),
            "window_unique_vehicles": len(self._window_ids),
            "window_average_active_vehicles": avg_active,
            "window_density_per_100m": density,
            "window_traffic_state": self._congestion_state(density),
            "frames_processed": self._frames_processed,
            "average_speed_kmh": round(float(np.mean(self._speeds_kmh)), 2) if self._speeds_kmh else None,
            "speed_reliable": self.pixels_per_meter is not None,
            "heavy_vehicle_ratio": heavy_vehicle_ratio,
            "stopped_vehicle_count": stopped_count,
            "stopped_vehicle_ratio": stopped_ratio,
            "queue_indicator": queue_indicator,
            "peak_hour_flag": self._is_peak_hour(timestamp_s),
        }

    def _build_window(self, end_s: float, complete: bool) -> dict[str, Any] | None:
        if self._window_start_s is None or not self._active_counts:
            return None
        duration = max(end_s - self._window_start_s, 0.001)
        avg_active = round(float(np.mean(self._active_counts)), 2)
        density = round(avg_active / self.road_segment_m * 100, 2)
        speed = round(float(np.mean(self._speeds_kmh)), 2) if self._speeds_kmh else None

        vehicle_breakdown = dict(sorted(self._window_breakdown.items()))
        total_vehicles = sum(vehicle_breakdown.values())
        heavy_vehicle_ratio = round(
            (vehicle_breakdown.get("bus", 0) + vehicle_breakdown.get("truck", 0)) / total_vehicles, 3
        ) if total_vehicles > 0 else 0.0

        active_count = self._active_counts[-1] if self._active_counts else 0
        stopped_count = len(self._stopped_track_ids)
        stopped_ratio = round(stopped_count / active_count, 3) if active_count > 0 else 0.0
        queue_indicator = round(stopped_ratio * min(density / 10.0, 1.0), 3)
        queue_indicator = max(0.0, min(queue_indicator, 1.0))

        return {
            "window_started_s": round(self._window_start_s, 3),
            "window_ended_s": round(end_s, 3),
            "window_duration_s": round(duration, 3),
            "window_complete": complete,
            "unique_vehicle_count": len(self._window_ids),
            "average_active_vehicles": avg_active,
            "peak_active_vehicles": max(self._active_counts),
            "vehicle_flow_per_hour": round(len(self._window_ids) / duration * 3600, 2),
            "vehicle_density_per_100m": density,
            "traffic_state": self._congestion_state(density),
            "average_speed_kmh": speed,
            "speed_calibrated": self.pixels_per_meter is not None,
            "speed_reliable": self.pixels_per_meter is not None,
            "vehicle_breakdown": vehicle_breakdown,
            "heavy_vehicle_ratio": heavy_vehicle_ratio,
            "stopped_vehicle_count": stopped_count,
            "stopped_vehicle_ratio": stopped_ratio,
            "queue_indicator": queue_indicator,
            "peak_hour_flag": self._is_peak_hour(end_s),
            "frames_processed": self._frames_processed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def update(self, tracks: list[TrackObservation], timestamp_s: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Add a frame and return live metrics plus any completed time window."""
        completed: list[dict[str, Any]] = []
        if self._window_start_s is None:
            self._reset_window(timestamp_s)
        elif timestamp_s - self._window_start_s >= self.window_seconds:
            finished = self._build_window(timestamp_s, complete=True)
            if finished:
                completed.append(finished)
            self._reset_window(timestamp_s)

        frame_ids = {track.track_id for track in tracks}

        for track in tracks:
            previous = self._last_positions.get(track.track_id)
            if previous:
                (previous_x, previous_y), previous_time = previous
                elapsed = timestamp_s - previous_time
                if elapsed > 0:
                    displacement = float(np.hypot(
                        track.centroid[0] - previous_x,
                        track.centroid[1] - previous_y,
                    ))
                    if displacement <= self.stopped_vehicle_threshold_px:
                        self._consecutive_low_motion[track.track_id] = (
                            self._consecutive_low_motion.get(track.track_id, 0) + 1
                        )
                    else:
                        self._consecutive_low_motion[track.track_id] = 0
                        self._stopped_track_ids.discard(track.track_id)

        for track in tracks:
            if self._consecutive_low_motion.get(track.track_id, 0) >= self.stopped_vehicle_min_frames:
                self._stopped_track_ids.add(track.track_id)

        for track_id in list(self._stopped_track_ids):
            if track_id not in frame_ids:
                self._stopped_track_ids.discard(track_id)
                self._consecutive_low_motion.pop(track_id, None)

        self._all_seen_ids.update(frame_ids)
        self._window_ids.update(frame_ids)
        self._window_breakdown.update(track.label for track in tracks)
        self._active_counts.append(len(frame_ids))
        self._frames_processed += 1

        if self.pixels_per_meter:
            for track in tracks:
                previous = self._last_positions.get(track.track_id)
                if previous:
                    (previous_x, previous_y), previous_time = previous
                    elapsed = timestamp_s - previous_time
                    if elapsed > 0:
                        pixels = float(np.hypot(track.centroid[0] - previous_x, track.centroid[1] - previous_y))
                        kmh = pixels / self.pixels_per_meter / elapsed * 3.6
                        if 0 <= kmh <= 250:
                            self._speeds_kmh.append(kmh)

        for track in tracks:
            self._last_positions[track.track_id] = (track.centroid, timestamp_s)

        live = self._compute_live_features(tracks, timestamp_s, frame_ids)
        return live, completed

    def flush(self, timestamp_s: float) -> dict[str, Any] | None:
        """Emit the in-progress window at the end of a bounded video run."""
        return self._build_window(timestamp_s, complete=False)


class YoloByteTrackPipeline:
    """Run vehicle-only YOLOv8 detection, ByteTrack IDs, and aggregation."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        window_seconds: int = 300,
        pixels_per_meter: float | None = None,
        road_segment_m: float = 100.0,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self._model: Any | None = None
        self.aggregator = TrafficFeatureAggregator(window_seconds, pixels_per_meter, road_segment_m)

    def _get_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        return self._model

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp_s: float
    ) -> tuple[np.ndarray, dict[str, Any], list[TrackObservation], list[dict[str, Any]]]:
        result = self._get_model().track(
            frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASS_IDS),
            conf=self.confidence,
            verbose=False,
        )[0]

        tracks: list[TrackObservation] = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            for bbox, class_id, confidence, track_id in zip(xyxy, class_ids, confidences, track_ids):
                tracks.append(TrackObservation(
                    track_id=int(track_id),
                    label=VEHICLE_CLASS_IDS.get(int(class_id), f"class-{class_id}"),
                    confidence=round(float(confidence), 3),
                    bbox_xyxy=tuple(int(value) for value in bbox),
                ))

        live_features, completed_windows = self.aggregator.update(tracks, timestamp_s)
        return result.plot(), live_features, tracks, completed_windows
