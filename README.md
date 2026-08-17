# TrafficRisk AI Control Room

TrafficRisk AI is a local, demo-ready traffic monitoring and decision-support system. It processes traffic video with YOLOv8 and ByteTrack, aggregates traffic features, predicts risk, creates transparent rule-based response recommendations, persists events locally, and displays the results in Streamlit and React control-room dashboards.

> **Safety note:** This is a decision-support demo. It does not control signals, dispatch police, contact emergency services, or make autonomous real-world decisions.

## What is complete

All seven core pipeline stages are implemented:

1. **CCTV / video input** — uploaded MP4, AVI, MOV, MKV, and RTSP/CCTV URLs.
2. **YOLOv8 vehicle detection** — cars, motorcycles, buses, and trucks.
3. **ByteTrack tracking** — persistent vehicle IDs across frames.
4. **Traffic feature aggregation** — windowed counts, density, flow, traffic state, vehicle types, and calibrated speed when available.
5. **Risk classifier** — Random Forest risk score, level, confidence proxy, explanation, and transparency warnings for imputed context.
6. **Response Engine** — transparent LOW/MEDIUM/HIGH/CRITICAL rule-based recommendations.
7. **SQLite Event Store + local API/WebSocket** — persistent feature, risk, and response events with live dashboard updates.

## Dashboards

### React Control Room

URL: `http://localhost:5173`

The React/Vite/Tailwind/Framer Motion dashboard provides:

- Reference-style dark control-room layout with sidebar navigation.
- KPI ribbon: junction count, vehicles, high-risk junctions, alerts, deployments, and calibrated speed.
- Four-junction control room cards.
- Traffic-risk map and risk heatmap presentation.
- Active-alert and response/deployment panels.
- CCTV-style monitoring tiles.
- Traffic analytics, risk analysis, response history, Nagpur risk map, and system/API status screens.
- API polling plus a direct WebSocket connection to new persisted events.
- A clearly labelled **Demo Mode** for presentations.

The React control room uses actual API/WebSocket event data as it becomes available. If no live event exists, it shows no live data. Demo Mode uses deliberately simulated Junction 1–4 scenarios and is visibly labelled; it is not CCTV data.

### Streamlit Vision Pipeline

URL: `http://localhost:8501`

The Streamlit application remains the working vision-pipeline console. Use it to upload a traffic video or provide an RTSP/CCTV URL, run YOLOv8 + ByteTrack, configure aggregation/calibration, inspect feature windows and risk output, and persist events used by the React dashboard.

## Local services

| Service | URL | Purpose |
| --- | --- | --- |
| React control room | `http://localhost:5173` | Primary demo dashboard |
| Streamlit vision console | `http://localhost:8501` | Video/RTSP processing workflow |
| Local event API | `http://127.0.0.1:8502` | Event and junction data |
| WebSocket stream | `ws://127.0.0.1:8502/ws/events` | New persisted event notifications |

### API endpoints

- `GET /health` — local service health and persisted event count.
- `GET /events?limit=100` — newest pipeline events.
- `GET /events/latest` — most recent pipeline event.
- `GET /junctions` — existing 705-junction Nagpur risk dataset.
- `GET /junctions?risk_category=HIGH` — category-filtered historical junction data.
- `WS /ws/events` — receives new SQLite event records as they are written.

The API is local only. CORS is limited to the Vite development origins on port 5173.

## Demo flow

### Presentation-only flow

1. Open `http://localhost:5173`.
2. Leave **Demo Mode** on to see the four explicit simulated junction scenarios:
   - Junction 1: LOW
   - Junction 2: MEDIUM
   - Junction 3: HIGH
   - Junction 4: CRITICAL
3. Explain the visual pipeline from camera input through the Event Store, API, WebSocket, and dashboard.
4. Open the Alerts page to show the rule-based decision-support presentation.

### Real local-pipeline flow

1. Open the Streamlit Vision Console.
2. Upload a supported traffic video or enter a reachable RTSP/CCTV URL.
3. Start `YOLOv8 + ByteTrack`.
4. At the end of a traffic-feature window, the system produces:
   - `traffic.feature_window.v1`
   - `traffic.risk_prediction.v1`
   - `traffic.response_recommendation.v1`
5. All events are stored in `data/trafficrisk_events.db`.
6. The API and WebSocket expose the event updates.
7. Refresh or watch the React control room: actual cards, analytics, risk analysis, alert history, and the live event feed update from stored events.

## Speed accuracy

Speed is shown in km/h **only** after a camera calibration value (pixels per metre) has been supplied in the Streamlit feature-aggregation settings. Otherwise the dashboard shows that speed is unavailable rather than converting pixel movement into a real-world speed.

## Run locally

### 1. Python environment

Install the project Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 2. Start the Streamlit vision console

```powershell
python -m streamlit run app.py --server.port 8501
```

### 3. Start the API and WebSocket service

```powershell
python -m uvicorn api_server:app --host 127.0.0.1 --port 8502
```

### 4. Start the React dashboard

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

For a production frontend bundle:

```powershell
cd frontend
npm.cmd run build
```

## Important files

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI and pipeline orchestration |
| `vision_pipeline.py` | YOLOv8, ByteTrack, and feature aggregation |
| `risk_classifier.py` | Historical-data Random Forest risk scoring |
| `response_engine.py` | Transparent rule-based recommendations |
| `pipeline_contracts.py` | Versioned feature, risk, and response contracts |
| `event_store.py` | SQLite event persistence |
| `api_server.py` | Starlette HTTP API, CORS, and WebSocket stream |
| `frontend/src/App.jsx` | React control-room dashboard |
| `frontend/src/styles.css` | Control-room visual system and responsive styles |
| `data/traffic_risk_data.csv` | Historical Nagpur junction risk data |

## Verification performed

- `python -m compileall .`
- Response Engine LOW/MEDIUM/HIGH/CRITICAL checks.
- SQLite event-store round-trip checks.
- API route-registration checks.
- React production build with `npm.cmd run build`.
- HTTP 200 checks for React dashboard, Streamlit dashboard, API health, and the 705-junction API endpoint.
- CORS verification from the React local origin.

## What works in demo

- Full seven-stage pipeline explanation.
- Four-junction visual operations view, clearly marked as Demo Mode when simulated.
- Visual risk map, heatmap, KPI cards, alerts, deployment recommendations, and CCTV-style tiles.
- Historical 705-junction Nagpur map data.
- Actual local event persistence, API retrieval, and WebSocket update delivery.
- Real local video/RTSP processing through the Streamlit Vision Console.
- Downloadable response-history CSV when response events exist.

## Integrations still pending for production

The project intentionally does **not** yet include:

- A real authenticated CCTV/VMS integration and secure RTSP credential management.
- A production message broker, background worker, or 24/7 stream processor.
- PostgreSQL/cloud storage, migrations, retention policies, backups, or multi-user access control.
- Public API authentication, rate limiting, TLS, monitoring, audit logging, or deployment infrastructure.
- A map-provider integration with roads, geocoding, traffic layers, or production GIS controls.
- Direct police, emergency-service, traffic-signal, or communications-system integration.
- Live pedestrian, weather, crash, roadwork, and other context feeds. Missing context is visible in the model’s imputation warnings.
- Model governance: field validation, bias/error analysis, drift monitoring, retraining process, and human operational approval workflow.

## Production next steps

1. Move event persistence from local SQLite to managed PostgreSQL.
2. Run vision processing as authenticated background workers, not inside a browser session.
3. Add secure camera/VMS integrations and a real-time queue or broker.
4. Add user authentication, roles, API security, logging, metrics, and alerts.
5. Validate calibration and risk-model performance using verified field data.
6. Require trained human review for every operational recommendation.

## Limitations

- The local WebSocket polls the SQLite store once per second; it is appropriate for the demo but not a high-throughput production event bus.
- The React CCTV tiles are presentation/status components, not direct camera players. Real video processing remains in Streamlit.
- Demo Mode does not write fake events to SQLite and is not represented as live data.
- The existing risk dataset is historical and should not be treated as a real-time citywide operational feed.
