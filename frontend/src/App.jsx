import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'


/* ─────────────────────────── constants ────────────────────────────── */
// Use the dashboard host by default. Hard-coding 127.0.0.1 makes a LAN demo
// appear offline because every viewer's browser calls its own machine.
const dashboardHost = window.location.hostname || '127.0.0.1'
const apiProtocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
const apiOrigin = import.meta.env.VITE_API_ORIGIN || `${apiProtocol}//${dashboardHost}:8502`
const wsOrigin = import.meta.env.VITE_WS_ORIGIN || `${wsProtocol}//${dashboardHost}:8502/ws/events`
const NAGPUR_CENTER = { lat: 21.1458, lng: 79.0882 }
const MAPS_API_KEY   = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || ''

const navItems = [
  ['control',  'Control Room',      '⬡'],
  ['dispatch', 'Dispatch Center',   '🚨'],
  ['map',      'Live Risk Map',     '◉'],
  ['redeploy', 'Redeployment',      '⚔'],
  ['allocation','Officer Allocation', '⛨'],
  ['traffic',  'Junction Status',   '◈'],
  ['analytics','Analytics',         '▣'],
  ['risk',     'Risk Analysis',     '▲'],
  ['system',   'System Status',     '⊞'],
]

const RISK_COLORS = { LOW:'#42b955', MEDIUM:'#f7c42f', HIGH:'#f29122', CRITICAL:'#ee4b50' }
const RISK_MARKER_COLORS = { LOW:'#22c55e', MEDIUM:'#eab308', HIGH:'#f97316', CRITICAL:'#ef4444' }

/* ── Risk scoring thresholds & mappings ── */
const RISK_THRESHOLDS = { LOW: 0.40, MEDIUM: 0.70, HIGH: 0.85 } // risk_score 0-1 scale
const DOT_COLORS  = { LOW: '#22c55e', MEDIUM: '#eab308', HIGH: '#f97316', CRITICAL: '#ef4444' }
const DOT_STROKE  = { LOW: '#16a34a', MEDIUM: '#ca8a04', HIGH: '#ea580c', CRITICAL: '#b91c1c' }
const DOT_SCALE   = { LOW: 5,         MEDIUM: 6,         HIGH: 7,         CRITICAL: 8         }
const DOT_ZINDEX  = { LOW: 2,         MEDIUM: 3,         HIGH: 5,         CRITICAL: 6         }

const riskFromScore = (score) => {
  const s = parseFloat(score) || 0
  if (s >= RISK_THRESHOLDS.HIGH)   return 'CRITICAL'
  if (s >= RISK_THRESHOLDS.MEDIUM) return 'HIGH'
  if (s >= RISK_THRESHOLDS.LOW)    return 'MEDIUM'
  return 'LOW'
}
const resolveLevel = (j) => {
  if (j.risk_category) return j.risk_category.toUpperCase()
  if (j.level)         return j.level.toUpperCase()
  return riskFromScore(j.risk_score ?? j.score ?? 0)
}

/* ─────────────────────────── utilities ────────────────────────────── */
const time = (ts) => ts ? new Date(ts).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}) : '—'

const levelClass = (l='') => ({
  LOW:'level-low', MEDIUM:'level-medium', HIGH:'level-high', CRITICAL:'level-critical',
  low:'level-low', medium:'level-medium', high:'level-high', critical:'level-critical',
  info:'level-info', purple:'level-purple',
}[l] || 'level-low')

const niceType = (t='') => t.replace(/^traffic\./,'').replace(/\.v\d+$/,'').replace(/_/g,' ')

const eventFacts = (event) => {
  const p = event?.payload || event?.response || event?.redeployment || {}
  if (event?.event_type?.includes('response')) return event?.response || p
  if (event?.event_type?.includes('risk'))     return event?.prediction || p
  if (event?.event_type?.includes('redeployment')) return event?.redeployment || p
  return p
}

/* Build live junction cards from WebSocket events */
const buildJunctions = (events, demoMode) => {
  if (demoMode) {
    return [
      { source:'Junction 1', junction_id:'1', level:'HIGH',    score:0.74, vehicles:38, density:14, flow:820, alert:'Increase coverage', demo:true, live:true },
      { source:'Junction 2', junction_id:'2', level:'CRITICAL', score:0.91, vehicles:56, density:22, flow:1100, alert:'Immediate response', demo:true, live:true },
      { source:'Junction 3', junction_id:'3', level:'MEDIUM',   score:0.52, vehicles:24, density:8,  flow:540,  alert:'Monitor', demo:true, live:true },
      { source:'Junction 4', junction_id:'4', level:'LOW',      score:0.21, vehicles:11, density:3,  flow:210,  alert:'Normal', demo:true, live:true },
    ]
  }
  const riskEvents = [...events].filter(e => e.event_type?.includes('risk_prediction'))
  riskEvents.sort((a, b) => new Date(a.recorded_at) - new Date(b.recorded_at))

  const smoothedScores = {}
  const alpha = 0.45 // EMA factor: higher = faster response, lower = smoother

  const byJunction = {}
  riskEvents.forEach(e => {
    // junction_id is the stable key — never fall back to source_name (video filename)
    const jId = e.junction_id || e.payload?.junction_id
    if (!jId) return

    // Support both the new hoisted format (prediction at root) and the old
    // payload-wrapped format (payload.prediction). Backend now hoists these
    // but old DB rows may still use the wrapped format.
    const pred = e.prediction || e.payload?.prediction || {}
    const feat = e.features   || e.payload?.features   || {}

    const currentScore = parseFloat(pred.risk_score ?? pred.score ?? e.risk_score ?? 0)

    if (smoothedScores[jId] === undefined) {
      smoothedScores[jId] = currentScore
    } else {
      smoothedScores[jId] = alpha * currentScore + (1 - alpha) * smoothedScores[jId]
    }

    const smoothedEvent = {
      ...e,
      // Always expose prediction and features at root for downstream consumers
      prediction: {
        ...pred,
        risk_score: smoothedScores[jId],
        risk_level: riskFromScore(smoothedScores[jId])
      },
      features: feat,
    }
    byJunction[jId] = smoothedEvent
  })
  return Object.values(byJunction).map(e => {
    const pred = e.prediction || {}
    const feat = e.features   || {}
    // lat/lon: root first (hoisted), then payload fallback
    const lat = e.latitude  ?? e.payload?.latitude
    const lon = e.longitude ?? e.payload?.longitude
    return {
      source:     e.source_name || e.junction_id,
      junction_id: e.junction_id || e.payload?.junction_id,
      camera_id:  e.camera_id   || e.payload?.camera_id,
      level:      pred.risk_level  || e.risk_level || 'LOW',
      score:      pred.risk_score  || e.risk_score || 0,
      vehicles:   feat.vehicle_count || feat.unique_vehicles_seen || 0,
      density:    feat.vehicle_density_per_100m || feat.window_density_per_100m || 0,
      flow:       feat.vehicle_flow_per_hour    || 0,
      speed:      feat.avg_speed_kmh ?? feat.average_speed_kmh ?? null,
      breakdown:  feat.vehicle_class_counts || feat.vehicle_breakdown || {},
      traffic_state: feat.window_traffic_state || feat.traffic_state || 'light',
      alert:      pred.explanation?.[0] || '',
      recorded_at: e.recorded_at,
      timestamp:  e.timestamp || e.recorded_at,
      latitude:   lat,
      longitude:  lon,
      live:       true,
      demo:       false,
    }
  })
}

/* Build redeployment table from events */
const buildRedeployments = (events) => {
  const rdEvents = events.filter(e => e.event_type?.includes('redeployment'))
  const byJunction = {}
  rdEvents.forEach(e => {
    const jId = e.junction_id || e.payload?.junction_id
    if (!jId) return
    if (!byJunction[jId] || new Date(e.recorded_at) > new Date(byJunction[jId].recorded_at)) {
      byJunction[jId] = e
    }
  })
  return Object.values(byJunction).map(e => {
    // Support both hoisted root format and payload-wrapped format
    const rd = e.redeployment || e.payload?.redeployment || {}
    const lat = e.latitude  ?? e.payload?.latitude
    const lon = e.longitude ?? e.payload?.longitude
    return {
      junction_id:   e.junction_id || e.payload?.junction_id,
      camera_id:     e.camera_id   || e.payload?.camera_id,
      risk_level:    e.risk_level     || rd.risk_level     || 'LOW',
      risk_score:    e.risk_score     || rd.risk_score     || 0,
      additional_officers: e.recommended_additional_officers ?? rd.recommended_additional_officers ?? 0,
      priority:      e.priority       || rd.priority       || 'LOW',
      action:        e.action         || rd.action         || 'MONITOR',
      reasons:       e.reason         || rd.reason         || [],
      trace:         e.decision_trace || rd.decision_trace || [],
      nearby:        e.nearby_redeployments || rd.nearby_redeployments || [],
      demo_notice:   e.demo_notice    || rd.demo_notice    || '',
      human_approval_required: e.human_approval_required ?? true,
      recorded_at:   e.recorded_at,
      latitude:      lat,
      longitude:     lon,
    }
  })
}

/* ─────────────────────── small components ─────────────────────────── */
const RiskBadge = ({ level }) => (
  <span className={`badge ${levelClass(level)}`}>{level || 'LOW'}</span>
)

const Empty = ({ children }) => (
  <div className="empty-state"><span>⊘</span><p>{children}</p></div>
)

const StatusPill = ({ label, online, detail }) => (
  <span className={`pill ${online ? 'pill-on' : 'pill-off'}`}>
    <i/>{label}: {detail}
  </span>
)


const isValidCoord = (lat, lng) => {
  const la = parseFloat(lat), lo = parseFloat(lng)
  return !isNaN(la) && !isNaN(lo) && la !== 0 && lo !== 0
    && la >= -90 && la <= 90 && lo >= -180 && lo <= 180
}
/* ─── Canvas-based heatmap overlay (replaces deprecated HeatmapLayer) ───────
 *
 *  Uses google.maps.OverlayView + an <canvas> element to render a pure-JS
 *  Gaussian kernel density estimate coloured by risk level.
 *  Works with any current Google Maps version — no extra library required.
 */
function createCanvasHeatOverlay(map) {
  // Risk colour stops: LOW → MEDIUM → HIGH → CRITICAL
  const COLOUR_STOPS = [
    [0.00, [34,  197, 94,  0  ]],   // transparent green
    [0.20, [34,  197, 94,  120]],   // green
    [0.40, [234, 179, 8,   160]],   // yellow
    [0.65, [249, 115, 22,  200]],   // orange
    [0.85, [239, 68,  68,  220]],   // red
    [1.00, [185, 28,  28,  240]],   // deep red
  ]

  function lerpColour(t) {
    let lo = COLOUR_STOPS[0], hi = COLOUR_STOPS[COLOUR_STOPS.length - 1]
    for (let i = 0; i < COLOUR_STOPS.length - 1; i++) {
      if (t >= COLOUR_STOPS[i][0] && t <= COLOUR_STOPS[i + 1][0]) {
        lo = COLOUR_STOPS[i]; hi = COLOUR_STOPS[i + 1]; break
      }
    }
    const span = hi[0] - lo[0] || 1
    const f    = (t - lo[0]) / span
    return lo[1].map((v, i) => Math.round(v + f * (hi[1][i] - v)))
  }

  // Build a lookup table for the Gaussian kernel once (size 256×256)
  const KERNEL_SIZE = 32       // px radius of each splat
  const kernel = new Float32Array((KERNEL_SIZE * 2 + 1) ** 2)
  const sigma  = KERNEL_SIZE / 3
  const kd     = KERNEL_SIZE * 2 + 1
  for (let y = 0; y < kd; y++) {
    for (let x = 0; x < kd; x++) {
      const dx = x - KERNEL_SIZE, dy = y - KERNEL_SIZE
      kernel[y * kd + x] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    }
  }

  let _points  = []    // [{lat, lng, weight, level}]
  let _visible = false
  let _canvas  = null
  let _div     = null

  class CanvasOverlay {
    constructor() { this.setMap(map) }
    onAdd() {
      _div    = document.createElement('div')
      _canvas = document.createElement('canvas')
      _div.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none'
      _canvas.style.cssText = 'position:absolute;top:0;left:0'
      _div.appendChild(_canvas)
      const panes = this.getPanes()
      panes.overlayLayer.appendChild(_div)
    }
    draw() {
      if (!_div || !_canvas) return
      const proj = this.getProjection()
      if (!proj) return

      const bounds = map.getBounds()
      if (!bounds) return
      const sw = proj.fromLatLngToDivPixel(bounds.getSouthWest())
      const ne = proj.fromLatLngToDivPixel(bounds.getNorthEast())
      const W  = Math.round(ne.x - sw.x)
      const H  = Math.round(sw.y - ne.y)
      if (W <= 0 || H <= 0) return

      _canvas.width  = W
      _canvas.height = H
      _canvas.style.left = sw.x + 'px'
      _canvas.style.top  = ne.y + 'px'

      const ctx = _canvas.getContext('2d')
      ctx.clearRect(0, 0, W, H)

      if (!_visible || _points.length === 0) return

      // Density accumulation buffer
      const buf = new Float32Array(W * H)
      let maxVal = 0

      for (const pt of _points) {
        const pixel = proj.fromLatLngToDivPixel(
          new window.google.maps.LatLng(pt.lat, pt.lng)
        )
        const px = Math.round(pixel.x - sw.x)
        const py = Math.round(pixel.y - ne.y)
        const w  = pt.weight        // 0..1

        // Splat the Gaussian kernel into the buffer
        for (let ky = 0; ky < kd; ky++) {
          const iy = py - KERNEL_SIZE + ky
          if (iy < 0 || iy >= H) continue
          for (let kx = 0; kx < kd; kx++) {
            const ix = px - KERNEL_SIZE + kx
            if (ix < 0 || ix >= W) continue
            const val = kernel[ky * kd + kx] * w
            buf[iy * W + ix] += val
            if (buf[iy * W + ix] > maxVal) maxVal = buf[iy * W + ix]
          }
        }
      }

      if (maxVal === 0) return

      // Render buffer → ImageData using colour lookup
      const imgData = ctx.createImageData(W, H)
      const data    = imgData.data
      for (let i = 0; i < buf.length; i++) {
        const t    = buf[i] / maxVal       // 0..1
        if (t < 0.02) continue            // skip near-zero
        const [r, g, b, a] = lerpColour(t)
        data[i * 4]     = r
        data[i * 4 + 1] = g
        data[i * 4 + 2] = b
        data[i * 4 + 3] = Math.round(a * t)
      }
      ctx.putImageData(imgData, 0, 0)
    }
    onRemove() {
      if (_div?.parentNode) _div.parentNode.removeChild(_div)
      _div = null; _canvas = null
    }
    setPoints(pts) { _points = pts || []; this.draw() }
    setVisible(v)  {
      _visible = !!v
      if (_canvas) _canvas.style.display = v ? '' : 'none'
      this.draw()
    }
  }
  // Set up inheritance from google.maps.OverlayView without reassigning the
  // read-only `prototype` property (class declarations make it non-writable).
  Object.setPrototypeOf(CanvasOverlay.prototype, window.google.maps.OverlayView.prototype)
  CanvasOverlay.prototype.constructor = CanvasOverlay

  // Redraw on map movement
  const overlay = new CanvasOverlay()
  window.google.maps.event.addListener(map, 'bounds_changed', () => overlay.draw())
  return overlay
}

/* ─────────────────────── GOOGLE MAP COMPONENT ──────────────────────── */
function GoogleMapPanel({ junctionCards, historicalJunctions, demoMode, redeployments }) {
  const mapRef       = useRef(null)
  const mapObj       = useRef(null)
  const markersDB    = useRef({})    // all 705 DB junction markers
  const markersLive  = useRef({})    // live video-analysis overlay markers
  const polylines    = useRef([])
  const canvasOverlay = useRef(null)
  const infoWin      = useRef(null)
  const boundsFitted = useRef(false)

  const [mapReady,    setMapReady]    = useState(false)
  const [mapError,    setMapError]    = useState(null)
  const [loadingDots, setLoadingDots] = useState(true)
  const [stats,       setStats]       = useState({ total:0, high:0, medium:0, low:0 })
  const [mapSearch,   setMapSearch]   = useState('')
  const [mapSuggestions, setMapSuggestions] = useState([])
  const [showHeatmap, setShowHeatmap] = useState(false)

  const handleSearch = (q) => {
    setMapSearch(q)
    if (!q.trim()) {
      setMapSuggestions([])
      return
    }
    const matches = (historicalJunctions || []).filter(j => 
      String(j.junction_id).toLowerCase().includes(q.toLowerCase()) ||
      (j.location || '').toLowerCase().includes(q.toLowerCase())
    ).slice(0, 5)
    setMapSuggestions(matches)
  }

  const selectJunction = (j) => {
    setMapSearch('')
    setMapSuggestions([])
    if (!mapObj.current) return
    const pos = { lat: parseFloat(j.latitude), lng: parseFloat(j.longitude) }
    mapObj.current.setCenter(pos)
    mapObj.current.setZoom(16)

    const level = resolveLevel(j)
    const score = parseFloat(j.risk_score ?? j.score ?? 0)
    infoWin.current.setContent(buildDBInfoHTML(j, level, score))

    const marker = markersDB.current[String(j.junction_id)] || markersLive.current[String(j.junction_id)]
    if (marker) {
      infoWin.current.open(mapObj.current, marker)
      marker.setAnimation(window.google.maps.Animation.BOUNCE)
      setTimeout(() => marker.setAnimation(null), 1400)
    }
  }

  /* ── Load Google Maps script once ── */
  useEffect(() => {
    if (!MAPS_API_KEY || MAPS_API_KEY === 'YOUR_GOOGLE_MAPS_API_KEY_HERE') {
      setMapError('Add VITE_GOOGLE_MAPS_API_KEY to frontend/.env.local')
      return
    }
    const initMap = () => {
      if (!mapRef.current) return
      mapObj.current = new window.google.maps.Map(mapRef.current, {
        center: NAGPUR_CENTER,
        zoom: 12,
        mapTypeId: 'roadmap',
        mapTypeControl: true,
        fullscreenControl: true,
        streetViewControl: false,
        styles: [
          { elementType:'geometry',          stylers:[{color:'#0d1f33'}] },
          { featureType:'road',              elementType:'geometry', stylers:[{color:'#1a3a5c'}] },
          { featureType:'road.arterial',     elementType:'geometry', stylers:[{color:'#1e4976'}] },
          { featureType:'road.highway',      elementType:'geometry', stylers:[{color:'#254f82'}] },
          { featureType:'poi',               stylers:[{visibility:'off'}] },
          { featureType:'water',             elementType:'geometry', stylers:[{color:'#0a2744'}] },
          { featureType:'landscape',         elementType:'geometry', stylers:[{color:'#0c1a2e'}] },
          { elementType:'labels.text.fill',  stylers:[{color:'#7aaec8'}] },
          { elementType:'labels.text.stroke',stylers:[{color:'#081726'}] },
          { featureType:'transit',           stylers:[{visibility:'off'}] },
        ],
      })
      infoWin.current = new window.google.maps.InfoWindow({ maxWidth: 290 })
      setMapReady(true)
    }

    if (window.google?.maps) { initMap(); return }
    const existing = document.getElementById('google-maps-script')
    if (existing) { existing.addEventListener('load', initMap); return }
    const script = document.createElement('script')
    script.id     = 'google-maps-script'
    script.src    = `https://maps.googleapis.com/maps/api/js?key=${MAPS_API_KEY}`
    script.async  = true
    script.defer  = true
    script.onload  = initMap
    script.onerror = () => setMapError('Google Maps failed to load — verify API key & enabled APIs')
    document.body.appendChild(script)
  }, [])

  /* ── Render ALL 705 DB junctions as colored risk dots or hide them for Heatmap ── */
  useEffect(() => {
    if (!mapReady || !mapObj.current) return
    const google   = window.google
    const all      = (historicalJunctions || [])
    const valid    = all.filter(j => isValidCoord(j.latitude, j.longitude))

    if (valid.length === 0) { setLoadingDots(false); return }

    const targetMap = showHeatmap ? null : mapObj.current

    const dbIds = new Set(valid.map(j => String(j.junction_id)))
    Object.keys(markersDB.current).forEach(id => {
      if (!dbIds.has(id)) { 
        markersDB.current[id].setMap(null)
        delete markersDB.current[id] 
      } else {
        markersDB.current[id].setMap(targetMap)
      }
    })

    const bounds = new google.maps.LatLngBounds()
    let high = 0, medium = 0, low = 0

    valid.forEach(j => {
      const id    = String(j.junction_id)
      const level = resolveLevel(j)
      const score = parseFloat(j.risk_score ?? j.score ?? 0)
      const pos   = { lat: parseFloat(j.latitude), lng: parseFloat(j.longitude) }
      const fill  = DOT_COLORS[level] || DOT_COLORS.LOW
      const str   = DOT_STROKE[level] || DOT_STROKE.LOW
      const scale = DOT_SCALE[level]  || 5
      const zi    = DOT_ZINDEX[level] || 2

      if (level === 'CRITICAL' || level === 'HIGH') high++
      else if (level === 'MEDIUM') medium++
      else low++

      if (markersDB.current[id]) {
        markersDB.current[id].setIcon({
          path: google.maps.SymbolPath.CIRCLE,
          scale, fillColor: fill, fillOpacity: 0.92,
          strokeWeight: 1.5, strokeColor: str,
        })
        markersDB.current[id].setMap(targetMap)
      } else {
        const marker = new google.maps.Marker({
          position: pos, map: targetMap,
          icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale, fillColor: fill, fillOpacity: 0.92,
            strokeWeight: 1.5, strokeColor: str,
          },
          title: `${j.location || id} · ${level} (${(score*100).toFixed(0)})`,
          zIndex: zi,
          optimized: true,
        })
        marker.addListener('click', () => {
          infoWin.current.setContent(buildDBInfoHTML(j, level, score))
          infoWin.current.open(mapObj.current, marker)
        })
        markersDB.current[id] = marker
      }
      bounds.extend(pos)
    })

    if (!boundsFitted.current) {
      mapObj.current.fitBounds(bounds, { top:50, right:20, bottom:20, left:20 })
      boundsFitted.current = true
    }

    setStats({ total: valid.length, high, medium, low })
    setLoadingDots(false)
  }, [mapReady, historicalJunctions, showHeatmap])

  /* ── Live event overlay and Heatmap layer ── */
  useEffect(() => {
    if (!mapReady || !mapObj.current) return
    const google = window.google

    const liveIds = new Set(junctionCards.map(j => String(j.junction_id)))
    Object.keys(markersLive.current).forEach(id => {
      if (!liveIds.has(id)) { markersLive.current[id].setMap(null); delete markersLive.current[id] }
    })

    junctionCards.forEach(j => {
      if (!isValidCoord(j.latitude, j.longitude)) return
      const id    = String(j.junction_id)
      const level = resolveLevel(j)
      const fill  = DOT_COLORS[level] || DOT_COLORS.LOW
      const pos   = { lat: parseFloat(j.latitude), lng: parseFloat(j.longitude) }
      const icon  = {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 16, fillColor: fill, fillOpacity: 1,
        strokeWeight: 3, strokeColor: '#ffffff',
      }
      if (markersLive.current[id]) {
        markersLive.current[id].setPosition(pos)
        markersLive.current[id].setIcon(icon)
      } else {
        const marker = new google.maps.Marker({
          position: pos, map: mapObj.current, icon,
          title: `LIVE: ${j.junction_id} · ${level}`,
          zIndex: 20, optimized: false,
        })
        marker.addListener('click', () => {
          infoWin.current.setContent(buildLiveInfoHTML(j))
          infoWin.current.open(mapObj.current, marker)
        })
        markersLive.current[id] = marker
      }
    })

    // ── Canvas heat overlay (replaces deprecated HeatmapLayer) ──────────
    // Build point list: {lat, lng, weight, level} from DB or live junctions.
    const heatPoints = showHeatmap
      ? (historicalJunctions || [])
          .filter(j => isValidCoord(j.latitude, j.longitude))
          .map(j => ({
            lat:    parseFloat(j.latitude),
            lng:    parseFloat(j.longitude),
            weight: Math.max(0.05, parseFloat(j.risk_score ?? 0)),
            level:  resolveLevel(j),
          }))
      : junctionCards
          .filter(j => isValidCoord(j.latitude, j.longitude))
          .map(j => ({
            lat:    parseFloat(j.latitude),
            lng:    parseFloat(j.longitude),
            weight: Math.max(0.1, parseFloat(j.score ?? j.risk_score ?? 0.3)),
            level:  j.level || 'LOW',
          }))

    if (!canvasOverlay.current) {
      // Create one OverlayView per map instance; reuse on subsequent renders.
      canvasOverlay.current = createCanvasHeatOverlay(mapObj.current)
    }
    canvasOverlay.current.setPoints(heatPoints)
    // Only show the canvas when the toggle is ON
    canvasOverlay.current.setVisible(showHeatmap)

    polylines.current.forEach(p => p.setMap(null))
    polylines.current = []
    redeployments.forEach(rd => {
      if (!isValidCoord(rd.latitude, rd.longitude) || !rd.nearby?.length) return
      rd.nearby.forEach(src => {
        if (!isValidCoord(src.latitude, src.longitude)) return
        const line = new google.maps.Polyline({
          path: [
            { lat: parseFloat(src.latitude), lng: parseFloat(src.longitude) },
            { lat: parseFloat(rd.latitude),  lng: parseFloat(rd.longitude)  },
          ],
          geodesic: true, strokeColor: '#f97316', strokeOpacity: 0.85, strokeWeight: 2,
          icons: [{ icon: { path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW, scale:3, strokeColor:'#f97316' }, offset:'100%' }],
          map: mapObj.current,
        })
        polylines.current.push(line)
      })
    })
  }, [mapReady, junctionCards, redeployments, showHeatmap, historicalJunctions])

  if (mapError) return (
    <div className="map-error-panel">
      <div className="map-error-icon">⚠️</div>
      <h3>Google Maps Not Available</h3>
      <p>{mapError}</p>
      <code>frontend/.env.local → VITE_GOOGLE_MAPS_API_KEY=your_key</code>
      <p style={{marginTop:8,fontSize:'.65rem',color:'#89a8c1'}}>Enable: Maps JavaScript API in Google Cloud Console</p>
    </div>
  )

  return (
    <div className="gmap-wrapper">
      <div ref={mapRef} className="gmap-canvas" />
      {!mapReady && <div className="gmap-loading">Loading Google Maps…</div>}
      {mapReady && loadingDots && (
        <div className="gmap-loading" style={{background:'rgba(8,23,38,0.7)',color:'#53d8ff',fontSize:'.72rem',top:'auto',bottom:60,borderRadius:6,padding:'6px 14px'}}>
          Placing {(historicalJunctions||[]).length} junction markers…
        </div>
      )}

      {/* Map Search Overlay & Heatmap Switcher */}
      {mapReady && (
        <div className="gmap-search-container" style={{display:'flex', gap:'8px', width:'460px'}}>
          <input 
            type="text"
            className="gmap-search-input"
            placeholder="🔍 Search ID or location segment..."
            value={mapSearch}
            onChange={(e) => handleSearch(e.target.value)}
            style={{flex: 1}}
          />
          <button 
            onClick={() => setShowHeatmap(!showHeatmap)}
            style={{
              background: showHeatmap ? 'rgba(83,216,255,0.22)' : 'rgba(8,23,38,0.92)',
              backdropFilter: 'blur(10px)',
              border: `1px solid ${showHeatmap ? '#53d8ff' : '#1e5f8c'}`,
              borderRadius: '8px',
              color: showHeatmap ? '#53d8ff' : '#eaf5ff',
              fontSize: '.72rem',
              fontWeight: 800,
              padding: '0 14px',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
              transition: 'all .2s'
            }}
          >
            {showHeatmap ? '📡 Heatmap: ON' : '📡 Heatmap View'}
          </button>
          
          {mapSuggestions.length > 0 && (
            <div className="gmap-search-results" style={{top: 'calc(100% + 4px)', width: 'calc(100% - 130px)'}}>
              {mapSuggestions.map((j, idx) => (
                <div 
                  key={idx} 
                  className="gmap-search-item"
                  onClick={() => selectJunction(j)}
                >
                  <b style={{color:'#53d8ff'}}>{j.junction_id}</b> — {j.location || 'Nagpur segment'}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Map Legend */}
      {mapReady && (
        <div className="gmap-legend">
          <div className="gmap-legend-title">{showHeatmap ? 'Risk Density' : 'Risk Level'}</div>
          {showHeatmap ? (
            <>
              <div className="gmap-legend-row"><span style={{background:'#ef4444',boxShadow:'0 0 8px #ef4444'}}/> High Density Area</div>
              <div className="gmap-legend-row"><span style={{background:'#eab308',boxShadow:'0 0 8px #eab308'}}/> Medium Density Area</div>
              <div className="gmap-legend-row"><span style={{background:'#22c55e',boxShadow:'0 0 8px #22c55e'}}/> Low Density Area</div>
            </>
          ) : (
            <>
              <div className="gmap-legend-row"><span style={{background:DOT_COLORS.HIGH}}/> High <b>{stats.high > 0 ? `(${stats.high})` : ''}</b></div>
              <div className="gmap-legend-row"><span style={{background:DOT_COLORS.MEDIUM}}/> Medium <b>{stats.medium > 0 ? `(${stats.medium})` : ''}</b></div>
              <div className="gmap-legend-row"><span style={{background:DOT_COLORS.LOW}}/> Low <b>{stats.low > 0 ? `(${stats.low})` : ''}</b></div>
            </>
          )}
          {stats.total > 0 && <div className="gmap-legend-total">{stats.total} Nagpur junctions</div>}
          {junctionCards.length > 0 && (
            <div className="gmap-legend-row" style={{marginTop:6,borderTop:'1px solid #1e384d',paddingTop:6}}>
              <span style={{background:'#fff',width:10,height:10,border:'2px solid #53d8ff'}}/> Live feed ({junctionCards.length})
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* Info popup for DB junction */
const buildDBInfoHTML = (j, level, score) => `
  <div style="background:#0d1f33;color:#eaf5ff;padding:12px 14px;border-radius:10px;font-family:system-ui,sans-serif;min-width:230px;max-width:290px;border:1px solid ${DOT_COLORS[level]||'#22c55e'}55">
    <div style="font-size:.88rem;font-weight:700;color:#53d8ff;margin-bottom:6px;line-height:1.3">${j.location || j.junction_id}</div>
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
      <span style="background:${DOT_COLORS[level]||'#22c55e'};color:#fff;font-size:.6rem;font-weight:800;padding:2px 8px;border-radius:10px;letter-spacing:.05em">${level}</span>
      <span style="color:#bacbd8;font-size:.7rem">Risk Score: <b>${(score*100).toFixed(0)}/100</b></span>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:.65rem">
      <tr><td style="color:#89a8c1;padding:2px 0">Junction ID</td><td style="color:#eaf5ff;font-weight:600">${j.junction_id}</td></tr>
      <tr><td style="color:#89a8c1;padding:2px 0">Camera ID</td><td style="color:#eaf5ff">${j.camera_id || '—'}</td></tr>
      <tr><td style="color:#89a8c1;padding:2px 0">Zone</td><td style="color:#eaf5ff">${j.zone || '—'}</td></tr>
      <tr><td style="color:#89a8c1;padding:2px 0">Latitude</td><td style="color:#eaf5ff">${parseFloat(j.latitude).toFixed(5)}</td></tr>
      <tr><td style="color:#89a8c1;padding:2px 0">Longitude</td><td style="color:#eaf5ff">${parseFloat(j.longitude).toFixed(5)}</td></tr>
    </table>
    ${j.risk_reason ? `<div style="margin-top:7px;padding:5px 8px;background:#0a1928;border-radius:5px;font-size:.62rem;color:#f7c42f;border:1px solid #f7c42f33">${j.risk_reason}</div>` : ''}
  </div>
`

/* Info popup for live video analysis event */
const buildLiveInfoHTML = (j) => {
  const level = resolveLevel(j)
  return `
    <div style="background:#0d1f33;color:#eaf5ff;padding:12px 14px;border-radius:10px;font-family:system-ui,sans-serif;min-width:230px;border:2px solid ${DOT_COLORS[level]||'#22c55e'}">
      <div style="font-size:.72rem;color:#45dd94;font-weight:700;margin-bottom:4px">● LIVE ANALYSIS</div>
      <div style="font-size:.88rem;font-weight:700;color:#53d8ff;margin-bottom:6px">${j.junction_id} — ${j.source || ''}</div>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
        <span style="background:${DOT_COLORS[level]};color:#fff;font-size:.6rem;font-weight:800;padding:2px 8px;border-radius:10px">${level}</span>
        <span style="color:#bacbd8;font-size:.7rem">Score: ${Number(j.score??j.risk_score??0).toFixed(3)}</span>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:.65rem">
        <tr><td style="color:#89a8c1;padding:2px 0">Vehicles</td><td style="color:#eaf5ff">${j.vehicles || '—'}</td></tr>
        <tr><td style="color:#89a8c1;padding:2px 0">Density</td><td style="color:#eaf5ff">${Number(j.density||0).toFixed(1)}/100m</td></tr>
        <tr><td style="color:#89a8c1;padding:2px 0">Flow</td><td style="color:#eaf5ff">${Math.round(j.flow||0)}/hr</td></tr>
        <tr><td style="color:#89a8c1;padding:2px 0">Camera</td><td style="color:#eaf5ff">${j.camera_id||'—'}</td></tr>
        <tr><td style="color:#89a8c1;padding:2px 0">Updated</td><td style="color:#eaf5ff">${new Date(j.recorded_at||Date.now()).toLocaleTimeString()}</td></tr>
      </table>
    </div>
  `
}

const buildInfoWindowHTML = buildLiveInfoHTML



/* ─────────────── Dynamic Redeployment Panel ───────────────────────── */
function RedeploymentPage({ redeployments, junctionCards, wsOnline }) {
  const totalOfficers = redeployments.reduce((s,r) => s + (r.additional_officers||0), 0)
  const urgent = redeployments.filter(r => r.priority === 'URGENT').length
  const [expanded, setExpanded] = useState(null)

  return (
    <div className="page-grid">
      <section className="full-section">
        <p className="overline">Decision support — Human approval required</p>
        <h2>Dynamic Officer Redeployment Recommendations</h2>
        <div className="ref-kpi-ribbon" style={{marginTop:12}}>
          <article className="ref-kpi red"><i>▲</i><div><span>Urgent</span><b>{urgent}</b><small>Junctions</small></div></article>
          <article className="ref-kpi amber"><i>♟</i><div><span>Total Additional Officers</span><b>{totalOfficers}</b><small>Recommended</small></div></article>
          <article className="ref-kpi blue"><i>◈</i><div><span>Active Junctions</span><b>{junctionCards.length}</b><small>Being monitored</small></div></article>
        </div>
        <div style={{marginTop:16,padding:'8px 12px',background:'#1c2d3d',border:'1px solid #2b4a6a',borderRadius:8,color:'#f7c42f',fontSize:'.7rem',fontWeight:700,letterSpacing:'.04em'}}>
          ⚠️ DEMO/RULE-BASED RECOMMENDATIONS — Human approval required before any actual officer movement. No real dispatch is occurring.
        </div>
        
        {/* Incident Simulation Control */}
        <div style={{
          marginTop:12, padding:'14px 18px', background:'rgba(238,75,80,0.06)', 
          border:'1px solid #7c242e', borderRadius:8, display:'flex', 
          justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:10
        }}>
          <div>
            <b style={{color:'#ee4b50', fontSize:'.75rem', display:'block'}}>💥 Simulated Nagpur Traffic Incident Injector</b>
            <span style={{color:'#bacbd8', fontSize:'.62rem'}}>
              Injects a critical double-vehicle collision event at Chhatrapati Square (J004) to demonstrate real-time automatic police dispatch and officer redeployment.
            </span>
          </div>
          <button 
            onClick={async () => {
              const incidentEvent = {
                event_id: `INCIDENT-${Date.now()}`,
                event_type: "traffic.risk_prediction.v1",
                source_name: "Nagpur Traffic Incident Sensor",
                recorded_at: new Date().toISOString(),
                junction_id: "J004",
                camera_id: "CAM-004",
                prediction: {
                  risk_score: 0.95,
                  risk_level: "CRITICAL",
                  explanation: ["Major double-vehicle collision in intersection", "Extreme congestion backlog spreading to adjacent lanes"]
                },
                features: {
                  vehicle_count: 42,
                  vehicle_density_per_100m: 24.5,
                  vehicle_flow_per_hour: 1520,
                  window_traffic_state: "congested",
                  avg_speed_kmh: 8.5
                },
                latitude: 21.1082,
                longitude: 79.0673
              }
              try {
                await fetch(`${apiOrigin}/events`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(incidentEvent)
                })
              } catch (e) {
                console.error("Simulation failed", e)
              }
            }}
            style={{
              padding:'8px 16px', background:'#7c242e', border:'1px solid #ee4b50', 
              borderRadius:6, color:'#ffb4bd', fontSize:'.7rem', fontWeight:800, cursor:'pointer'
            }}
          >
            🚨 Inject Simulated Accident
          </button>
        </div>
      </section>

      {redeployments.length === 0 ? (
        <section className="full-section">
          <Empty>No redeployment recommendations yet. Process a video with HIGH or CRITICAL risk to see recommendations.</Empty>
        </section>
      ) : redeployments.map((rd, i) => (
        <section key={`${rd.junction_id}-${i}`} className="full-section" style={{marginBottom:0}}>
          <div className={`redeploy-card ${levelClass(rd.risk_level)}`}>
            <div className="redeploy-head">
              <div>
                <b>{rd.junction_id}</b>
                <RiskBadge level={rd.risk_level} />
                <span className="redeploy-priority" data-priority={rd.priority}>{rd.priority}</span>
              </div>
              <div style={{display:'flex',gap:12,alignItems:'center'}}>
                <span style={{color:'#ffd067',fontWeight:800,fontSize:'.8rem'}}>+{rd.additional_officers} officers recommended</span>
                <span style={{color:'#89a8c1',fontSize:'.65rem'}}>Updated {time(rd.recorded_at)}</span>
                <button className="trace-btn" onClick={() => setExpanded(expanded===i ? null : i)}>
                  {expanded===i ? 'Hide trace ▲' : 'Decision trace ▼'}
                </button>
              </div>
            </div>
            <div className="redeploy-body">
              <div>
                <b style={{color:'#bacbd8',fontSize:'.65rem',display:'block',marginBottom:4}}>ACTION</b>
                <span style={{color:'#53d8ff',fontWeight:700,fontSize:'.75rem'}}>{rd.action}</span>
              </div>
              <div>
                <b style={{color:'#bacbd8',fontSize:'.65rem',display:'block',marginBottom:4}}>RISK SCORE</b>
                <span style={{color:'#f7c42f',fontWeight:700,fontSize:'.75rem'}}>{Number(rd.risk_score).toFixed(3)}</span>
              </div>
              <div>
                <b style={{color:'#bacbd8',fontSize:'.65rem',display:'block',marginBottom:4}}>CAMERA</b>
                <span style={{fontWeight:700,fontSize:'.75rem'}}>{rd.camera_id || '—'}</span>
              </div>
              <div>
                <b style={{color:'#bacbd8',fontSize:'.65rem',display:'block',marginBottom:4}}>STATUS</b>
                <span style={{color:'#f29122',fontWeight:700,fontSize:'.75rem'}}>⏳ Awaiting human approval</span>
              </div>
            </div>
            {rd.reasons?.length > 0 && (
              <div className="redeploy-reasons">
                {rd.reasons.map((r,j) => <span key={j} className="reason-tag">• {r}</span>)}
              </div>
            )}
            {expanded===i && (
              <div className="redeploy-trace">
                <b style={{color:'#53d8ff',fontSize:'.65rem',letterSpacing:'.06em'}}>DECISION TRACE</b>
                {rd.trace.map((t,j) => <p key={j} style={{margin:'4px 0',color:'#bacbd8',fontSize:'.65rem'}}>→ {t}</p>)}
                {rd.nearby?.length > 0 && (
                  <>
                    <b style={{color:'#f7c42f',fontSize:'.65rem',letterSpacing:'.06em',marginTop:8,display:'block'}}>SUGGESTED REDEPLOYMENT SOURCES</b>
                    {rd.nearby.map((src,j) => (
                      <div key={j} style={{background:'#0c1e2f',border:'1px solid #1e384d',borderRadius:4,padding:'4px 8px',margin:'4px 0',fontSize:'.63rem'}}>
                        <b style={{color:'#42b955'}}>{src.from_junction_id}</b>
                        <span style={{color:'#bacbd8'}}> — {src.from_junction_name} ({src.distance_km} km)</span>
                        <span style={{color:'#89a8c1'}}> · {src.historical_risk} risk · {src.officers_present} officers present</span>
                      </div>
                    ))}
                  </>
                )}
                <small style={{color:'#89a8c1',marginTop:8,display:'block'}}>{rd.demo_notice}</small>
              </div>
            )}
          </div>
        </section>
      ))}
    </div>
  )
}

/* ─────────────── Dispatch Center ──────────────────────────────────── */
const STATUS_META = {
  PENDING:     { color:'#f7c42f', bg:'#3a3210', border:'#7c6824', icon:'⏳', label:'Pending' },
  DISPATCHED:  { color:'#53d8ff', bg:'#0d2a3a', border:'#1e5f8c', icon:'🚔', label:'Dispatched' },
  ON_SCENE:    { color:'#42b955', bg:'#1a3a24', border:'#24703a', icon:'✅', label:'On Scene' },
  RESOLVED:    { color:'#89a8c1', bg:'#0a1522', border:'#1e384d', icon:'☑️', label:'Resolved' },
  CANCELLED:   { color:'#ee4b50', bg:'#3d1c24', border:'#7c242e', icon:'✖️', label:'Cancelled' },
}
const PRIORITY_META = {
  URGENT: { color:'#ee4b50', bg:'#3d1c24', border:'#7c242e' },
  HIGH:   { color:'#f29122', bg:'#3a2210', border:'#7c4e24' },
  MEDIUM: { color:'#f7c42f', bg:'#3a3210', border:'#7c6824' },
  LOW:    { color:'#42b955', bg:'#1a3a24', border:'#24703a' },
}

function DispatchCenter({ dispatches, wsOnline, apiOnline }) {
  const [filter, setFilter] = useState('ALL')
  const [manualJunctionId, setManualJunctionId] = useState('')
  const [manualOfficers, setManualOfficers]     = useState(2)
  const [submitting, setSubmitting]             = useState(false)
  const [flashNew, setFlashNew]                 = useState(null)

  // Play an alert ping when a new PENDING urgent dispatch arrives
  useEffect(() => {
    if (!dispatches.length) return
    const latest = dispatches[0]
    if (latest?.status === 'PENDING' && latest?.priority === 'URGENT') {
      setFlashNew(latest.dispatch_id)
      const t = setTimeout(() => setFlashNew(null), 3500)
      return () => clearTimeout(t)
    }
  }, [dispatches[0]?.dispatch_id])

  const shown = useMemo(() => {
    if (filter === 'ALL') return dispatches
    return dispatches.filter(d => d.status === filter)
  }, [dispatches, filter])

  const stats = useMemo(() => ({
    total:      dispatches.length,
    pending:    dispatches.filter(d => d.status === 'PENDING').length,
    active:     dispatches.filter(d => ['PENDING','DISPATCHED','ON_SCENE'].includes(d.status)).length,
    resolved:   dispatches.filter(d => d.status === 'RESOLVED').length,
    urgent:     dispatches.filter(d => d.priority === 'URGENT').length,
  }), [dispatches])

  const updateStatus = async (dispatch_id, newStatus) => {
    try {
      await fetch(`${apiOrigin}/dispatch/${dispatch_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type':'application/json' },
        body: JSON.stringify({ status: newStatus }),
      })
    } catch(e) { console.error('Dispatch update failed', e) }
  }

  const manualDispatch = async () => {
    if (!manualJunctionId.trim()) return
    setSubmitting(true)
    try {
      await fetch(`${apiOrigin}/dispatch`, {
        method: 'POST',
        headers: { 'Content-Type':'application/json' },
        body: JSON.stringify({
          junction_id:   manualJunctionId.trim(),
          officers_sent: manualOfficers,
          risk_level:    'HIGH',
          priority:      'HIGH',
          reason:        'Manual dispatch by control room operator',
          triggered_by:  'manual',
          status:        'PENDING',
        }),
      })
      setManualJunctionId('')
    } catch(e) { console.error('Manual dispatch failed', e) }
    setSubmitting(false)
  }

  return (
    <div className="page-grid">
      <section className="full-section">
        <p className="overline">Real-time · Auto + Manual · Records saved to DB</p>
        <h2>🚨 Police Dispatch Center</h2>

        {/* KPI ribbon */}
        <div className="ref-kpi-ribbon" style={{marginTop:12,marginBottom:16}}>
          <article className="ref-kpi red"><i>🚨</i><div><span>Active Dispatches</span><b>{stats.active}</b><small>Pending / En-route</small></div></article>
          <article className="ref-kpi amber"><i>⚡</i><div><span>Urgent Priority</span><b>{stats.urgent}</b><small>Requires immediate action</small></div></article>
          <article className="ref-kpi blue"><i>✅</i><div><span>Resolved Today</span><b>{stats.resolved}</b><small>Closed records</small></div></article>
          <article className="ref-kpi green"><i>📋</i><div><span>Total Records</span><b>{stats.total}</b><small>All time</small></div></article>
          <article className="ref-kpi purple" style={{alignItems:'center',justifyContent:'center',gap:6}}>
            <div style={{display:'flex',alignItems:'center',gap:6}}>
              <div style={{width:8,height:8,borderRadius:'50%',background: wsOnline ? '#42b955' : '#ee4b50',boxShadow: wsOnline ? '0 0 8px #42b955' : 'none'}}/>
              <span style={{fontSize:'.65rem',color:'#bacbd8',fontWeight:700}}>{wsOnline ? 'LIVE' : 'OFFLINE'}</span>
            </div>
            <small style={{fontSize:'.6rem',color:'#89a8c1'}}>WebSocket</small>
          </article>
        </div>

        {/* Manual dispatch form */}
        <div style={{background:'rgba(238,75,80,0.06)',border:'1px solid #7c242e',borderRadius:10,padding:'14px 18px',marginBottom:16,display:'flex',gap:10,alignItems:'flex-end',flexWrap:'wrap'}}>
          <div style={{flex:1,minWidth:160}}>
            <label style={{fontSize:'.6rem',color:'#ee4b50',fontWeight:800,letterSpacing:'.08em',display:'block',marginBottom:4}}>JUNCTION ID (Manual Override)</label>
            <input
              value={manualJunctionId}
              onChange={e => setManualJunctionId(e.target.value)}
              placeholder="e.g. J042 or type name"
              style={{width:'100%',boxSizing:'border-box',padding:'8px 12px',background:'#0a1928',border:'1px solid #7c242e',borderRadius:6,color:'#eaf5ff',fontSize:'.75rem',outline:'none'}}
            />
          </div>
          <div style={{minWidth:100}}>
            <label style={{fontSize:'.6rem',color:'#ee4b50',fontWeight:800,letterSpacing:'.08em',display:'block',marginBottom:4}}>OFFICERS TO SEND</label>
            <input
              type="number" min={1} max={20} value={manualOfficers}
              onChange={e => setManualOfficers(Number(e.target.value))}
              style={{width:'100%',padding:'8px 12px',background:'#0a1928',border:'1px solid #7c242e',borderRadius:6,color:'#eaf5ff',fontSize:'.75rem',outline:'none'}}
            />
          </div>
          <button
            onClick={manualDispatch}
            disabled={submitting || !manualJunctionId.trim() || !apiOnline}
            style={{padding:'9px 18px',background:'#7c242e',border:'1px solid #ee4b50',borderRadius:6,color:'#ffb4bd',fontSize:'.72rem',fontWeight:800,cursor:'pointer',letterSpacing:'.06em',opacity:(submitting||!apiOnline)?0.5:1}}
          >
            {submitting ? 'DISPATCHING…' : '🚨 DISPATCH NOW'}
          </button>
          {!apiOnline && <small style={{color:'#f7c42f',fontSize:'.6rem'}}>⚠ API offline — start api_server.py</small>}
        </div>

        {/* Filter tabs */}
        <div className="filters" style={{marginBottom:12}}>
          {['ALL','PENDING','DISPATCHED','ON_SCENE','RESOLVED','CANCELLED'].map(f => (
            <button key={f} className={filter===f ? 'selected' : ''} onClick={() => setFilter(f)}>
              {STATUS_META[f]?.icon || '◉'} {STATUS_META[f]?.label || 'All'} 
              {f !== 'ALL' && <span style={{marginLeft:4,opacity:.7}}>({dispatches.filter(d=>d.status===f).length})</span>}
              {f === 'ALL' && <span style={{marginLeft:4,opacity:.7}}>({dispatches.length})</span>}
            </button>
          ))}
        </div>

        {/* Dispatch records table */}
        {shown.length === 0 ? (
          <div style={{textAlign:'center',padding:'48px 24px',color:'#4a7090',fontSize:'.8rem'}}>
            <div style={{fontSize:'2.5rem',marginBottom:8}}>🚔</div>
            No dispatch records{filter !== 'ALL' ? ` with status "${filter}"` : ''}.<br/>
            <small>Dispatches are auto-created when a HIGH/CRITICAL risk is detected by the pipeline.</small>
          </div>
        ) : (
          <div style={{display:'flex',flexDirection:'column',gap:10}}>
            {shown.map(d => {
              const sm   = STATUS_META[d.status]   || STATUS_META.PENDING
              const pm   = PRIORITY_META[d.priority] || PRIORITY_META.HIGH
              const isFlash = flashNew === d.dispatch_id
              return (
                <div key={d.dispatch_id} style={{
                  background: isFlash ? 'rgba(238,75,80,0.12)' : '#0c1e2f',
                  border: `1px solid ${isFlash ? '#ee4b50' : '#1e384d'}`,
                  borderRadius: 10,
                  padding: '12px 16px',
                  transition: 'border-color .3s,background .3s',
                  animation: isFlash ? 'dispatch-flash 0.6s ease infinite alternate' : 'none',
                }}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8,flexWrap:'wrap',gap:8}}>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <span style={{fontSize:'1rem'}}>{sm.icon}</span>
                      <b style={{color:'#eaf5ff',fontSize:'.88rem'}}>{d.junction_name || d.junction_id}</b>
                      <span style={{fontSize:'.62rem',color:'#89a8c1'}}>#{d.dispatch_id}</span>
                    </div>
                    <div style={{display:'flex',gap:6,alignItems:'center',flexWrap:'wrap'}}>
                      {/* Priority badge */}
                      <span style={{background:pm.bg,border:`1px solid ${pm.border}`,color:pm.color,fontSize:'.6rem',fontWeight:800,padding:'2px 8px',borderRadius:10,letterSpacing:'.06em'}}>
                        {d.priority}
                      </span>
                      {/* Status badge */}
                      <span style={{background:sm.bg,border:`1px solid ${sm.border}`,color:sm.color,fontSize:'.6rem',fontWeight:700,padding:'2px 8px',borderRadius:10}}>
                        {sm.label}
                      </span>
                      {/* Risk level */}
                      <span style={{background:'#0a1928',border:'1px solid #1e384d',color:RISK_COLORS[d.risk_level]||'#bacbd8',fontSize:'.6rem',padding:'2px 8px',borderRadius:10}}>
                        {d.risk_level || '—'}
                      </span>
                    </div>
                  </div>

                  {/* Metrics row */}
                  <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(130px,1fr))',gap:'6px 12px',marginBottom:8,fontSize:'.65rem',color:'#89a8c1'}}>
                    <span>🚔 <b style={{color:'#eaf5ff'}}>{d.officers_sent}</b> officers</span>
                    <span>📍 <b style={{color:'#eaf5ff'}}>{d.junction_id}</b></span>
                    <span>⚡ Score: <b style={{color:'#eaf5ff'}}>{d.risk_score != null ? (d.risk_score*100).toFixed(0)+'/100' : '—'}</b></span>
                    <span>🕐 <b style={{color:'#eaf5ff'}}>{d.created_at ? new Date(d.created_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}) : '—'}</b></span>
                    <span style={{gridColumn:'1/-1',color:'#bacbd8'}}>📝 {d.reason || '—'}</span>
                  </div>

                  {/* Action buttons */}
                  {!['RESOLVED','CANCELLED'].includes(d.status) && (
                    <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
                      {d.status === 'PENDING' && (
                        <button onClick={() => updateStatus(d.dispatch_id,'DISPATCHED')}
                          style={{padding:'5px 12px',background:'#0d2a3a',border:'1px solid #53d8ff',borderRadius:5,color:'#53d8ff',fontSize:'.62rem',fontWeight:700,cursor:'pointer'}}>
                          🚔 Mark Dispatched
                        </button>
                      )}
                      {d.status === 'DISPATCHED' && (
                        <button onClick={() => updateStatus(d.dispatch_id,'ON_SCENE')}
                          style={{padding:'5px 12px',background:'#1a3a24',border:'1px solid #42b955',borderRadius:5,color:'#42b955',fontSize:'.62rem',fontWeight:700,cursor:'pointer'}}>
                          ✅ Mark On Scene
                        </button>
                      )}
                      {['PENDING','DISPATCHED','ON_SCENE'].includes(d.status) && (
                        <>
                          <button onClick={() => updateStatus(d.dispatch_id,'RESOLVED')}
                            style={{padding:'5px 12px',background:'#071525',border:'1px solid #1e384d',borderRadius:5,color:'#89a8c1',fontSize:'.62rem',fontWeight:700,cursor:'pointer'}}>
                            ☑️ Resolve
                          </button>
                          <button onClick={() => updateStatus(d.dispatch_id,'CANCELLED')}
                            style={{padding:'5px 12px',background:'#3d1c24',border:'1px solid #7c242e',borderRadius:5,color:'#ffb4bd',fontSize:'.62rem',fontWeight:700,cursor:'pointer'}}>
                            ✖ Cancel
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </section>
    </div>
  )
}

/* ─────────────── Officer Allocation Page (Problem Statement B) ─────── */
function OfficerAllocation({ historicalJunctions }) {
  const [totalOfficers, setTotalOfficers] = useState(60)
  const [activeTab, setActiveTab] = useState('comparison') // 'ranked' | 'comparison' | 'unmanned'
  const [searchQuery, setSearchQuery] = useState('')
  const [manualOverrides, setManualOverrides] = useState({})

  // 1. Run personnel-allocation algorithm with priority calculation
  const { allocatedJunctions, baselineCount, recommendedCount, unmannedHighRiskJunctions, addressedUnmannedCount } = useMemo(() => {
    let list = historicalJunctions.map(j => {
      const riskScore = parseFloat(j.risk_score) || 0
      const isUnmanned = parseInt(j.officer_present) === 0 ? 1 : 0
      
      // priority score: risk_score * (1 + 0.5 * (1 - officer_present))
      const priorityScore = riskScore * (1 + 0.5 * isUnmanned)
      return {
        ...j,
        riskScore,
        isUnmanned,
        priorityScore,
        recommended_officers: 0
      }
    })

    // Sort by priority score descending
    list.sort((a, b) => b.priorityScore - a.priorityScore)

    let officersRemaining = totalOfficers
    
    // Allocate based on rules:
    // High risk (risk_score >= 0.70 or category HIGH) gets 2
    // Medium risk (risk_score >= 0.40 or category MEDIUM) gets 1
    list = list.map(j => {
      if (officersRemaining <= 0) return j
      let officers = 0
      if (j.risk_category === 'HIGH' || j.riskScore >= 0.7) {
        officers = 2
      } else if (j.risk_category === 'MEDIUM' || j.riskScore >= 0.4) {
        officers = 1
      }
      officers = Math.min(officers, officersRemaining)
      officersRemaining -= officers
      return {
        ...j,
        recommended_officers: officers
      }
    })

    // Apply manual overrides
    list = list.map(j => {
      if (manualOverrides[j.junction_id] !== undefined) {
        return {
          ...j,
          recommended_officers: manualOverrides[j.junction_id]
        }
      }
      return j
    })

    // Calculate aggregations
    const baselineCount = historicalJunctions.reduce((acc, j) => acc + (parseInt(j.officer_present) || 0), 0)
    const recommendedCount = list.reduce((acc, j) => acc + (j.recommended_officers || 0), 0)
    
    const unmannedHighRiskJunctions = list.filter(j => 
      (j.risk_category === 'HIGH' || j.riskScore >= 0.7) && j.isUnmanned === 1
    )
    const addressedUnmannedCount = unmannedHighRiskJunctions.filter(j => j.recommended_officers > 0).length

    return {
      allocatedJunctions: list,
      baselineCount,
      recommendedCount,
      unmannedHighRiskJunctions,
      addressedUnmannedCount
    }
  }, [historicalJunctions, totalOfficers, manualOverrides])

  // Filter junctions by search query
  const filteredList = useMemo(() => {
    const query = searchQuery.toLowerCase().trim()
    let list = allocatedJunctions
    if (query) {
      list = list.filter(j => 
        j.junction_id.toLowerCase().includes(query) || 
        (j.location || '').toLowerCase().includes(query)
      )
    }
    
    if (activeTab === 'comparison') {
      return list.filter(j => parseInt(j.officer_present) !== j.recommended_officers)
    } else if (activeTab === 'unmanned') {
      return list.filter(j => (j.risk_category === 'HIGH' || j.riskScore >= 0.7) && j.isUnmanned === 1)
    }
    return list // 'ranked'
  }, [allocatedJunctions, activeTab, searchQuery])

  const handleOverride = (junction_id, delta) => {
    const currentAlloc = allocatedJunctions.find(j => j.junction_id === junction_id)
    if (!currentAlloc) return
    const baseVal = manualOverrides[junction_id] !== undefined 
      ? manualOverrides[junction_id] 
      : currentAlloc.recommended_officers
    const newVal = Math.max(0, Math.min(5, baseVal + delta))
    setManualOverrides(curr => ({
      ...curr,
      [junction_id]: newVal
    }))
  }

  const resetOverrides = () => setManualOverrides({})

  return (
    <div className="page-grid">
      <section className="full-section">
        <p className="overline">Problem Statement B Expected Solution (iv, v, vi, vii, viii)</p>
        <h2>Personnel Allocation & Dynamic Redeployment</h2>
        <p style={{color:'#89a8c1',fontSize:'.75rem',maxWidth:700,lineHeight:1.4}}>
          Compare baseline deployments with algorithmically optimized coverage. Unmanned high-risk junctions are automatically prioritized with a 50% score weight boost. Use the manual override controls to fine-tune allocation.
        </p>

        {/* Algorithm Settings Slider */}
        <div style={{background:'#0c1e2f',border:'1px solid #1e384d',borderRadius:8,padding:16,marginTop:16,display:'grid',gridTemplateColumns:'1.5fr 1fr',gap:24,alignItems:'center'}}>
          <div>
            <label style={{display:'flex',justifyContent:'space-between',fontSize:'.7rem',fontWeight:700,color:'#53d8ff',marginBottom:8}}>
              <span>TOTAL OFFICERS LIMIT (PERSONNEL ALLOCATION PANEL)</span>
              <b>{totalOfficers} officers</b>
            </label>
            <input 
              type="range" 
              min="10" 
              max="150" 
              value={totalOfficers} 
              onChange={(e) => setTotalOfficers(parseInt(e.target.value))}
              style={{width:'100%',accentColor:'#53d8ff',cursor:'pointer'}}
            />
            <div style={{display:'flex',justifyContent:'space-between',fontSize:'.58rem',color:'#6b8ba3',marginTop:4}}>
              <span>10 Min</span>
              <span>60 Optimized Nagpur Peak</span>
              <span>150 Max Capacity</span>
            </div>
          </div>
          <div style={{display:'flex',gap:8,justifyContent:'flex-end'}}>
            {Object.keys(manualOverrides).length > 0 && (
              <button className="ref-clear" style={{margin:0,fontSize:'.65rem'}} onClick={resetOverrides}>
                Reset Overrides ({Object.keys(manualOverrides).length})
              </button>
            )}
          </div>
        </div>

        {/* KPI Ribbon */}
        <div className="ref-kpi-ribbon" style={{marginTop:16}}>
          <article className="ref-kpi blue">
            <i>♟</i>
            <div>
              <span>Baseline Deployed</span>
              <b>{baselineCount}</b>
              <small>Deployed at registry start</small>
            </div>
          </article>
          <article className="ref-kpi purple">
            <i>⚖</i>
            <div>
              <span>Recommended Deployed</span>
              <b>{recommendedCount}</b>
              <small>Allocated by priority algorithm</small>
            </div>
          </article>
          <article className="ref-kpi red">
            <i>▲</i>
            <div>
              <span>High-Risk Unmanned</span>
              <b>{unmannedHighRiskJunctions.length}</b>
              <small>High risk & 0 baseline officers</small>
            </div>
          </article>
          <article className="ref-kpi green">
            <i>✓</i>
            <div>
              <span>Coverage Addressed</span>
              <b>{addressedUnmannedCount} / {unmannedHighRiskJunctions.length}</b>
              <small>Unmanned areas with recommended coverage</small>
            </div>
          </article>
        </div>
      </section>

      {/* Tabs and Search Bar */}
      <section className="full-section" style={{paddingTop:16}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',gap:16,flexWrap:'wrap',marginBottom:12,borderBottom:'1px solid #1e384d',paddingBottom:12}}>
          <div className="filters">
            <button 
              className={activeTab === 'comparison' ? 'selected' : ''} 
              onClick={() => setActiveTab('comparison')}
            >
              ⇄ Baseline vs Recommended ({allocatedJunctions.filter(j => parseInt(j.officer_present) !== j.recommended_officers).length})
            </button>
            <button 
              className={activeTab === 'ranked' ? 'selected' : ''} 
              onClick={() => setActiveTab('ranked')}
            >
              ☰ Priority Ranked List ({allocatedJunctions.length})
            </button>
            <button 
              className={activeTab === 'unmanned' ? 'selected' : ''} 
              onClick={() => setActiveTab('unmanned')}
            >
              ⚠ High-Risk Unmanned ({unmannedHighRiskJunctions.length})
            </button>
          </div>
          
          <input 
            type="text"
            placeholder="Search by ID or Location name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{background:'#071626',border:'1px solid #1e384d',borderRadius:6,padding:'7px 12px',fontSize:'.68rem',color:'#fff',minWidth:250}}
          />
        </div>

        {/* Table Data */}
        {filteredList.length === 0 ? (
          <Empty>No junctions found matching the search criteria or active tab.</Empty>
        ) : (
          <div className="data-table">
            <div className="table-head" style={{gridTemplateColumns:'80px 1.8fr 110px 110px 110px 130px 1.2fr',gap:10}}>
              <span>ID</span>
              <span>Location / CCTV Segment</span>
              <span>Risk Score</span>
              <span>Priority Score</span>
              <span>Baseline Deployed</span>
              <span>Recommended Deployed</span>
              <span>Manual Override</span>
            </div>
            
            <div style={{maxHeight:450,overflowY:'auto',display:'grid',gap:2}}>
              {filteredList.slice(0, 100).map((j, i) => {
                const isOverridden = manualOverrides[j.junction_id] !== undefined
                const change = j.recommended_officers - (parseInt(j.officer_present) || 0)
                const changeColor = change > 0 ? '#42b955' : change < 0 ? '#ee4b50' : '#89a8c1'
                const changeText = change > 0 ? `+${change}` : change === 0 ? 'No Change' : `${change}`

                // Explanation/reasons
                let explainText = j.risk_reason || 'Low risk learned baseline'
                if (j.isUnmanned && (j.risk_category === 'HIGH' || j.riskScore >= 0.7)) {
                  explainText = `⚠ Prioritized: Unmanned High-Risk area. ${explainText}`
                }

                return (
                  <div 
                    key={`${j.junction_id}-${i}`} 
                    className="table-row" 
                    style={{
                      gridTemplateColumns:'80px 1.8fr 110px 110px 110px 130px 1.2fr',
                      borderBottom:'1px solid #0f2133',
                      background: isOverridden ? 'rgba(83,216,255,0.03)' : (j.isUnmanned && j.risk_category === 'HIGH') ? 'rgba(238,75,80,0.02)' : 'transparent',
                      minWidth:'100%'
                    }}
                  >
                    <b style={{color:'#53d8ff'}}>{j.junction_id}</b>
                    <div>
                      <span style={{fontWeight:600,color:'#fff',fontSize:'.7rem'}}>{j.location}</span>
                      <small style={{display:'block',color:'#6b8ba3',fontSize:'.55rem',marginTop:2}}>
                        {explainText}
                      </small>
                    </div>
                    <span>
                      <RiskBadge level={j.risk_category} />
                      <span style={{marginLeft:6,fontSize:'.62rem',color:'#bacbd8'}}>{(j.riskScore*100).toFixed(0)}/100</span>
                    </span>
                    <b style={{color:'#f7c42f'}}>{j.priorityScore.toFixed(3)}</b>
                    <span style={{paddingLeft:10}}>{j.officer_present} officer(s)</span>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <b style={{color:j.recommended_officers > 0 ? '#53d8ff' : '#4a6280'}}>{j.recommended_officers} officer(s)</b>
                      <span style={{fontSize:'.58rem',color:changeColor,fontWeight:800}}>({changeText})</span>
                    </div>
                    <div style={{display:'flex',gap:4}}>
                      <button 
                        style={{background:'#0c2033',border:'1px solid #1c3850',color:'#42b955',borderRadius:4,width:22,height:22,display:'grid',placeItems:'center',fontWeight:'bold',fontSize:'.75rem',cursor:'pointer'}}
                        onClick={() => handleOverride(j.junction_id, 1)}
                        title="Add officer"
                      >
                        +
                      </button>
                      <button 
                        style={{background:'#0c2033',border:'1px solid #1c3850',color:'#ee4b50',borderRadius:4,width:22,height:22,display:'grid',placeItems:'center',fontWeight:'bold',fontSize:'.75rem',cursor:'pointer'}}
                        onClick={() => handleOverride(j.junction_id, -1)}
                        title="Remove officer"
                        disabled={j.recommended_officers === 0}
                      >
                        -
                      </button>
                      {isOverridden && (
                        <button 
                          style={{background:'transparent',border:'none',color:'#ee4b50',fontSize:'.58rem',marginLeft:4,cursor:'pointer'}}
                          onClick={() => {
                            const newOverrides = { ...manualOverrides }
                            delete newOverrides[j.junction_id]
                            setManualOverrides(newOverrides)
                          }}
                        >
                          Reset
                        </button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </section>
    </div>
  )
}

/* ─────────────── Junction Status (Live Traffic) ───────────────────── */
function LiveTraffic({ junctions }) {
  return (
    <div className="page-grid">
      <section className="full-section">
        <p className="overline">Active junction monitoring</p>
        <h2>Live Junction Status</h2>
        {junctions.length ? (
          <div className="junction-grid">
            {junctions.map((j,i) => <JunctionCard key={`${j.junction_id}-${i}`} junction={j} />)}
          </div>
        ) : (
          <Empty>No live junctions active. Start a video in Streamlit to see real junction data.</Empty>
        )}
      </section>
    </div>
  )
}

function JunctionCard({ junction }) {
  return (
    <div className={`junction-card ${levelClass(junction.level)}`}>
      <div className="junction-head">
        <div>
          <b>{junction.source}</b>
          {junction.junction_id && <small style={{color:'#89a8c1',marginLeft:6}}>{junction.junction_id}</small>}
        </div>
        <div style={{display:'flex',gap:6,alignItems:'center'}}>
          {junction.demo ? <span style={{color:'#ffd067',fontSize:'.62rem',fontWeight:700}}>● DEMO</span>
                         : <span style={{color:'#45dd94',fontSize:'.62rem',fontWeight:700}}>● LIVE</span>}
          <RiskBadge level={junction.level} />
        </div>
      </div>
      <div className="junction-metrics">
        <span>Score<b>{Number(junction.score).toFixed(3)}</b></span>
        <span>Vehicles<b>{junction.vehicles}</b></span>
        <span>Density<b>{Number(junction.density).toFixed(1)}</b></span>
        <span>Flow<b>{Math.round(junction.flow)}/hr</b></span>
        <span>Camera<b>{junction.camera_id||junction.source}</b></span>
        {junction.speed != null
          ? <span>Speed<b>{Math.round(junction.speed)} km/h</b></span>
          : <span style={{color:'#89a8c1'}}>Speed<b style={{color:'#4a6280'}}>N/A</b></span>}
      </div>
      {junction.latitude && (
        <small style={{color:'#4a6280',fontSize:'.58rem',marginTop:4,display:'block'}}>
          📍 {Number(junction.latitude).toFixed(4)}, {Number(junction.longitude).toFixed(4)}
        </small>
      )}
      <small style={{color:'#89a8c1',fontSize:'.6rem',marginTop:4,display:'block'}}>Updated {time(junction.recorded_at)}</small>
    </div>
  )
}

/* ─────────────── Analytics ──────────────────────────────────────── */
function Analytics({ events }) {
  const risk = events.filter(e => e.event_type?.includes('risk'))
  const low  = risk.filter(e => (e.prediction?.risk_level || '') === 'LOW').length
  const med  = risk.filter(e => (e.prediction?.risk_level || '') === 'MEDIUM').length
  const high = risk.filter(e => (e.prediction?.risk_level || '') === 'HIGH').length
  const crit = risk.filter(e => (e.prediction?.risk_level || '') === 'CRITICAL').length
  const total = Math.max(risk.length, 1)
  return (
    <div className="page-grid">
      <section className="full-section">
        <p className="overline">Event analytics</p>
        <h2>{events.length} persisted events</h2>
        <div className="heatmap-strip">
          {[['LOW',low,'#42b955'],['MEDIUM',med,'#f7c42f'],['HIGH',high,'#f29122'],['CRITICAL',crit,'#ee4b50']].map(([l,n,c]) => (
            <div key={l} style={{borderColor:c}}>
              <b>{n}</b><span style={{color:c}}>{l}</span>
              <small>{((n/total)*100).toFixed(0)}%</small>
            </div>
          ))}
        </div>
      </section>
      <section className="full-section">
        <p className="overline">Recent events</p>
        <h2>Latest pipeline output</h2>
        {events.length ? (
          <div className="data-table">
            <div className="table-head"><span>Time</span><span>Junction</span><span>Type</span><span>Level</span></div>
            {events.slice(0, 25).map((e,i) => (
              <div key={`${e.event_id}-${i}`} className="table-row">
                <span>{time(e.recorded_at)}</span>
                <span>{e.junction_id || e.source_name}</span>
                <span>{niceType(e.event_type)}</span>
                <span><RiskBadge level={eventFacts(e).risk_level || 'LOW'} /></span>
              </div>
            ))}
          </div>
        ) : <Empty>No events yet.</Empty>}
      </section>
    </div>
  )
}

/* ─────────────── Risk Analysis ──────────────────────────────────── */
function RiskAnalysis({ events }) {
  const riskEvents = events.filter(e => e.event_type?.includes('risk'))
  return (
    <div className="page-grid">
      <section className="full-section">
        <p className="overline">Risk predictions</p>
        <h2>{riskEvents.length} risk events</h2>
        {riskEvents.length ? (
          <div className="data-table">
            <div className="table-head"><span>Time</span><span>Junction</span><span>Score</span><span>Level</span><span>Confidence</span><span>Explanation</span></div>
            {riskEvents.slice(0, 30).map((e,i) => {
              const p = e.prediction || {}
              return (
                <div key={`${e.event_id}-${i}`} className="table-row">
                  <span>{time(e.recorded_at)}</span>
                  <span>{e.junction_id || e.source_name}</span>
                  <span>{Number(p.risk_score||0).toFixed(3)}</span>
                  <span><RiskBadge level={p.risk_level} /></span>
                  <span>{Number(p.model_confidence||0).toFixed(0)}%</span>
                  <span style={{color:'#89a8c1',fontSize:'.6rem'}}>{(p.explanation||[]).join(' · ')}</span>
                </div>
              )
            })}
          </div>
        ) : <Empty>No risk predictions yet.</Empty>}
      </section>
    </div>
  )
}

/* ─────────────── System Status ──────────────────────────────────── */
function SystemStatus({ apiOnline, wsOnline, events, pipelineStatus }) {
  const stages = [
    ['CCTV Input',       pipelineStatus.cctv       || 'IDLE'],
    ['YOLOv8',           pipelineStatus.yolov8      || 'IDLE'],
    ['ByteTrack',        pipelineStatus.bytetrack   || 'IDLE'],
    ['Feature Engine',   pipelineStatus.features    || 'IDLE'],
    ['Risk Classifier',  pipelineStatus.risk        || 'IDLE'],
    ['Response Engine',  pipelineStatus.response    || 'IDLE'],
    ['Redeployment',     pipelineStatus.redeployment|| 'IDLE'],
    ['SQLite Store',     apiOnline ? 'LIVE' : 'OFFLINE'],
    ['Local API',        apiOnline ? 'LIVE' : 'OFFLINE'],
    ['WebSocket',        wsOnline  ? 'LIVE' : 'OFFLINE'],
    ['React Dashboard',  'LIVE'],
  ]
  const stageColor = (s) => s === 'LIVE' || s === 'ACTIVE' || s === 'PROCESSING' || s === 'TRACKING' || s === 'COLLECTING' || s === 'SCORING' || s === 'DECIDING' || s === 'RUNNING'
    ? '#35d2a4' : s === 'READY' ? '#f7c42f' : s === 'OFFLINE' ? '#ee4b50' : '#7aaec8'

  return (
    <div className="page-grid">
      <section className="full-section">
        <p className="overline">Pipeline status</p>
        <h2>Live system diagnostics</h2>
        <div className="pipeline-stages">
          {stages.map(([name, status]) => (
            <div key={name} className="pipeline-stage">
              <span>{name}</span>
              <b style={{color: stageColor(status)}}>{status}</b>
            </div>
          ))}
        </div>
      </section>
      <section className="full-section">
        <p className="overline">Recent events</p>
        <h2>Live event feed</h2>
        {events.length ? (
          <div className="event-feed">
            {events.slice(0, 20).map((e,i) => (
              <div key={`${e.event_id}-${i}`}>
                <time>{time(e.recorded_at)}</time>
                <span>{e.junction_id || e.source_name}</span>
                <b>{niceType(e.event_type)}</b>
                <em>{eventFacts(e).risk_level || eventFacts(e).priority || 'stored'}</em>
              </div>
            ))}
          </div>
        ) : <Empty>No events yet.</Empty>}
      </section>
    </div>
  )
}

/* ─────────────── Control Room (main dashboard) ──────────────────── */
function LiveEventFeedPanel({ events }) {
  const display = events.filter(e =>
    e.event_type?.includes('feature_window') ||
    e.event_type?.includes('risk_prediction') ||
    e.event_type?.includes('response') ||
    e.event_type?.includes('redeployment') ||
    e.event_type?.includes('status')
  ).slice(0, 10)
  return (
    <section className="ref-rail-panel">
      <div className="ref-panel-head"><h2>Live Pipeline Feed</h2><span>LOGS</span></div>
      {display.length ? (
        <div style={{padding:'0 8px',display:'grid',gap:5,overflowY:'auto',maxHeight:280}}>
          {display.map((e,i) => {
            const jText = e.junction_id ? e.junction_id : 'SYSTEM'
            const typeShort = e.event_type?.includes('feature') ? '📊 WINDOW' :
                              e.event_type?.includes('risk') ? '⚠️ RISK' :
                              e.event_type?.includes('redeploy') ? '🚔 REDEPLOY' :
                              e.event_type?.includes('response') ? '✅ RESPONSE' : 'ℹ️ STATUS'
            const lvl = eventFacts(e).risk_level || eventFacts(e).priority || ''
            return (
              <div key={`${e.event_id}-${i}`} style={{display:'flex',gap:6,fontSize:'.6rem',padding:'5px 6px',border:'1px solid #1e384d',borderRadius:4,background:'#0a1928',alignItems:'center'}}>
                <time style={{color:'#8199ad',minWidth:56}}>{time(e.recorded_at)}</time>
                <b style={{color:'#ffd067',minWidth:52}}>{jText}</b>
                <span style={{color:'#53d8ff',minWidth:80}}>{typeShort}</span>
                {lvl && <RiskBadge level={lvl} />}
              </div>
            )
          })}
        </div>
      ) : <Empty>Awaiting pipeline events…</Empty>}
    </section>
  )
}

function ReferenceControlRoom({ events, junctions, historicalJunctions = [], demoMode, setDemoMode, apiOnline, wsOnline, clearSession, pipelineStatus, redeployments }) {
  const usingDemo   = junctions.some(j => j.demo)
  const responses   = events.filter(e => e.event_type?.includes('response'))
  const totalVehicles = junctions.reduce((s,j) => s + Number(j.vehicles||0), 0)
  const highRisk    = junctions.filter(j => ['HIGH','CRITICAL'].includes(j.level)).length
  const alerts      = responses.length || junctions.filter(j => j.alert && j.alert !== 'MONITORING').length
  const totalRedeploy = redeployments.reduce((s,r) => s + (r.additional_officers||0), 0)

  const alertRows = responses.length ? responses.slice(0, 4).map(e => {
    const d = eventFacts(e)
    return { title:`${d.risk_level||'TRAFFIC'} ALERT`, source:e.junction_id||e.source_name, detail:d.recommended_action||'Rule-based recommendation', level:d.risk_level||'MEDIUM', time:time(e.recorded_at) }
  }) : usingDemo ? [
    { title:'CRITICAL ALERT', source:'J004 · Market Area', detail:'Severe congestion · immediate demo response', level:'CRITICAL', time:'LIVE' },
    { title:'HIGH RISK',      source:'J001 · Main Road',   detail:'High traffic density · increase monitoring',  level:'HIGH',     time:'LIVE' },
    { title:'MEDIUM RISK',    source:'J003 · Highway Exit',detail:'Moderate traffic · continue monitoring',     level:'MEDIUM',   time:'LIVE' },
  ] : []

  return (
    <div className="ref-dashboard">
      <section className="ref-topline">
        <div>
          <h1>TrafficRisk AI Control Room</h1>
          <p>Real-time traffic monitoring, risk prediction & decision support</p>
        </div>
        <div className="ref-head-actions">
          <button className={`ref-demo ${demoMode ? 'on' : ''}`} onClick={() => setDemoMode(!demoMode)}>
            {demoMode ? '● DEMO MODE ON' : '○ REAL PIPELINE'}
          </button>
          {!demoMode && (
            <button className="ref-clear" onClick={clearSession}>✖ CLEAR SESSION</button>
          )}
          <div className="ref-live"><i className={apiOnline ? 'pulse' : ''}/> {apiOnline ? 'Live' : 'Offline'}</div>
          <div className="ref-clock">{new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'})}<small>Control room</small></div>
        </div>
      </section>

      {usingDemo && <div className="ref-demo-notice">DEMO MODE — Simulated scenarios. Real pipeline events will override this view.</div>}
      {!usingDemo && junctions.length === 0 && (
        <div className="ref-demo-notice" style={{background:'#1c2834',color:'#89a8c1',borderColor:'#2b3f54'}}>
          REAL PIPELINE — Select a junction in Streamlit, upload a video, and click Start. Events will appear here instantly via WebSocket.
        </div>
      )}

      <section className="ref-kpi-ribbon">
        <article className="ref-kpi blue"><i>●</i><div><span>Active Junctions</span><b>{usingDemo ? '4' : junctions.length}</b><small>Video feeds</small></div></article>
        <article className="ref-kpi green"><i>▰</i><div><span>Total Vehicles</span><b>{totalVehicles||'—'}</b><small>Detected</small></div></article>
        <article className="ref-kpi red"><i>▲</i><div><span>High Risk</span><b>{highRisk}</b><small>Need attention</small></div></article>
        <article className="ref-kpi amber"><i>♟</i><div><span>Active Alerts</span><b>{alerts}</b><small>Recommendations</small></div></article>
        <article className="ref-kpi purple"><i>⚔</i><div><span>Officers Recommended</span><b>{totalRedeploy}</b><small>Decision support</small></div></article>
        <article className="ref-kpi cyan"><i>◴</i>
          <div>
            <span>Avg Speed</span>
            <b>{junctions.some(j => j.speed!=null) ? `${Math.round(junctions.reduce((s,j)=>s+Number(j.speed||0),0)/Math.max(1,junctions.filter(j=>j.speed!=null).length))} km/h` : '—'}</b>
            <small>{junctions.some(j=>j.speed!=null) ? 'Calibrated' : 'Calibration required'}</small>
          </div>
        </article>
      </section>

      <section className="ref-layout">
        <div className="ref-main">
          {/* Google Map */}
          <section className="ref-map-card">
            <div className="ref-panel-head"><h2>Live Risk Map — Nagpur</h2><span>{usingDemo ? 'DEMO' : `LIVE · ${historicalJunctions.length} junctions`}</span></div>
            <GoogleMapPanel junctionCards={junctions} historicalJunctions={historicalJunctions} demoMode={usingDemo} redeployments={redeployments} />
          </section>
          {/* CCTV tiles */}
          <div className="ref-feed-panel">
            <div className="ref-panel-head"><h2>Live CCTV Junction Status</h2><span>{usingDemo ? 'DEMO' : 'EVENT-DRIVEN'}</span></div>
            {junctions.length ? (
              <div className="ref-feeds">
                {junctions.slice(0,4).map((j,i) => <CctvTile key={`${j.junction_id}-${i}`} junction={j} index={i} />)}
              </div>
            ) : <Empty>No live junction data — start a video analysis or enable Demo Mode.</Empty>}
          </div>
        </div>
        <aside className="ref-rail">
          <section className="ref-rail-panel">
            <div className="ref-panel-head"><h2>Active Alerts</h2><span style={{color:alertRows.length?'#ff5f5f':'#4a6280'}}>{alertRows.length || 'CLEAR'}</span></div>
            {alertRows.length ? (
              <div className="ref-alerts">
                {alertRows.map((r,i) => (
                  <article key={i} className={levelClass(r.level)}>
                    <i>{r.level==='CRITICAL'?'▲':r.level==='HIGH'?'!':'▲'}</i>
                    <div><b>{r.title}</b><span>{r.source} · {r.detail}</span></div>
                    <time>{r.time}</time>
                  </article>
                ))}
              </div>
            ) : <Empty>No active alerts.</Empty>}
          </section>
          {/* Redeployment summary */}
          <section className="ref-rail-panel">
            <div className="ref-panel-head"><h2>Redeployment</h2><span style={{color:'#f29122'}}>DECISION SUPPORT</span></div>
            {redeployments.length ? (
              <div style={{display:'grid',gap:6,padding:'4px 8px 8px'}}>
                {redeployments.slice(0,4).map((rd,i) => (
                  <div key={i} style={{background:'#0a1928',border:'1px solid #1e384d',borderRadius:6,padding:'6px 8px',fontSize:'.63rem'}}>
                    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:3}}>
                      <b style={{color:'#53d8ff'}}>{rd.junction_id}</b>
                      <RiskBadge level={rd.risk_level} />
                    </div>
                    <div style={{color:'#f7c42f',fontWeight:700}}>+{rd.additional_officers} officers recommended</div>
                    <div style={{color:'#89a8c1',marginTop:2}}>{rd.priority} priority</div>
                    <div style={{color:'#f29122',marginTop:2,fontStyle:'italic',fontSize:'.58rem'}}>⏳ Human approval required</div>
                  </div>
                ))}
              </div>
            ) : <Empty>No redeployment recommendations.</Empty>}
          </section>
          <LiveEventFeedPanel events={events} />
          <PipelineStatusPanel pipelineStatus={pipelineStatus} apiOnline={apiOnline} wsOnline={wsOnline} />
        </aside>
      </section>
    </div>
  )
}

function PipelineStatusPanel({ pipelineStatus, apiOnline, wsOnline }) {
  const stages = [
    ['CCTV',       pipelineStatus.cctv       || 'IDLE'],
    ['YOLOv8',     pipelineStatus.yolov8      || 'IDLE'],
    ['ByteTrack',  pipelineStatus.bytetrack   || 'IDLE'],
    ['Features',   pipelineStatus.features    || 'IDLE'],
    ['Risk',       pipelineStatus.risk        || 'IDLE'],
    ['Response',   pipelineStatus.response    || 'IDLE'],
    ['Redeploy',   pipelineStatus.redeployment|| 'IDLE'],
    ['SQLite',     apiOnline ? 'LIVE' : 'OFFLINE'],
    ['API',        apiOnline ? 'LIVE' : 'OFFLINE'],
    ['WebSocket',  wsOnline  ? 'LIVE' : 'OFFLINE'],
  ]
  const color = (s) => s==='LIVE'||s==='ACTIVE'||s==='PROCESSING'||s==='TRACKING'||s==='COLLECTING'||s==='SCORING'||s==='DECIDING'||s==='RUNNING'
    ? '#35d2a4' : s==='READY' ? '#f7c42f' : s==='OFFLINE' ? '#ee4b50' : '#4a6280'
  return (
    <section className="ref-rail-panel">
      <div className="ref-panel-head"><h2>Pipeline Status</h2></div>
      <div style={{padding:'4px 8px',display:'grid',gap:3}}>
        {stages.map(([name, status]) => (
          <div key={name} style={{display:'flex',justifyContent:'space-between',fontSize:'.63rem',padding:'3px 0',borderBottom:'1px solid #0f2133'}}>
            <span style={{color:'#89a8c1'}}>{name}</span>
            <b style={{color:color(status)}}>{status}</b>
          </div>
        ))}
      </div>
    </section>
  )
}

function CctvTile({ junction, index }) {
  const cars   = junction.breakdown?.car         ?? Math.round(Number(junction.vehicles||0)*.52)
  const bikes  = junction.breakdown?.motorcycle  ?? Math.round(Number(junction.vehicles||0)*.31)
  const buses  = junction.breakdown?.bus         ?? Math.round(Number(junction.vehicles||0)*.05)
  const trucks = junction.breakdown?.truck       ?? Math.round(Number(junction.vehicles||0)*.12)
  return (
    <article className={`cctv-tile cctv-${index}`}>
      <div className="cctv-head">
        <b>{junction.junction_id || junction.source}</b>
        <span style={{color:junction.demo?'#ffbd47':'#45dd94'}}>● {junction.demo?'DEMO':'LIVE'}</span>
      </div>
      <div className="cctv-frame">
        <div className="road-lanes"/>
        <div className="det-box box-a"/><div className="det-box box-b"/><div className="det-box box-c"/>
        <small>{junction.demo ? 'SIMULATED VIEW' : `LIVE: ${junction.camera_id||'CAM'}`}</small>
      </div>
      <div className="cctv-foot">
        <span>Cars<b>{cars}</b></span>
        <span>Bikes<b>{bikes}</b></span>
        <span>Buses<b>{buses}</b></span>
        <span>Trucks<b>{trucks}</b></span>
      </div>
    </article>
  )
}

/* ─────────────────────────── App ────────────────────────────────── */
const customStyles = `
  .workspace { width: calc(100% - 252px); margin-left: 252px; padding: 20px clamp(14px,3vw,48px) 20px; } header { display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; border-bottom:1px solid #1c354b; padding-bottom:14px; } h1,h2,h3,p { margin-top: 0; } h1 { font-size: clamp(2.25rem,4vw,4.3rem); line-height: .98; letter-spacing: -.06em; margin: 11px 0 19px; } h1 span { color:#57ccff; } h2 { font-size: clamp(1.28rem,2vw,1.8rem); letter-spacing:-.035em; margin-bottom:10px; } h3 { font-size:1rem; margin-bottom:10px; } header h2 { margin: 3px 0 0; font-size: 1.45rem; } .overline,.eyebrow { margin:0; text-transform:uppercase; letter-spacing:.15em; font-weight:700; color:#54caf7; font-size:.66rem; } .header-status { display:flex; align-items:center; gap:12px; color:#7696b1; font-size:.75rem; }
  .gmap-wrapper { position:relative; display:flex; flex-direction:column; flex:1; min-height:480px; border-radius:0 0 10px 10px; overflow:hidden; }
  .gmap-canvas  { flex:1; width:100%; min-height:480px; }
  .gmap-loading { position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#0d1f33;color:#7aaec8;font-size:.75rem; }
  .map-error-panel { display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px 24px;gap:12px;background:#0a1928;border-radius:10px;color:#bacbd8;text-align:center;min-height:300px; }
  .map-error-panel h3 { color:#f7c42f;margin:0; }
  .map-error-panel code { background:#0c2236;padding:8px 14px;border-radius:6px;color:#53d8ff;font-size:.72rem;border:1px solid #1d394f; }
  .map-error-icon { font-size:2.5rem; }
  .gmap-legend { position:absolute;bottom:28px;right:12px;background:rgba(8,23,38,0.88);backdrop-filter:blur(8px);border:1px solid #1e384d;border-radius:10px;padding:10px 14px;z-index:10;min-width:140px; }
  .gmap-legend-title { color:#53d8ff;font-size:.65rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px; }
  .gmap-legend-row { display:flex;align-items:center;gap:7px;font-size:.67rem;color:#bacbd8;margin-bottom:4px; }
  .gmap-legend-row span { display:inline-block;width:10px;height:10px;border-radius:50%;min-width:10px; }
  .gmap-legend-row b { color:#eaf5ff; }
  .gmap-legend-total { margin-top:6px;padding-top:6px;border-top:1px solid #1e384d;font-size:.63rem;color:#89a8c1;text-align:center; }
  .gmap-info-head { display:flex;justify-content:space-between;align-items:center;margin-bottom:8px; }
  .gmap-info-head b { color:#53d8ff;font-size:.8rem; }
  .gmap-info-head button { background:none;border:none;color:#89a8c1;cursor:pointer;font-size:.9rem; }
  .gmap-info-body { display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:.65rem;color:#bacbd8; }
  .pipeline-stages { display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-top:12px; }
  .pipeline-stage  { background:#0c1e2f;border:1px solid #1e384d;border-radius:8px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;font-size:.7rem; }
  .pipeline-stage span { color:#89a8c1; }
  .pipeline-stage b { font-weight:800;font-size:.72rem; }
  .junction-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-top:12px; }
  .junction-card { background:#0c1e2f;border:1px solid #1e384d;border-radius:10px;padding:14px;transition:border-color .2s; }
  .junction-card.level-high { border-color:#f29122; }
  .junction-card.level-critical { border-color:#ee4b50; }
  .junction-card.level-medium { border-color:#f7c42f; }
  .junction-card.level-low { border-color:#42b955; }
  .junction-head { display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px; }
  .junction-metrics { display:grid;grid-template-columns:repeat(3,1fr);gap:6px;font-size:.62rem;color:#89a8c1; }
  .junction-metrics span { display:flex;flex-direction:column;gap:2px; }
  .junction-metrics b { color:#bacbd8;font-size:.72rem; }
  .redeploy-card { background:#0c1e2f;border:1px solid #1e384d;border-radius:10px;padding:14px;margin-bottom:10px;transition:border-color .2s; }
  .redeploy-card.level-high { border-color:#f29122; }
  .redeploy-card.level-critical { border-color:#ee4b50; }
  .redeploy-card.level-medium { border-color:#f7c42f; }
  .redeploy-head { display:flex;justify-content:space-between;align-items:center;margin-bottom:10px; }
  .redeploy-head b { font-size:.9rem;color:#eaf5ff;margin-right:8px; }
  .redeploy-priority { font-size:.6rem;font-weight:800;padding:2px 8px;border-radius:10px;letter-spacing:.05em; }
  .redeploy-priority[data-priority=URGENT]  { background:#3d1c24;color:#ee4b50;border:1px solid #7c242e; }
  .redeploy-priority[data-priority=HIGH]    { background:#3a2210;color:#f29122;border:1px solid #7c4e24; }
  .redeploy-priority[data-priority=MEDIUM]  { background:#3a3210;color:#f7c42f;border:1px solid #7c6824; }
  .redeploy-priority[data-priority=LOW]     { background:#1a3a24;color:#42b955;border:1px solid #24703a; }
  .redeploy-body { display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:10px; }
  .redeploy-reasons { display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px; }
  .reason-tag { background:#0a1928;border:1px solid #1e384d;border-radius:4px;padding:2px 8px;color:#bacbd8;font-size:.6rem; }
  .redeploy-trace { background:#060f1a;border:1px solid #0f2133;border-radius:8px;padding:10px;margin-top:8px; }
  .trace-btn { background:#0a1928;border:1px solid #1e384d;border-radius:4px;padding:3px 8px;color:#53d8ff;font-size:.6rem;cursor:pointer; }
  .trace-btn:hover { background:#1d394f; }
  .ref-clear { padding:6px 10px;background:#3b1c20;border:1px solid #7c242e;border-radius:6px;color:#ffb4bd;font-size:.62rem;font-weight:800;cursor:pointer; }
  .ref-clear:hover { background:#5c2027; }
  .offline-banner { background:#3d1c24;border-bottom:1px solid #7c242e;color:#ffb4bd;padding:8px 20px;text-align:center;font-size:.7rem;font-weight:700;letter-spacing:.05em; }
  .ref-kpi.green { --kpi-color:#42b955; } .ref-kpi.green i { color:#42b955; }
  .ref-kpi.purple { --kpi-color:#9b59b6; } .ref-kpi.purple i { color:#9b59b6; }
  /* ── Layout overrides: fill space, no empty gaps ── */
  .ref-kpi-ribbon { gap:8px; margin-bottom:10px; }
  .ref-layout { grid-template-columns:minmax(0,1.65fr) minmax(0,0.85fr) !important; gap:10px !important; align-items:stretch !important; }
  .ref-main { grid-template-rows:1fr auto; align-items:stretch; }
  .ref-map-card { display:flex !important; flex-direction:column !important; }
  .ref-map-card .ref-panel-head { flex-shrink:0; }
  .ref-rail { align-self:stretch; }
  .ref-dashboard { gap:8px; }
  .ref-topline { margin-bottom:0; padding-bottom:6px; }
  .ref-demo-notice { margin-bottom:6px; font-size:.63rem; padding:6px 12px; }

  /* ── Map search overlay ── */
  .gmap-search-container { position:absolute; top:12px; left:50%; transform:translateX(-50%); z-index:20; width:340px; }
  .gmap-search-input { width:100%; box-sizing:border-box; padding:9px 14px; background:rgba(8,23,38,0.92); backdrop-filter:blur(10px); border:1px solid #1e5f8c; border-radius:8px; color:#eaf5ff; font-size:.75rem; outline:none; transition:border-color .2s,box-shadow .2s; }
  .gmap-search-input:focus { border-color:#53d8ff; box-shadow:0 0 0 3px rgba(83,216,255,0.15); }
  .gmap-search-input::placeholder { color:#4a7090; }
  .gmap-search-results { position:absolute; top:calc(100% + 4px); left:0; right:0; background:rgba(8,23,38,0.97); border:1px solid #1e384d; border-radius:8px; overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,0.5); }
  .gmap-search-item { padding:9px 14px; font-size:.72rem; color:#bacbd8; cursor:pointer; border-bottom:1px solid #0f2133; transition:background .15s; }
  .gmap-search-item:last-child { border-bottom:none; }
  .gmap-search-item:hover { background:rgba(83,216,255,0.08); color:#eaf5ff; }

  /* ── Filter tabs (Officer Allocation & other pages) ── */
  .filters { display:flex; gap:6px; flex-wrap:wrap; }
  .filters button { background:#0a1928; border:1px solid #1e384d; border-radius:6px; color:#89a8c1; font-size:.65rem; padding:5px 12px; cursor:pointer; transition:all .18s; }
  .filters button:hover { border-color:#53d8ff; color:#eaf5ff; }
  .filters button.selected { background:rgba(83,216,255,0.12); border-color:#53d8ff; color:#53d8ff; font-weight:700; }

  /* ── KPI tiles: pulse glow on hover ── */
  .ref-kpi { cursor:default; transition:transform .18s,box-shadow .18s; }
  .ref-kpi:hover { transform:translateY(-2px); box-shadow:0 6px 24px rgba(83,216,255,0.1); }

  /* ── Data table: polished scrollbar ── */
  .data-table { overflow:hidden; border:1px solid #1e384d; border-radius:8px; }
  .data-table .table-head { background:#071525; padding:8px 12px; font-size:.63rem; font-weight:700; color:#53d8ff; letter-spacing:.07em; text-transform:uppercase; display:grid; gap:12px; }
  .data-table .table-row { padding:7px 12px; display:grid; gap:12px; font-size:.67rem; color:#bacbd8; transition:background .12s; border-top:1px solid #0f2133; }
  .data-table .table-row:hover { background:rgba(83,216,255,0.04); }
  .data-table > div::-webkit-scrollbar { width:5px; } 
  .data-table > div::-webkit-scrollbar-track { background:#071525; }
  .data-table > div::-webkit-scrollbar-thumb { background:#1e384d; border-radius:4px; }

  /* ── Sidebar button active indicator improvement ── */
  nav button { position:relative; transition:background .15s,padding-left .15s; }
  nav button.active::before { content:''; position:absolute; left:0; top:20%; bottom:20%; width:3px; background:#53d8ff; border-radius:0 3px 3px 0; }
  nav button.active { background:rgba(83,216,255,0.1); }

  /* ── Smooth panel card transitions ── */
  .ref-rail-panel, .ref-feed-panel, .ref-map-card { transition:box-shadow .2s; }
  .ref-rail-panel:hover, .ref-feed-panel:hover { box-shadow:0 4px 20px rgba(0,0,0,0.3); }

  /* ── Dispatch Center — urgent flash animation ── */
  @keyframes dispatch-flash {
    from { box-shadow:0 0 0 0 rgba(238,75,80,0.0); border-color:#7c242e; }
    to   { box-shadow:0 0 18px 4px rgba(238,75,80,0.35); border-color:#ee4b50; }
  }

  /* ── Dispatch badge on sidebar nav ── */
  .nav-badge {
    display:inline-flex; align-items:center; justify-content:center;
    min-width:16px; height:16px; padding:0 4px;
    background:#ee4b50; border-radius:8px;
    font-size:.55rem; font-weight:800; color:#fff;
    margin-left:auto; letter-spacing:0;
  }
`

export default function App() {
  const [active, setActive] = useState('control')
  const [events, setEvents] = useState([])
  const [junctions, setJunctions] = useState([])
  const [dispatches, setDispatches] = useState([])
  const [apiOnline, setApiOnline]   = useState(false)
  const [wsOnline,  setWsOnline]    = useState(false)
  const [demoMode, setDemoMode]     = useState(false)
  const [pipelineStatus, setPipelineStatus] = useState({
    cctv:'IDLE', yolov8:'IDLE', bytetrack:'IDLE', features:'IDLE', risk:'READY', response:'READY', redeployment:'READY'
  })

  const refresh = useCallback(async () => {
    try {
      const [health, recent, map, dispatchRes] = await Promise.all([
        fetch(`${apiOrigin}/health`),
        fetch(`${apiOrigin}/events?limit=150`),
        fetch(`${apiOrigin}/junctions`),
        fetch(`${apiOrigin}/dispatch?limit=100`),
      ])
      if (!health.ok || !recent.ok) throw new Error('API unavailable')
      setApiOnline(true)
      setEvents(await recent.json())
      if (map.ok) setJunctions(await map.json())
      if (dispatchRes.ok) setDispatches(await dispatchRes.json())
    } catch {
      setApiOnline(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, 6000)
    return () => clearInterval(interval)
  }, [refresh])

  useEffect(() => {
    let socket, retry
    const connect = () => {
      socket = new WebSocket(wsOrigin)
      socket.onopen  = () => setWsOnline(true)
      socket.onmessage = (msg) => {
        const event = JSON.parse(msg.data)
        if (event.event_type === 'traffic.session_clear.v1') {
          setEvents([])
          setPipelineStatus({cctv:'IDLE',yolov8:'IDLE',bytetrack:'IDLE',features:'IDLE',risk:'READY',response:'READY',redeployment:'READY'})
          return
        }
        if (event.event_type === 'traffic.dispatch.v1') {
          setDispatches(cur => {
            const filtered = cur.filter(d => d.dispatch_id !== event.dispatch_id)
            return [event, ...filtered].slice(0, 200)
          })
          return
        }
        if (event.event_type === 'traffic.pipeline_status.v1') {
          const det = event.payload?.details || event.details || {}
          setPipelineStatus(det)
          setEvents(cur => [event, ...cur.filter(e => e.id !== event.id)].slice(0, 150))
          return
        }
        setEvents(cur => [event, ...cur.filter(e => e.id !== event.id)].slice(0, 150))
      }
      socket.onclose = () => { setWsOnline(false); retry = setTimeout(connect, 3000) }
      socket.onerror = () => socket.close()
    }
    connect()
    return () => { clearTimeout(retry); socket?.close() }
  }, [])

  const clearSession = async () => {
    try {
      await fetch(`${apiOrigin}/events/clear`, { method:'POST' })
      setEvents([])
      setPipelineStatus({cctv:'IDLE',yolov8:'IDLE',bytetrack:'IDLE',features:'IDLE',risk:'READY',response:'READY',redeployment:'READY'})
    } catch(e) { console.error('Clear failed:', e) }
  }

  const activeSessionId = useMemo(() => {
    const liveEvents = events.filter(e => !e.event_type?.includes('status') && e.session_id)
    return liveEvents[0]?.session_id || null
  }, [events])

  const filteredEvents = useMemo(() => {
    if (!demoMode && activeSessionId)
      return events.filter(e => !e.session_id || e.session_id === activeSessionId)
    return events
  }, [events, demoMode, activeSessionId])

  const junctionCards  = useMemo(() => buildJunctions(filteredEvents, demoMode), [filteredEvents, demoMode])
  const redeployments  = useMemo(() => buildRedeployments(filteredEvents), [filteredEvents])

  const sharedProps = { events:filteredEvents, junctions:junctionCards, historicalJunctions: junctions, demoMode, setDemoMode, apiOnline, wsOnline, clearSession, pipelineStatus, redeployments, dispatches }

  const section = {
    control:  <ReferenceControlRoom {...sharedProps} />,
    dispatch: <DispatchCenter dispatches={dispatches} wsOnline={wsOnline} apiOnline={apiOnline} />,
    map:      <div className="page-grid"><section className="full-section" style={{height:600}}><p className="overline">Real Nagpur CCTV Grid</p><h2>Google Maps Live Risk View</h2><GoogleMapPanel junctionCards={junctionCards} historicalJunctions={junctions} demoMode={demoMode} redeployments={redeployments} /></section></div>,
    redeploy: <RedeploymentPage redeployments={redeployments} junctionCards={junctionCards} wsOnline={wsOnline} />,
    allocation: <OfficerAllocation historicalJunctions={junctions} />,
    traffic:  <LiveTraffic junctions={junctionCards} />,
    analytics:<Analytics events={filteredEvents} />,
    risk:     <RiskAnalysis events={filteredEvents} />,
    system:   <SystemStatus apiOnline={apiOnline} wsOnline={wsOnline} events={filteredEvents} pipelineStatus={pipelineStatus} />,
  }[active]

  return (
    <main className="app-shell">
      <style>{customStyles}</style>
      <aside className="sidebar">
        <div className="brand"><span>TR</span><div><b>TRAFFICRISK</b><small>AI CONTROL ROOM</small></div></div>
        <nav>
          {navItems.map(([id, label, icon]) => {
            const activeDispatchCount = id === 'dispatch'
              ? dispatches.filter(d => ['PENDING','DISPATCHED','ON_SCENE'].includes(d.status)).length
              : 0
            return (
              <button key={id} onClick={() => setActive(id)} className={active===id ? 'active' : ''}>
                <i>{icon}</i>{label}
                {activeDispatchCount > 0 && <span className="nav-badge">{activeDispatchCount}</span>}
              </button>
            )
          })}
        </nav>
        <div className="sidebar-footer">
          <span><i className={apiOnline ? 'pulse' : ''}/> {apiOnline ? 'API ONLINE' : 'API OFFLINE'}</span>
          <small>Nagpur · Local demo</small>
        </div>
      </aside>
      <section className="workspace">
        {!apiOnline && <div className="offline-banner">⚠️ API OFFLINE — Start api_server.py on port 8502</div>}
        {apiOnline && !wsOnline && <div className="offline-banner" style={{background:'#3b2f15',borderBottom:'1px solid #7c5824',color:'#ffe7b4'}}>⚠️ WEBSOCKET OFFLINE — Retrying…</div>}
        <header>
          <div>
            <p className="overline">Operations dashboard {activeSessionId ? `· Session: ${activeSessionId}` : ''}</p>
            <h2>{navItems.find(([id]) => id===active)?.[1]}</h2>
          </div>
          <div className="header-status">
            <span>{filteredEvents.length} events</span>
            <StatusPill label="WebSocket" online={wsOnline} detail={wsOnline?'Live':'Offline'} />
          </div>
        </header>
        <AnimatePresence mode="wait">
          <motion.div key={active} initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-8}} transition={{duration:.18}}>
            {section}
          </motion.div>
        </AnimatePresence>
      </section>
    </main>
  )
}
