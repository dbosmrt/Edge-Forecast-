import { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, LayersControl, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import 'leaflet.heat'
import './App.css'

const API_URL = 'http://localhost:8000'

// Region centers (demo)
const REGIONS = {
  'Uttarakhand, India': { center: [30.0668, 79.0193], zoom: 8 },
  'Himachal Pradesh, India': { center: [31.1048, 77.1734], zoom: 8 },
  'California, USA': { center: [36.7783, -119.4179], zoom: 6 },
  'New South Wales, Australia': { center: [-32.0, 147.0], zoom: 6 },
}

// Heatmap component
function HeatmapLayer({ points, show }) {
  const map = useMap()
  const heatLayerRef = useRef(null)

  useEffect(() => {
    if (!show || !points.length) {
      if (heatLayerRef.current) {
        map.removeLayer(heatLayerRef.current)
        heatLayerRef.current = null
      }
      return
    }

    const heatData = points.map(p => [p.lat, p.lon, p.probability])

    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current)
    }

    heatLayerRef.current = L.heatLayer(heatData, {
      radius: 15,
      blur: 20,
      maxZoom: 10,
      max: 1.0,
      gradient: {
        0.0: '#00ff00',
        0.3: '#ffff00',
        0.6: '#ff9900',
        1.0: '#ff0000'
      }
    }).addTo(map)

    return () => {
      if (heatLayerRef.current) {
        map.removeLayer(heatLayerRef.current)
      }
    }
  }, [points, show, map])

  return null
}

function App() {
  const [darkMode, setDarkMode] = useState(false)
  const [riskData, setRiskData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeSection, setActiveSection] = useState('map')
  const [showLogin, setShowLogin] = useState(false)
  const [showDatasetModal, setShowDatasetModal] = useState(false)

  // Display options
  const [viewMode, setViewMode] = useState('points') // 'points' or 'heatmap'
  const [showHigh, setShowHigh] = useState(true)
  const [showMedium, setShowMedium] = useState(true)
  const [showLow, setShowLow] = useState(true)
  const [maxPoints, setMaxPoints] = useState(2000) // Limit for performance
  const [selectedRegion, setSelectedRegion] = useState('Uttarakhand, India')

  useEffect(() => {
    fetchRiskData()
  }, [])

  const fetchRiskData = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/api/risk-tiles?limit=${maxPoints}`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setRiskData(data)
    } catch (err) {
      setError(`Connection failed: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'HIGH': return '#F44336'
      case 'MEDIUM': return '#FF9800'
      case 'LOW': return '#4CAF50'
      default: return '#9E9E9E'
    }
  }

  const formatTimestamp = (ts) => {
    if (!ts) return 'N/A'
    try { return new Date(ts).toLocaleString() }
    catch { return ts }
  }

  // Filter cells
  const filteredCells = (riskData?.grid_cells || []).filter(cell => {
    if (cell.risk === 'HIGH' && !showHigh) return false
    if (cell.risk === 'MEDIUM' && !showMedium) return false
    if (cell.risk === 'LOW' && !showLow) return false
    return true
  })

  const region = REGIONS[selectedRegion]

  return (
    <div className={`app ${darkMode ? 'dark' : 'light'}`}>
      {/* Navigation */}
      <nav className="navbar compact">
        <div className="nav-brand">
          <span className="logo-icon">F</span>
          <h1>Edge-Forecast</h1>
        </div>

        <div className="nav-links">
          <button className={activeSection === 'map' ? 'active' : ''} onClick={() => setActiveSection('map')}>Map</button>
          <button className={activeSection === 'about' ? 'active' : ''} onClick={() => setActiveSection('about')}>About</button>
          <button className={activeSection === 'dataset' ? 'active' : ''} onClick={() => setActiveSection('dataset')}>Dataset</button>
          <a href="https://github.com/dbosmrt/Edge-Forecast-" target="_blank" rel="noopener noreferrer" className="github-link">GitHub</a>
        </div>

        <div className="nav-actions">
          <button className="icon-btn" onClick={() => setDarkMode(!darkMode)} title="Toggle theme">
            {darkMode ? 'Light' : 'Dark'}
          </button>
          <button className="icon-btn" onClick={fetchRiskData} title="Refresh">Refresh</button>
          <button className="auth-btn" onClick={() => setShowLogin(true)}>Login</button>
        </div>
      </nav>

      {/* Login Modal */}
      {showLogin && (
        <div className="modal-overlay" onClick={() => setShowLogin(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2>Login / Sign Up</h2>
            <form onSubmit={e => { e.preventDefault(); alert('Login feature coming soon!'); setShowLogin(false); }}>
              <input type="text" placeholder="Full Name" required />
              <input type="email" placeholder="Email" required />
              <input type="password" placeholder="Password" required />
              <div className="modal-actions">
                <button type="submit" className="btn-primary">Login</button>
                <button type="button" className="btn-secondary" onClick={() => setShowLogin(false)}>Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Dataset Modal */}
      {showDatasetModal && (
        <div className="modal-overlay" onClick={() => setShowDatasetModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2>Upload Dataset</h2>
            <p>Upload CSV with environmental data for prediction.</p>
            <div className="upload-box">
              <input type="file" accept=".csv" onChange={() => { alert('Upload feature coming soon!'); setShowDatasetModal(false); }} />
              <span>Choose CSV File</span>
            </div>
            <p className="hint">Required: latitude, longitude, NDVI, temperature, etc.</p>
            <button className="btn-secondary" onClick={() => setShowDatasetModal(false)}>Cancel</button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="main-content">
        {activeSection === 'map' && (
          <section className="map-section compact">
            {/* Control Bar */}
            <div className="control-bar">
              {/* Region Selector */}
              <div className="control-group">
                <label>Region:</label>
                <select value={selectedRegion} onChange={e => setSelectedRegion(e.target.value)}>
                  {Object.keys(REGIONS).map(r => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>

              {/* View Mode */}
              <div className="control-group">
                <label>View:</label>
                <select value={viewMode} onChange={e => setViewMode(e.target.value)}>
                  <option value="points">Points</option>
                  <option value="heatmap">Heatmap</option>
                </select>
              </div>

              {/* Risk Filters */}
              <div className="control-group filters">
                <label className={`chip ${showHigh ? 'active high' : ''}`}>
                  <input type="checkbox" checked={showHigh} onChange={() => setShowHigh(!showHigh)} />
                  High ({riskData?.metadata?.risk_distribution?.HIGH?.count || 0})
                </label>
                <label className={`chip ${showMedium ? 'active medium' : ''}`}>
                  <input type="checkbox" checked={showMedium} onChange={() => setShowMedium(!showMedium)} />
                  Medium ({riskData?.metadata?.risk_distribution?.MEDIUM?.count || 0})
                </label>
                <label className={`chip ${showLow ? 'active low' : ''}`}>
                  <input type="checkbox" checked={showLow} onChange={() => setShowLow(!showLow)} />
                  Low ({riskData?.metadata?.risk_distribution?.LOW?.count || 0})
                </label>
              </div>

              {/* Point Limit */}
              <div className="control-group">
                <label>Max points:</label>
                <select value={maxPoints} onChange={e => { setMaxPoints(Number(e.target.value)); }}>
                  <option value={500}>500</option>
                  <option value={1000}>1,000</option>
                  <option value={2000}>2,000</option>
                  <option value={5000}>5,000</option>
                  <option value={10000}>10,000</option>
                </select>
                <button className="btn-small" onClick={fetchRiskData}>Apply</button>
              </div>

              {/* Status */}
              <div className="status">
                {loading ? 'Loading...' : error ? <span className="error">{error}</span> : `${filteredCells.length} points`}
              </div>

              <button className="btn-small" onClick={() => setShowDatasetModal(true)}>Upload Data</button>
            </div>

            {/* Map */}
            <div className="map-container full">
              <MapContainer
                key={selectedRegion}
                center={region.center}
                zoom={region.zoom}
                style={{ height: '100%', width: '100%' }}
                scrollWheelZoom={true}
              >
                <LayersControl position="topright">
                  <LayersControl.BaseLayer checked name="Street">
                    <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  </LayersControl.BaseLayer>
                  <LayersControl.BaseLayer name="Satellite">
                    <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" />
                  </LayersControl.BaseLayer>
                  <LayersControl.BaseLayer name="Hybrid">
                    <TileLayer url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}" />
                  </LayersControl.BaseLayer>
                  <LayersControl.BaseLayer name="Terrain">
                    <TileLayer url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png" />
                  </LayersControl.BaseLayer>
                  <LayersControl.BaseLayer name="Dark">
                    <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />
                  </LayersControl.BaseLayer>
                </LayersControl>

                {/* Heatmap Layer */}
                <HeatmapLayer points={filteredCells} show={viewMode === 'heatmap'} />

                {/* Points Layer */}
                {viewMode === 'points' && filteredCells.map((cell, i) => (
                  <CircleMarker
                    key={i}
                    center={[cell.lat, cell.lon]}
                    radius={5}
                    pathOptions={{
                      fillColor: getRiskColor(cell.risk),
                      fillOpacity: 0.8,
                      color: '#fff',
                      weight: 1
                    }}
                  >
                    <Popup>
                      <div className="popup-content">
                        <strong style={{ color: getRiskColor(cell.risk) }}>{cell.risk} RISK</strong>
                        <p>Probability: {(cell.probability * 100).toFixed(1)}%</p>
                        <p>Lat: {cell.lat.toFixed(4)}</p>
                        <p>Lon: {cell.lon.toFixed(4)}</p>
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
              </MapContainer>
            </div>
          </section>
        )}

        {activeSection === 'about' && (
          <section className="page-section">
            <h2>About Edge-Forecast</h2>
            <p>ML-powered fire risk prediction using satellite imagery, weather data, and historical fire patterns.</p>
            <div className="feature-grid">
              <div className="feature"><h4>Satellite Data</h4><p>MODIS, Landsat</p></div>
              <div className="feature"><h4>Weather</h4><p>Temperature, wind, precip</p></div>
              <div className="feature"><h4>Fire History</h4><p>Past burn patterns</p></div>
              <div className="feature"><h4>ML Model</h4><p>98.5% ROC-AUC</p></div>
            </div>
          </section>
        )}

        {activeSection === 'dataset' && (
          <section className="page-section">
            <h2>Dataset Information</h2>
            <p><strong>Sources:</strong> MODIS, Landsat, ERA5, SRTM, FIRMS</p>
            <p><strong>Coverage:</strong> Uttarakhand, India | ~29,000 grid cells | ~1km resolution</p>
            <button className="btn-primary" onClick={() => setShowDatasetModal(true)}>Upload Your Own Dataset</button>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="footer compact">
        <span>Updated: {formatTimestamp(riskData?.metadata?.timestamp)}</span>
        <span>|</span>
        <span>v{riskData?.metadata?.model_version || '1.0.0'}</span>
        <span>|</span>
        <span>Edge-Forecast 2026</span>
      </footer>
    </div>
  )
}

export default App
