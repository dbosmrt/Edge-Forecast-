<div align="center">
  <h1>Edge-Forecast</h1>
  <p><strong>Machine Learning-Based Forest Fire Risk Prediction System for Uttarakhand, India</strong></p>
  <p>
    <a href="#problem-statement">Problem Statement</a> |
    <a href="#methodology">Methodology</a> |
    <a href="#features">Features</a> |
    <a href="#setup">Setup</a> |
    <a href="#tech-stack">Tech Stack</a>
  </p>
</div>

---

<h2 id="problem-statement">Problem Statement</h2>

<p>
Forest fires in Uttarakhand, India, pose a significant threat to biodiversity, local communities, and the environment. The state experiences hundreds of fire incidents annually, particularly during the dry summer months (March-June). Traditional fire monitoring relies on reactive approaches—detecting fires after they have already started—which limits the effectiveness of response efforts.
</p>

<p>
<strong>Edge-Forecast addresses this challenge by providing a proactive, predictive system that:</strong>
</p>

<ul>
  <li>Identifies high-risk areas <em>before</em> fires occur using satellite imagery, weather data, and historical fire patterns</li>
  <li>Enables forest departments and disaster management authorities to allocate resources strategically</li>
  <li>Provides a visual interface for monitoring fire risk across the entire state</li>
  <li>Supports decision-making for preventive measures such as fire lines, controlled burns, and patrol scheduling</li>
</ul>

---

<h2 id="methodology">Methodology</h2>

<h3>1. Data Collection</h3>

<p>The system aggregates multi-source geospatial data using Google Earth Engine (GEE):</p>

<table>
  <thead>
    <tr>
      <th>Data Source</th>
      <th>Variables</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MODIS (Terra/Aqua)</td>
      <td>NDVI, EVI</td>
      <td>Vegetation health and fuel load</td>
    </tr>
    <tr>
      <td>Landsat 8</td>
      <td>Land Surface Temperature (LST)</td>
      <td>Surface heat patterns</td>
    </tr>
    <tr>
      <td>ERA5 Reanalysis</td>
      <td>Temperature, Wind Speed, Precipitation</td>
      <td>Weather conditions</td>
    </tr>
    <tr>
      <td>SRTM</td>
      <td>Elevation, Slope, Aspect</td>
      <td>Terrain characteristics</td>
    </tr>
    <tr>
      <td>MODIS MCD64A1</td>
      <td>Burned Area</td>
      <td>Historical fire footprints</td>
    </tr>
    <tr>
      <td>FIRMS</td>
      <td>Active Fire Detections</td>
      <td>Fire occurrence labels</td>
    </tr>
  </tbody>
</table>

<h3>2. Feature Engineering</h3>

<p>Raw satellite data is transformed into predictive features:</p>

<ul>
  <li><strong>Vegetation indices:</strong> NDVI, EVI indicating biomass and fuel availability</li>
  <li><strong>Thermal features:</strong> Day/night LST, temperature anomalies</li>
  <li><strong>Weather features:</strong> Wind speed, precipitation deficit, humidity</li>
  <li><strong>Terrain features:</strong> Slope, aspect, elevation (affects fire spread)</li>
  <li><strong>Historical features:</strong> Burn count in last 5 years, years since last burn</li>
  <li><strong>Land cover:</strong> Forest type, vegetation density</li>
</ul>

<h3>3. Machine Learning Model</h3>

<p>The prediction pipeline uses a <strong>Gradient Boosting Classifier</strong> trained on historical fire occurrence data:</p>

<table>
  <thead>
    <tr>
      <th>Component</th>
      <th>Implementation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Algorithm</td>
      <td>Gradient Boosting Classifier (scikit-learn)</td>
    </tr>
    <tr>
      <td>Class Imbalance</td>
      <td>RandomOverSampler (imblearn)</td>
    </tr>
    <tr>
      <td>Train/Test Split</td>
      <td>80/20 stratified</td>
    </tr>
    <tr>
      <td>Hyperparameters</td>
      <td>n_estimators=100, max_depth=5, learning_rate=0.1</td>
    </tr>
  </tbody>
</table>

<p><strong>Model Performance:</strong></p>

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ROC-AUC</td>
      <td><strong>0.9851</strong></td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>0.43</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>0.78</td>
    </tr>
    <tr>
      <td>F1-Score</td>
      <td>0.55</td>
    </tr>
  </tbody>
</table>

<h3>4. Risk Classification</h3>

<p>Model outputs are raw probabilities (0-1). A heuristic policy converts these to actionable risk categories:</p>

<table>
  <thead>
    <tr>
      <th>Risk Level</th>
      <th>Probability Range</th>
      <th>Color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LOW</td>
      <td>0.0 - 0.3</td>
      <td>Green (#4CAF50)</td>
    </tr>
    <tr>
      <td>MEDIUM</td>
      <td>0.3 - 0.6</td>
      <td>Orange (#FF9800)</td>
    </tr>
    <tr>
      <td>HIGH</td>
      <td>0.6 - 1.0</td>
      <td>Red (#F44336)</td>
    </tr>
  </tbody>
</table>

<p>These thresholds are configurable without retraining the model.</p>

---

<h2 id="features">Features</h2>

<h3>Web Application</h3>

<ul>
  <li><strong>Interactive Map:</strong> View fire risk across 29,000+ grid cells with Leaflet/OpenStreetMap</li>
  <li><strong>Multiple Views:</strong> Switch between point markers and heatmap visualization</li>
  <li><strong>Layer Options:</strong> Street map, satellite imagery, hybrid, terrain, dark mode</li>
  <li><strong>Risk Filters:</strong> Toggle visibility of High, Medium, and Low risk zones</li>
  <li><strong>Click Details:</strong> Click any point to see probability, coordinates, and risk level</li>
  <li><strong>Performance Controls:</strong> Limit displayed points (500-10,000) for smooth rendering</li>
  <li><strong>Dark/Light Theme:</strong> Toggle between themes</li>
  <li><strong>Region Selector:</strong> Demo support for multiple regions</li>
</ul>

<h3>Backend API</h3>

<ul>
  <li><strong>RESTful Endpoints:</strong> FastAPI-based API serving risk predictions</li>
  <li><strong>Smart Pagination:</strong> Prioritizes HIGH and MEDIUM risk points in limited queries</li>
  <li><strong>Filtering:</strong> Filter by risk level, limit count</li>
  <li><strong>CORS Enabled:</strong> Accessible from any frontend origin</li>
</ul>

---

<h2 id="setup">Setup and Installation</h2>

<h3>Prerequisites</h3>

<ul>
  <li>Python 3.9+</li>
  <li>Node.js 18+</li>
  <li>Git</li>
</ul>

<h3>1. Clone Repository</h3>

<pre><code>git clone https://github.com/dbosmrt/Edge-Forecast-.git
cd Edge-Forecast-
</code></pre>

<h3>2. Install Python Dependencies</h3>

<pre><code>pip install -r requirements.txt
</code></pre>

<h3>3. Install Node.js Dependencies</h3>

<pre><code>cd client
npm install
cd ..
</code></pre>

<h3>4. Run the ML Pipeline (First Time Only)</h3>

<pre><code>cd model
python train_model.py
python evaluate_model.py
cd ..
</code></pre>

<h3>5. Generate Risk Tiles</h3>

<pre><code>cd server
python pipeline.py
cd ..
</code></pre>

<h3>6. Start the Servers</h3>

<p>Terminal 1 - API Server:</p>
<pre><code>cd server
python api.py
</code></pre>

<p>Terminal 2 - Frontend:</p>
<pre><code>cd client
npm run dev
</code></pre>

<h3>7. Access the Application</h3>

<ul>
  <li>Frontend: <a href="http://localhost:5173">http://localhost:5173</a></li>
  <li>API Docs: <a href="http://localhost:8000/docs">http://localhost:8000/docs</a></li>
</ul>

---

<h2 id="tech-stack">Technology Stack</h2>

<table>
  <thead>
    <tr>
      <th>Layer</th>
      <th>Technology</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data Collection</td>
      <td>Google Earth Engine (Python API)</td>
      <td>Satellite data extraction</td>
    </tr>
    <tr>
      <td>Data Processing</td>
      <td>Pandas, NumPy</td>
      <td>Feature engineering</td>
    </tr>
    <tr>
      <td>Machine Learning</td>
      <td>scikit-learn, imbalanced-learn</td>
      <td>Model training and evaluation</td>
    </tr>
    <tr>
      <td>Model Persistence</td>
      <td>joblib</td>
      <td>Save/load trained models</td>
    </tr>
    <tr>
      <td>API Server</td>
      <td>FastAPI, Uvicorn</td>
      <td>REST API endpoints</td>
    </tr>
    <tr>
      <td>Frontend Framework</td>
      <td>React, Vite</td>
      <td>Web application</td>
    </tr>
    <tr>
      <td>Mapping</td>
      <td>Leaflet, react-leaflet, leaflet.heat</td>
      <td>Interactive maps and heatmaps</td>
    </tr>
    <tr>
      <td>Styling</td>
      <td>CSS (custom)</td>
      <td>Dark/light themes</td>
    </tr>
  </tbody>
</table>

---

<h2 id="project-structure">Project Structure</h2>

<pre><code>Edge-Forecast-/
|-- model/                    # ML training and evaluation
|   |-- preprocess.py         # Feature engineering
|   |-- train_model.py        # Model training with data leakage fix
|   |-- evaluate_model.py     # Metrics and visualization
|
|-- server/                   # Backend API and inference pipeline
|   |-- api.py                # FastAPI endpoints
|   |-- pipeline.py           # Inference orchestrator
|   |-- config.py             # Settings and thresholds
|   |-- feature_builder.py    # Preprocessing (matches training)
|   |-- inference.py          # Model prediction
|   |-- heuristics.py         # Risk label assignment
|   |-- risk_tiles.py         # Output formatting
|   |-- output/               # Generated risk tiles JSON
|
|-- client/                   # React frontend
|   |-- src/
|       |-- App.jsx           # Main application
|       |-- App.css           # Styles
|
|-- artifacts/                # Trained model and metrics
|-- dataset/                  # Training data (gitignored)
|-- requirements.txt          # Python dependencies
</code></pre>

---

<h2 id="limitations">Limitations</h2>

<ol>
  <li><strong>Temporal Lag:</strong> Satellite data updates every 1-16 days depending on the sensor. The system cannot detect real-time changes.</li>
  <li><strong>Precision-Recall Tradeoff:</strong> With 0.43 precision and 0.78 recall, the model has more false positives than false negatives. This is intentional—missing a fire is worse than a false alarm.</li>
  <li><strong>Geographic Scope:</strong> Currently trained only on Uttarakhand data. Applying to other regions requires retraining.</li>
  <li><strong>Resolution:</strong> Predictions are at approximately 1km x 1km resolution, which may miss localized fire risks.</li>
  <li><strong>Weather Dependency:</strong> Model performance degrades with cloud cover (missing optical satellite data).</li>
  <li><strong>Static Predictions:</strong> Current implementation generates predictions on demand rather than continuous real-time updates.</li>
</ol>

---

<h2 id="future-improvements">Future Improvements</h2>

<ol>
  <li><strong>Real-time Integration:</strong> Connect to live weather APIs and FIRMS active fire detection for continuous monitoring.</li>
  <li><strong>Deep Learning:</strong> Explore CNN/LSTM architectures for spatial-temporal pattern recognition.</li>
  <li><strong>Multi-region Support:</strong> Extend to other fire-prone regions in India (Himachal Pradesh, Madhya Pradesh).</li>
  <li><strong>Mobile Application:</strong> Develop a mobile app for field officers to receive alerts and report observations.</li>
  <li><strong>Alert System:</strong> Implement email/SMS notifications when high-risk zones are detected.</li>
  <li><strong>User Authentication:</strong> Add proper user management for access control and personalized views.</li>
  <li><strong>Custom Data Upload:</strong> Allow users to upload their own environmental data for predictions.</li>
  <li><strong>Higher Resolution:</strong> Integrate Sentinel-2 data for 10m resolution predictions.</li>
  <li><strong>Explainability:</strong> Add SHAP/LIME explanations for why a particular area is high-risk.</li>
</ol>

---

<h2 id="api-reference">API Reference</h2>

<table>
  <thead>
    <tr>
      <th>Endpoint</th>
      <th>Method</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>/api/risk-tiles</code></td>
      <td>GET</td>
      <td>Get risk predictions with optional limit and risk_level filters</td>
    </tr>
    <tr>
      <td><code>/api/risk-tiles/summary</code></td>
      <td>GET</td>
      <td>Get risk distribution summary without full data</td>
    </tr>
    <tr>
      <td><code>/api/health</code></td>
      <td>GET</td>
      <td>Health check endpoint</td>
    </tr>
    <tr>
      <td><code>/api/metadata</code></td>
      <td>GET</td>
      <td>Model version and timestamp</td>
    </tr>
  </tbody>
</table>

<p><strong>Example Request:</strong></p>

<pre><code>GET /api/risk-tiles?limit=1000

Response:
{
  "metadata": {
    "model_version": "1.0.0",
    "timestamp": "2026-02-01T06:51:33+00:00",
    "total_cells": 29122,
    "risk_distribution": {
      "HIGH": {"count": 325, "percentage": 1.12},
      "MEDIUM": {"count": 410, "percentage": 1.41},
      "LOW": {"count": 28387, "percentage": 97.48}
    }
  },
  "grid_cells": [
    {"lat": 30.123, "lon": 79.456, "probability": 0.72, "risk": "HIGH", "color": "#F44336"}
  ],
  "count": 1000
}
</code></pre>

---

<h2 id="references">Key Research References</h2>

<ul>
  <li>
    Jain, S., et al. (2018).
    <em>Forest fire risk assessment in India using remote sensing and GIS.</em>
    Current Science.<br>
    <a href="https://www.currentscience.ac.in/Volumes/114/03/0581.pdf" target="_blank">
      https://www.currentscience.ac.in/Volumes/114/03/0581.pdf
    </a>
  </li>

  <li>
    Chuvieco, E., et al. (2010).
    <em>Integration of ecological and meteorological information for fire risk mapping.</em>
    Remote Sensing of Environment.<br>
    <a href="https://www.sciencedirect.com/science/article/pii/S0034425710000905" target="_blank">
      https://www.sciencedirect.com/science/article/pii/S0034425710000905
    </a>
  </li>

  <li>
    Abatzoglou, J. T., and Williams, A. P. (2016).
    <em>Impact of anthropogenic climate change on wildfire across western US forests.</em>
    Proceedings of the National Academy of Sciences (PNAS).<br>
    <a href="https://www.pnas.org/doi/10.1073/pnas.1607171113" target="_blank">
      https://www.pnas.org/doi/10.1073/pnas.1607171113
    </a>
  </li>

  <li>
    Jain, P., et al. (2020).
    <em>Machine learning applications in wildfire science.</em>
    Remote Sensing.<br>
    <a href="https://www.mdpi.com/2072-4292/12/17/2820" target="_blank">
      https://www.mdpi.com/2072-4292/12/17/2820
    </a>
  </li>

  <li>
    ISPRS Archives (2023).
    <em>Forest fire risk mapping in Uttarakhand using Google Earth Engine.</em><br>
    <a href="https://isprs-archives.copernicus.org/articles/XLVIII-M-3-2023/27/2023/" target="_blank">
      https://isprs-archives.copernicus.org/articles/XLVIII-M-3-2023/27/2023/
    </a>
  </li>
</ul>

---

<h2 id="license">License</h2>

<p>This project is developed for research and educational purposes.</p>

---

<h2 id="acknowledgments">Acknowledgments</h2>

<ul>
  <li>Google Earth Engine for providing access to satellite imagery</li>
  <li>NASA FIRMS for active fire detection data</li>
  <li>OpenStreetMap contributors for base map tiles</li>
</ul>
