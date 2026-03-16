const COLS = {
  blockName:   "block_name",
  district:    "district_n",
  prediction:  "final_prediction",
  confidence:  "rf_confidence",
  cluster:     "cluster",
  solar:       "solar_mean",
  wind:        "wind_mean",
  biomass:     "pop_mean",
  distRoads:   "dist_roads_mean",
  distTrans:   "dist_trans_mean",
  distSub:     "dist_sub_mean",
  constraint:  "constraint_pct"
};

const COLORS = {
  SOLAR:   "#FF8C00",
  WIND:    "#1E90FF",
  BIOMASS: "#32CD32",
  HYBRID:  "#9B59B6",
  UNKNOWN: "#2a3550"
};

let map, blocksLayer;
let blocksData = null;
let currentMode = "allocation";
let confidenceThreshold = 0;
let overlayLayers = { solar: null, wind: null, biomass: null };

// ---- INIT MAP ----
function initMap() {
  map = L.map('map', { center: [20.5, 84.5], zoom: 7 });
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap © CARTO',
    subdomains: 'abcd', maxZoom: 19
  }).addTo(map);
}

// ---- HELPERS ----
function getPrediction(props) {
  return (props[COLS.prediction] || "UNKNOWN").toString().toUpperCase().trim();
}

function getConfidence(props) {
  let c = parseFloat(props[COLS.confidence]) || 0;
  return c <= 1 ? c * 100 : c;  
}
async function loadData()
{
  try 
  {
    const res = await fetch('data/blocks_complete.geojson');
    blocksData = await res.json();
    renderLayer();
    updateStats();
    updateSummary();
    loadHotspot('solar',   'data/top_solar_zones.geojson',   '#FF8C00');
    loadHotspot('wind',    'data/top_wind_zones.geojson',    '#1E90FF');
    loadHotspot('biomass', 'data/top_biomass_zones.geojson', '#32CD32');
  } 
  catch(e) 
  {
    console.error("Failed to load data:", e);
    document.getElementById('summary-content').innerHTML =
      '<p style="color:#ff5555">Error loading data. Open DevTools (F12) for details.</p>';
  }
}
function renderLayer() 
{
  if (blocksLayer) map.removeLayer(blocksLayer);

  if (currentMode === "allocation") 
  {
    blocksLayer = L.geoJSON(blocksData,
    {
      style: styleAllocation,
      onEachFeature: onEachBlock
    }).addTo(map);
  } 
  else
  {
    const colMap = { solar: COLS.solar, wind: COLS.wind, biomass: COLS.biomass };
    const scaleMap = {
      solar:   ['#d73027','#fee08b','#1a9850'],
      wind:    ['#deebf7','#6baed6','#08519c'],
      biomass: ['#ffffcc','#a1dab4','#225ea8']
    };
    const col = colMap[currentMode];
    const vals = blocksData.features.map(f => parseFloat(f.properties[col])).filter(v => !isNaN(v));
    const scale = chroma.scale(scaleMap[currentMode]).domain([Math.min(...vals), Math.max(...vals)]);
    blocksLayer = L.geoJSON(blocksData, {
      style: (f) => {
        const v = parseFloat(f.properties[col]);
        return {
          fillColor: isNaN(v) ? "#1a2035" : scale(v).hex(),
          fillOpacity: 0.75, color: "#1a2540", weight: 0.5
        };
      },
      onEachFeature: onEachBlock
    }).addTo(map);
  }
}
function styleAllocation(feature) {
  const props = feature.properties;
  const pred = getPrediction(props);
  const conf = getConfidence(props);
  const hidden = conf < confidenceThreshold;
  return {
    fillColor:   hidden ? "#1a2035" : (COLORS[pred] || COLORS.UNKNOWN),
    fillOpacity: hidden ? 0.15 : 0.65,
    color:       hidden ? "#222a3a" : (COLORS[pred] || COLORS.UNKNOWN),
    weight: 0.8,
    opacity: hidden ? 0.3 : 0.9
  };
}
function onEachBlock(feature, layer) {
  const p = feature.properties;
  const name = p[COLS.blockName] || "Unknown";
  const pred = getPrediction(p);
  const conf = getConfidence(p).toFixed(1);

  layer.bindTooltip(
    `<strong>${name}</strong><br/>${p[COLS.district] || ""}<br/>
     <span style="color:${COLORS[pred] || '#aaa'}">${pred}</span> — ${conf}% confidence`,
    { sticky: true, opacity: 0.97 }
  );

  layer.on('mouseover', function() {
    this.setStyle({ weight: 2.5, fillOpacity: 0.9 });
    this.bringToFront();
  });
  layer.on('mouseout', function() {
    blocksLayer.resetStyle(this);
  });
  layer.on('click', function() {
    showDetail(p);
  });
}
function showDetail(p) {
  const pred = getPrediction(p);
  const conf = getConfidence(p);

  document.getElementById('detail-default').style.display = 'none';
  document.getElementById('detail-block').style.display  = 'block';

  document.getElementById('detail-block-name').textContent = p[COLS.blockName] || "Unknown";
  document.getElementById('detail-district').textContent   = p[COLS.district] ? "📍 " + p[COLS.district] : "";

  const badge = document.getElementById('detail-prediction-badge');
  badge.textContent  = pred;
  badge.className    = "badge-" + pred;

  const bar = document.getElementById('detail-confidence-bar');
  bar.style.width      = conf + "%";
  bar.style.background = COLORS[pred] || "#aaa";
  document.getElementById('detail-confidence-text').textContent = conf.toFixed(1) + "%";

  document.getElementById('detail-cluster').textContent =
    "Cluster " + (p[COLS.cluster] ?? "N/A");
  const features = [
    ["Solar Mean",        p[COLS.solar]],
    ["Wind Mean",         p[COLS.wind]],
    ["Population Mean",   p[COLS.biomass]],
    ["Dist. to Roads",    p[COLS.distRoads]],
    ["Dist. to Trans.",   p[COLS.distTrans]],
    ["Dist. to Subst.",   p[COLS.distSub]],
    ["Constraint %",      p[COLS.constraint]]
  ];

  document.getElementById('detail-features-table').innerHTML =
    features.map(([label, val]) =>
      val != null
        ? `<tr><td>${label}</td><td>${parseFloat(val).toFixed(3)}</td></tr>`
        : ""
    ).join("");
}

function closeDetail() {
  document.getElementById('detail-default').style.display = 'block';
  document.getElementById('detail-block').style.display   = 'none';
}
function switchLayer(mode) {
  currentMode = mode;
  renderLayer();
  document.getElementById('legend-box').style.display =
    mode === 'allocation' ? 'block' : 'none';
}
function filterByConfidence(val)
{
  confidenceThreshold = parseFloat(val);
  document.getElementById('conf-value').textContent = val + "%";
  if (currentMode === 'allocation') renderLayer();
  updateStats();
}
async function loadHotspot(type, url, color) {
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();
    overlayLayers[type] = L.geoJSON(data, {
      style: { color, weight: 2, fillColor: color, fillOpacity: 0.2, dashArray: '5 5' }
    });
  } catch(e) { console.log("Hotspot not loaded:", type); }
}

function toggleOverlay(type) {
  const layer = overlayLayers[type];
  if (!layer) return;
  if (map.hasLayer(layer)) map.removeLayer(layer);
  else layer.addTo(map);
}
function updateStats() {
  if (!blocksData) return;
  const counts = { SOLAR:0, WIND:0, BIOMASS:0, HYBRID:0 };
  let highConf = 0, showing = 0;

  blocksData.features.forEach(f => {
    const pred = getPrediction(f.properties);
    const conf = getConfidence(f.properties);
    if (counts[pred] !== undefined) counts[pred]++;
    if (conf >= 80) highConf++;
    if (conf >= confidenceThreshold) showing++;
  });

  const total = blocksData.features.length;
  document.getElementById('stat-solar').textContent    = counts.SOLAR;
  document.getElementById('stat-wind').textContent     = counts.WIND;
  document.getElementById('stat-biomass').textContent  = counts.BIOMASS;
  document.getElementById('stat-hybrid').textContent   = counts.HYBRID;
  document.getElementById('stat-highconf').textContent =
    `${highConf} (${((highConf/total)*100).toFixed(1)}%)`;
  document.getElementById('stat-showing').textContent  = showing;
}
function updateSummary() {
  if (!blocksData) return;
  const counts = { SOLAR:0, WIND:0, BIOMASS:0, HYBRID:0 };
  blocksData.features.forEach(f => {
    const pred = getPrediction(f.properties);
    if (counts[pred] !== undefined) counts[pred]++;
  });
  document.getElementById('summary-content').innerHTML = `
    <p>☀️ Solar blocks: <strong style="color:#FF8C00">${counts.SOLAR}</strong></p>
    <p>💨 Wind blocks: <strong style="color:#1E90FF">${counts.WIND}</strong></p>
    <p>🌿 Biomass blocks: <strong style="color:#32CD32">${counts.BIOMASS}</strong></p>
    <p>⚡ Hybrid blocks: <strong style="color:#9B59B6">${counts.HYBRID}</strong></p>
    <p>📊 Total: <strong style="color:#aac4ff">${blocksData.features.length}</strong> blocks</p>
  `;
}
function showTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  document.getElementById('btn-' + tab).classList.add('active');
  if (tab === 'map') setTimeout(() => map && map.invalidateSize(), 100);
}
// ADD THIS at the bottom of app.js
document.addEventListener('DOMContentLoaded', function () {
  initMap();
  loadData();
});