"""
SowSmart — Precision AI Farming
Streamlit app integrating:
  - OpenCV image segmentation (vision/segmentation.py)
  - Feature extraction: soil quality, moisture, NDVI (vision/features.py)
  - K-Means agronomic zone detection (vision/zone_detection.py)
  - DEAP NSGA-II genetic algorithm (genetic_algorithm/engine.py)
  - Open-Meteo rainfall data (data/rain_data.py)
  - Pakistani crop database (data/crops.py)
"""

import io
import time
import tempfile
import os

import numpy as np
import cv2
import streamlit as st
from streamlit_geolocation import streamlit_geolocation  # type: ignore
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ── project imports ──────────────────────────────────────────────────────────
from data.crops import CROPS, CROP_NAMES, NUM_CROPS
from data.rain_data import get_effective_rain
from vision.segmenter import segment_image
from vision.features import get_features, get_ndvi_proxy
from vision.zone_detection import detect_zones
from genetic_algorithm.engine import run_ga

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SowSmart · Precision AI Farming",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp {
    background: #0b1a12;
    color: #d4e8c2;
}

section[data-testid="stSidebar"] {
    background: #0f2318 !important;
    border-right: 1px solid #1e3a28;
}

.metric-card {
    background: #112219;
    border: 1px solid #1e3a28;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 6px 0;
}

.metric-card .label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a8a6a;
}

.metric-card .value {
    font-size: 22px;
    font-weight: 500;
    color: #9de87a;
    font-family: 'Syne', sans-serif;
}

.stButton > button {
    background: #2d6a3f !important;
    color: #d4e8c2 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.06em !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: background 0.2s ease !important;
}

.stButton > button:hover {
    background: #3d8a55 !important;
}

.stSlider > div > div > div { accent-color: #5cb87a; }

div[data-testid="stExpander"] {
    border: 1px solid #1e3a28 !important;
    border-radius: 4px !important;
    background: #0f2318 !important;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3d6e50;
    border-bottom: 1px solid #1e3a28;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}

.gen-log-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 3px 0;
    border-bottom: 1px solid #1a2e20;
    color: #8ab89a;
}

.crop-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin: 4px 8px 4px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── colour palette for crops ──────────────────────────────────────────────────
CROP_COLORS = {
    "Wheat":     "#e8c97a",
    "Cotton":    "#f0f0e8",
    "Rice":      "#7ab8e8",
    "Sugarcane": "#8ae878",
    "Maize":     "#f0b85a",
    "Sunflower": "#f5d84a",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def fig_to_image(fig):
    """Convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)


def draw_crop_grid(grid: np.ndarray) -> plt.Figure:
    """Render the N×N crop assignment grid as a coloured heatmap."""
    n = grid.shape[0]
    color_grid = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            name = CROP_NAMES[grid[i, j]]
            hex_c = CROP_COLORS.get(name, "#888888")
            color_grid[i, j] = mcolors.to_rgb(hex_c)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0b1a12")
    ax.imshow(color_grid, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(1, n + 1), fontsize=7, color="#5a8a6a")
    ax.set_yticklabels(range(1, n + 1), fontsize=7, color="#5a8a6a")
    ax.tick_params(length=0)
    ax.set_facecolor("#0b1a12")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a28")

    patches = [
        mpatches.Patch(color=CROP_COLORS[c], label=c) for c in CROP_NAMES
    ]
    ax.legend(
        handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
        framealpha=0, labelcolor="#d4e8c2", fontsize=9,
    )
    ax.set_title("Optimal Crop Placement", color="#9de87a",
                 fontsize=13, pad=10, fontweight="bold")
    fig.tight_layout()
    return fig


def draw_zone_grid(zone_labels: np.ndarray) -> plt.Figure:
    """Render the K-Means zone map."""
    zone_cmap = plt.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0b1a12")
    ax.imshow(zone_labels, cmap=zone_cmap, interpolation="nearest")
    ax.set_title("Agronomic Zones (K-Means)", color="#9de87a", fontsize=11, pad=8)
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_feature_map(arr: np.ndarray, title: str, cmap: str = "YlGn") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0b1a12")
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, color="#9de87a", fontsize=11, pad=8)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#5a8a6a", labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#5a8a6a")
    cbar.outline.set_edgecolor("#1e3a28")
    fig.tight_layout()
    return fig


def draw_fitness_log(log: list) -> plt.Figure:
    gens  = [e["gen"] for e in log]
    best  = [e["best_fit"] for e in log]
    avg   = [e["average_fit"] for e in log]

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0b1a12")
    ax.set_facecolor("#0f2318")
    ax.plot(gens, best, color="#9de87a", linewidth=1.8, label="Best fitness")
    ax.plot(gens, avg,  color="#5a8a6a", linewidth=1.2, linestyle="--", label="Avg fitness")
    ax.set_xlabel("Generation", color="#5a8a6a", fontsize=9)
    ax.set_ylabel("Fitness (profit)", color="#5a8a6a", fontsize=9)
    ax.set_title("GA Convergence", color="#9de87a", fontsize=11)
    ax.tick_params(colors="#5a8a6a", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a28")
    ax.legend(fontsize=8, framealpha=0, labelcolor="#d4e8c2")
    fig.tight_layout()
    return fig


def compute_crop_distribution(grid: np.ndarray) -> dict:
    flat = grid.flatten()
    total = len(flat)
    dist = {}
    for idx, name in enumerate(CROP_NAMES):
        count = int(np.sum(flat == idx))
        dist[name] = {"count": count, "pct": round(count / total * 100, 1)}
    return dist


def estimate_total_profit(grid: np.ndarray) -> float:
    return sum(
        CROPS[CROP_NAMES[grid[i, j]]]["profit"]
        for i in range(grid.shape[0])
        for j in range(grid.shape[1])
    )


def estimate_water_need(grid: np.ndarray, lat=None, lon=None) -> float:
    n = grid.shape[0]
    raw = sum(
        CROPS[CROP_NAMES[grid[i, j]]]["water"]
        for i in range(n) for j in range(n)
    )
    rain = get_effective_rain(n * n, lat, lon)
    return max(0.0, raw - rain)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌱 SowSmart")
    st.markdown('<div class="section-header">Location</div>', unsafe_allow_html=True)
    st.write("Sync your farm's GPS coordinates for local rainfall data.")
    location = streamlit_geolocation()

    lat, lon = None, None
    if location and location.get("latitude") is not None:
        lat = location["latitude"]
        lon = location["longitude"]
        st.success(f"📍 Lat {lat:.4f} · Lon {lon:.4f}")
    else:
        st.info("GPS unavailable — rainfall offset disabled.")

    st.markdown('<div class="section-header">Grid & GA Parameters</div>', unsafe_allow_html=True)
    grid_n    = st.slider("Grid Size (N × N)", 5, 20, 10, 1)
    pop_size  = st.slider("Population Size",   50, 500, 150, 10)
    gen_size  = st.slider("Generations",        50, 500, 100, 10)
    cx_prob   = st.slider("Crossover Prob.",    0.3, 1.0, 0.7, 0.05)
    mut_prob  = st.slider("Mutation Prob.",     0.01, 0.5, 0.1, 0.01)
    penalty_w = st.slider("Soil Penalty Weight", 0.5, 5.0, 2.0, 0.5)

    st.markdown('<div class="section-header">Farm Image</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload satellite / drone image",
        type=["jpeg", "jpg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_image:
        st.image(Image.open(uploaded_image), caption="Uploaded image", use_container_width=True)

    run_btn = st.button("▶  Run Optimisation")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🌾 SowSmart · Precision Crop Optimiser")
st.markdown(
    "Upload a farm image, tune the parameters in the sidebar, "
    "then hit **Run Optimisation** to find the ideal crop layout."
)

# ── guard: no image ───────────────────────────────────────────────────────────
if not uploaded_image:
    st.info("⬅  Upload a farm image in the sidebar to get started.")
    st.stop()

# ── save upload to a temp file so OpenCV can read it ─────────────────────────
suffix = os.path.splitext(uploaded_image.name)[-1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded_image.getvalue())
    tmp_path = tmp.name

# ── STEP 1: segmentation ──────────────────────────────────────────────────────
with st.spinner("Segmenting image…"):
    cells = segment_image(tmp_path, n=grid_n)

st.markdown('<div class="section-header">Step 1 — Image Segmentation</div>', unsafe_allow_html=True)
col_img, col_info = st.columns([2, 1])
with col_img:
    st.image(Image.open(uploaded_image), caption=f"Input ({grid_n}×{grid_n} grid)", use_container_width=True)
with col_info:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Grid cells</div>
        <div class="value">{grid_n * grid_n}</div>
    </div>
    <div class="metric-card">
        <div class="label">Cell resolution</div>
        <div class="value">~{Image.open(uploaded_image).size[0]//grid_n} × {Image.open(uploaded_image).size[1]//grid_n} px</div>
    </div>
    """, unsafe_allow_html=True)

# ── STEP 2: feature extraction ────────────────────────────────────────────────
with st.spinner("Extracting soil quality, moisture & NDVI…"):
    soil_quality, moisture_level = get_features(cells)
    ndvi_arr = get_ndvi_proxy(cells)

st.markdown('<div class="section-header">Step 2 — Feature Maps</div>', unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.pyplot(draw_feature_map(soil_quality, "Soil Quality", "YlOrBr"))
with fc2:
    st.pyplot(draw_feature_map(moisture_level, "Moisture Level", "Blues"))
with fc3:
    st.pyplot(draw_feature_map(ndvi_arr, "NDVI Proxy", "YlGn"))

# ── STEP 3: zone detection ────────────────────────────────────────────────────
with st.spinner("Running K-Means zone detection…"):
    zone_labels = detect_zones(soil_quality, moisture_level, ndvi_arr)

st.markdown('<div class="section-header">Step 3 — Agronomic Zones</div>', unsafe_allow_html=True)
zc1, zc2 = st.columns([1, 2])
with zc1:
    st.pyplot(draw_zone_grid(zone_labels))
with zc2:
    num_zones = zone_labels.max() + 1
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Zones detected</div>
        <div class="value">{num_zones}</div>
    </div>
    <div class="metric-card">
        <div class="label">Mean soil quality</div>
        <div class="value">{soil_quality.mean():.3f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Mean moisture</div>
        <div class="value">{moisture_level.mean():.3f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Mean NDVI proxy</div>
        <div class="value">{ndvi_arr.mean():.3f}</div>
    </div>
    """, unsafe_allow_html=True)

# ── STEP 4: genetic algorithm ─────────────────────────────────────────────────
if not run_btn:
    st.info("Click **▶ Run Optimisation** in the sidebar to start the genetic algorithm.")
    st.stop()

st.markdown('<div class="section-header">Step 4 — NSGA-II Genetic Algorithm</div>', unsafe_allow_html=True)

progress_bar  = st.progress(0, text="Initialising population…")
status_text   = st.empty()
gen_log_area  = st.empty()

live_log: list[dict] = []

def progress_callback(gen, n_gen, best_fit, best_ind):
    pct = int((gen + 1) / n_gen * 100)
    progress_bar.progress(pct, text=f"Generation {gen+1}/{n_gen}  ·  Best fitness: {best_fit:,.0f}")
    status_text.markdown(
        f"<span style='font-size:12px;color:#5a8a6a;'>gen {gen+1:>4} · "
        f"best profit: <span style='color:#9de87a'>{best_fit:>14,.0f} PKR</span></span>",
        unsafe_allow_html=True,
    )


with st.spinner("Running optimisation — this may take a moment…"):
    t0 = time.time()
    best_grid, log, pareto_front = run_ga(
        soil_quality=soil_quality,
        moisture_level=moisture_level,
        ndvi=ndvi_arr,
        cx_prob=cx_prob,
        mut_prob=mut_prob,
        n=grid_n,
        n_gen=gen_size,
        pop_size=pop_size,
        progress_callback=progress_callback,
    )
    elapsed = time.time() - t0

progress_bar.progress(100, text="Optimisation complete ✓")
status_text.empty()

# ── STEP 5: results ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 5 — Results</div>', unsafe_allow_html=True)

total_profit = estimate_total_profit(best_grid)
water_need   = estimate_water_need(best_grid, lat, lon)
dist         = compute_crop_distribution(best_grid)

# summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"""<div class="metric-card"><div class="label">Est. Total Profit</div>
<div class="value">PKR {total_profit:,.0f}</div></div>""", unsafe_allow_html=True)
m2.markdown(f"""<div class="metric-card"><div class="label">Net Water Need</div>
<div class="value">{water_need:,.0f} mm</div></div>""", unsafe_allow_html=True)
m3.markdown(f"""<div class="metric-card"><div class="label">Run Time</div>
<div class="value">{elapsed:.1f} s</div></div>""", unsafe_allow_html=True)
m4.markdown(f"""<div class="metric-card"><div class="label">Generations</div>
<div class="value">{gen_size}</div></div>""", unsafe_allow_html=True)

# crop grid + distribution
rc1, rc2 = st.columns([3, 2])
with rc1:
    st.pyplot(draw_crop_grid(best_grid))
with rc2:
    st.markdown("**Crop Distribution**")
    for name, info in sorted(dist.items(), key=lambda x: -x[1]["count"]):
        if info["count"] == 0:
            continue
        bar_w = info["pct"]
        st.markdown(
            f"""<div style="margin:5px 0;">
                <span style="font-size:12px;color:#8ab89a;">{name}</span>
                <div style="background:#1a2e20;border-radius:3px;height:14px;margin-top:3px;">
                    <div style="background:{CROP_COLORS[name]};width:{bar_w}%;height:14px;border-radius:3px;"></div>
                </div>
                <span style="font-size:11px;color:#5a8a6a;">{info['count']} cells · {info['pct']}%</span>
            </div>""",
            unsafe_allow_html=True,
        )

# fitness convergence plot
st.markdown("**GA Convergence**")
st.pyplot(draw_fitness_log(log))

# generation log table
with st.expander("📋 Generation Log", expanded=False):
    header_cols = st.columns([1, 2, 2])
    header_cols[0].markdown("**Gen**")
    header_cols[1].markdown("**Best Fitness**")
    header_cols[2].markdown("**Avg Fitness**")
    for entry in log[::max(1, len(log) // 50)]:   # show at most ~50 rows
        row = st.columns([1, 2, 2])
        row[0].markdown(f"`{entry['gen']}`")
        row[1].markdown(f"`{entry['best_fit']:,.0f}`")
        row[2].markdown(f"`{entry['average_fit']:,.0f}`")

# raw grid download
csv_lines = [",".join(CROP_NAMES[best_grid[i, j]] for j in range(grid_n)) for i in range(grid_n)]
st.download_button(
    label="⬇  Download crop map (CSV)",
    data="\n".join(csv_lines),
    file_name="sowsmart_crop_map.csv",
    mime="text/csv",
)

# cleanup temp file
os.unlink(tmp_path)