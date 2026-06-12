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

# ── Design tokens ─────────────────────────────────────────────────────────────
# Palette: warm off-white surface, deep charcoal text, sage accent, amber highlight
# Rationale: high contrast on both light backgrounds and chart axes;
#            sage/amber evoke agriculture without the "terminal green" cliché.
PALETTE = {
    "bg":          "#F7F5F0",   # warm parchment — main surface
    "surface":     "#EDEAE3",   # slightly darker card background
    "border":      "#D4CFC5",   # subtle divider
    "text":        "#1A1916",   # near-black body
    "muted":       "#6B6760",   # secondary labels
    "accent":      "#4A7C59",   # sage green — primary action / highlight
    "accent_lt":   "#C8DDD0",   # light sage — progress fills
    "amber":       "#C47D2E",   # harvest amber — secondary accent
    "amber_lt":    "#F5E6CC",   # light amber — chip backgrounds
    "chart_bg":    "#EDEAE3",   # chart face colour
    "chart_grid":  "#D4CFC5",   # chart gridlines
}

CROP_COLORS = {
    "Wheat":     "#C9A84C",
    "Cotton":    "#A8B8A0",
    "Rice":      "#5B8FA8",
    "Sugarcane": "#4A7C59",
    "Maize":     "#C47D2E",
    "Sunflower": "#D4B44A",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
    color: {PALETTE['text']};
}}

/* ── App background ── */
.stApp {{
    background: {PALETTE['bg']};
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {PALETTE['surface']} !important;
    border-right: 1px solid {PALETTE['border']};
}}
section[data-testid="stSidebar"] * {{
    color: {PALETTE['text']} !important;
}}

/* ── Headings ── */
h1, h2, h3 {{
    font-family: 'Lora', Georgia, serif !important;
    color: {PALETTE['text']} !important;
    letter-spacing: -0.01em;
}}

/* ── Metric cards ── */
.ss-card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 16px 20px;
    margin: 6px 0;
}}
.ss-card .ss-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['muted']};
    margin-bottom: 4px;
}}
.ss-card .ss-value {{
    font-size: 20px;
    font-weight: 600;
    color: {PALETTE['accent']};
    font-family: 'Lora', serif;
}}

/* ── Section eyebrow ── */
.ss-eyebrow {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {PALETTE['muted']};
    border-bottom: 1px solid {PALETTE['border']};
    padding-bottom: 6px;
    margin: 28px 0 14px 0;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {PALETTE['accent']} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: opacity 0.15s ease !important;
}}
.stButton > button:hover {{
    opacity: 0.88 !important;
}}

/* ── Sliders ── */
.stSlider > div > div > div {{ accent-color: {PALETTE['accent']}; }}
.stSlider label {{ color: {PALETTE['text']} !important; font-size: 13px !important; }}

/* ── Expander ── */
div[data-testid="stExpander"] {{
    border: 1px solid {PALETTE['border']} !important;
    border-radius: 6px !important;
    background: {PALETTE['surface']} !important;
}}

/* ── Progress bar ── */
.stProgress > div > div > div {{
    background: {PALETTE['accent']} !important;
}}

/* ── Info / success boxes ── */
div[data-testid="stAlert"] {{
    border-radius: 6px !important;
}}

/* ── Download button ── */
.stDownloadButton > button {{
    background: {PALETTE['surface']} !important;
    color: {PALETTE['accent']} !important;
    border: 1px solid {PALETTE['accent']} !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}}
.stDownloadButton > button:hover {{
    background: {PALETTE['accent_lt']} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Matplotlib theme ──────────────────────────────────────────────────────────
def _apply_chart_style(fig, ax_list=None):
    """Apply consistent chart styling."""
    fig.patch.set_facecolor(PALETTE["chart_bg"])
    for ax in (ax_list or fig.axes):
        ax.set_facecolor(PALETTE["chart_bg"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.xaxis.label.set_color(PALETTE["muted"])
        ax.yaxis.label.set_color(PALETTE["muted"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_from_bytes(data: bytes) -> Image.Image:
    """Open a PIL Image from raw bytes (avoids file-pointer issues)."""
    return Image.open(io.BytesIO(data))


def draw_crop_grid(grid: np.ndarray) -> plt.Figure:
    n = grid.shape[0]
    color_grid = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            name = CROP_NAMES[grid[i, j]]
            color_grid[i, j] = mcolors.to_rgb(CROP_COLORS.get(name, "#999999"))

    fig, ax = plt.subplots(figsize=(6, 6))
    _apply_chart_style(fig)
    ax.imshow(color_grid, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(1, n + 1), fontsize=6, color=PALETTE["muted"])
    ax.set_yticklabels(range(1, n + 1), fontsize=6, color=PALETTE["muted"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    patches = [mpatches.Patch(color=CROP_COLORS[c], label=c) for c in CROP_NAMES]
    ax.legend(
        handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
        framealpha=0, labelcolor=PALETTE["text"], fontsize=9,
    )
    ax.set_title("Optimal Crop Placement", color=PALETTE["text"],
                 fontsize=12, pad=10, fontfamily="serif", fontweight="bold")
    fig.tight_layout()
    return fig


def draw_zone_grid(zone_labels: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    _apply_chart_style(fig)
    ax.imshow(zone_labels, cmap="tab10", interpolation="nearest")
    ax.set_title("Agronomic Zones", color=PALETTE["text"],
                 fontsize=11, pad=8, fontfamily="serif")
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_feature_map(arr: np.ndarray, title: str, cmap: str = "YlGn") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    _apply_chart_style(fig)
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, color=PALETTE["text"], fontsize=11, pad=8, fontfamily="serif")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"], labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["muted"])
    cbar.outline.set_edgecolor(PALETTE["border"])
    fig.tight_layout()
    return fig


def draw_fitness_log(log: list) -> plt.Figure:
    gens = [e["gen"] for e in log]
    best = [e["best_fit"] for e in log]
    avg  = [e["average_fit"] for e in log]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    _apply_chart_style(fig)
    ax.plot(gens, best, color=PALETTE["accent"],  linewidth=2,   label="Best fitness")
    ax.plot(gens, avg,  color=PALETTE["amber"],   linewidth=1.4, linestyle="--", label="Avg fitness")
    ax.set_xlabel("Generation",      fontsize=9)
    ax.set_ylabel("Fitness (profit)", fontsize=9)
    ax.set_title("GA Convergence", color=PALETTE["text"],
                 fontsize=11, fontfamily="serif")
    ax.grid(axis="y", color=PALETTE["chart_grid"], linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=9, framealpha=0, labelcolor=PALETTE["text"])
    fig.tight_layout()
    return fig


def compute_crop_distribution(grid: np.ndarray) -> dict:
    flat  = grid.flatten()
    total = len(flat)
    return {
        name: {"count": int(np.sum(flat == idx)),
               "pct":   round(int(np.sum(flat == idx)) / total * 100, 1)}
        for idx, name in enumerate(CROP_NAMES)
    }


def estimate_total_profit(grid: np.ndarray) -> float:
    return sum(
        CROPS[CROP_NAMES[grid[i, j]]]["profit"]
        for i in range(grid.shape[0])
        for j in range(grid.shape[1])
    )


def estimate_water_need(grid: np.ndarray, lat=None, lon=None) -> float:
    n   = grid.shape[0]
    raw = sum(CROPS[CROP_NAMES[grid[i, j]]]["water"] for i in range(n) for j in range(n))
    rain = get_effective_rain(n * n, lat, lon)
    return max(0.0, raw - rain)


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="ss-card">
        <div class="ss-label">{label}</div>
        <div class="ss-value">{value}</div>
    </div>"""


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="ss-eyebrow">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🌱 SowSmart")
    st.caption("Precision crop optimisation")
    st.divider()

    st.markdown("**Location**")
    st.caption("Sync GPS for local rainfall data.")
    location = streamlit_geolocation()

    lat, lon = None, None
    if location and location.get("latitude") is not None:
        lat = location["latitude"]
        lon = location["longitude"]
        st.success(f"📍 {lat:.4f}, {lon:.4f}", icon="✅")
    else:
        st.info("GPS unavailable — rainfall offset disabled.")

    st.divider()
    st.markdown("**Grid & GA Parameters**")
    grid_n    = st.slider("Grid size (N × N)",   5,   20,  10,  1)
    pop_size  = st.slider("Population size",     50,  500, 150, 10)
    gen_size  = st.slider("Generations",         50,  500, 100, 10)
    cx_prob   = st.slider("Crossover prob.",     0.3, 1.0, 0.7, 0.05)
    mut_prob  = st.slider("Mutation prob.",      0.01, 0.5, 0.1, 0.01)
    penalty_w = st.slider("Soil penalty weight", 0.5, 5.0, 2.0, 0.5)

    st.divider()
    st.markdown("**Farm Image**")
    uploaded_image = st.file_uploader(
        "Upload satellite / drone image",
        type=["jpeg", "jpg", "png"],
        label_visibility="collapsed",
    )

    # Read bytes once — reuse everywhere to avoid file-pointer issues
    image_bytes: bytes | None = uploaded_image.getvalue() if uploaded_image else None

    if image_bytes:
        st.image(pil_from_bytes(image_bytes), caption="Uploaded image",
                 use_column_width=True)

    st.divider()
    run_btn = st.button("▶  Run Optimisation")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# SowSmart")
st.markdown(
    "Upload a farm image, configure the parameters in the sidebar, "
    "then run the optimiser to find the ideal crop layout for your field."
)
st.divider()

# ── Guard: no image ───────────────────────────────────────────────────────────
if not image_bytes:
    st.info("Upload a farm image in the sidebar to get started.", icon="⬅")
    st.stop()

# ── Save to temp file for OpenCV ──────────────────────────────────────────────
suffix = os.path.splitext(uploaded_image.name)[-1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(image_bytes)
    tmp_path = tmp.name

# ── STEP 1 — Segmentation ─────────────────────────────────────────────────────
with st.spinner("Segmenting image…"):
    cells = segment_image(tmp_path, n=grid_n)

eyebrow("Step 1 — Image Segmentation")

pil_img = pil_from_bytes(image_bytes)
img_w, img_h = pil_img.size

col_img, col_info = st.columns([2, 1])
with col_img:
    st.image(pil_img, caption=f"Input · {grid_n}×{grid_n} grid",
             use_column_width=True)
with col_info:
    st.markdown(
        metric_card("Grid cells", f"{grid_n * grid_n}") +
        metric_card("Cell resolution",
                    f"~{img_w // grid_n} × {img_h // grid_n} px"),
        unsafe_allow_html=True,
    )

# ── STEP 2 — Feature Extraction ───────────────────────────────────────────────
with st.spinner("Extracting soil quality, moisture & NDVI…"):
    soil_quality, moisture_level = get_features(cells)
    ndvi_arr = get_ndvi_proxy(cells)

eyebrow("Step 2 — Feature Maps")
fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.pyplot(draw_feature_map(soil_quality, "Soil Quality", "YlOrBr"))
with fc2:
    st.pyplot(draw_feature_map(moisture_level, "Moisture Level", "Blues"))
with fc3:
    st.pyplot(draw_feature_map(ndvi_arr, "NDVI Proxy", "YlGn"))

# ── STEP 3 — Zone Detection ───────────────────────────────────────────────────
with st.spinner("Running K-Means zone detection…"):
    zone_labels = detect_zones(soil_quality, moisture_level, ndvi_arr)

eyebrow("Step 3 — Agronomic Zones")
zc1, zc2 = st.columns([1, 2])
with zc1:
    st.pyplot(draw_zone_grid(zone_labels))
with zc2:
    num_zones = int(zone_labels.max()) + 1
    st.markdown(
        metric_card("Zones detected",   str(num_zones)) +
        metric_card("Mean soil quality", f"{soil_quality.mean():.3f}") +
        metric_card("Mean moisture",     f"{moisture_level.mean():.3f}") +
        metric_card("Mean NDVI proxy",   f"{ndvi_arr.mean():.3f}"),
        unsafe_allow_html=True,
    )

# ── STEP 4 — Genetic Algorithm ────────────────────────────────────────────────
if not run_btn:
    st.info("Click **▶ Run Optimisation** in the sidebar to start the genetic algorithm.", icon="ℹ️")
    st.stop()

eyebrow("Step 4 — NSGA-II Genetic Algorithm")

progress_bar = st.progress(0, text="Initialising population…")
status_text  = st.empty()


def progress_callback(gen: int, n_gen: int, best_fit: float, best_ind):
    pct = int((gen + 1) / n_gen * 100)
    progress_bar.progress(
        pct,
        text=f"Generation {gen + 1}/{n_gen}  ·  Best fitness: {best_fit:,.0f} PKR",
    )
    status_text.caption(f"Gen {gen + 1:>4}  ·  Best profit: {best_fit:>14,.0f} PKR")


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

# ── STEP 5 — Results ──────────────────────────────────────────────────────────
eyebrow("Step 5 — Results")

total_profit = estimate_total_profit(best_grid)
water_need   = estimate_water_need(best_grid, lat, lon)
dist         = compute_crop_distribution(best_grid)

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.markdown(metric_card("Est. Total Profit",  f"PKR {total_profit:,.0f}"), unsafe_allow_html=True)
m2.markdown(metric_card("Net Water Need",     f"{water_need:,.0f} mm"),    unsafe_allow_html=True)
m3.markdown(metric_card("Run Time",           f"{elapsed:.1f} s"),         unsafe_allow_html=True)
m4.markdown(metric_card("Generations",        str(gen_size)),              unsafe_allow_html=True)

st.write("")

# Crop grid + distribution
rc1, rc2 = st.columns([3, 2])
with rc1:
    st.pyplot(draw_crop_grid(best_grid))
with rc2:
    st.markdown("**Crop Distribution**")
    for name, info in sorted(dist.items(), key=lambda x: -x[1]["count"]):
        if info["count"] == 0:
            continue
        bar_w   = info["pct"]
        color   = CROP_COLORS.get(name, "#999")
        st.markdown(
            f"""<div style="margin:8px 0;">
                <div style="display:flex;justify-content:space-between;
                            font-size:12px;color:{PALETTE['text']};margin-bottom:4px;">
                    <span style="font-weight:600;">{name}</span>
                    <span style="color:{PALETTE['muted']};">{info['count']} cells · {info['pct']}%</span>
                </div>
                <div style="background:{PALETTE['border']};border-radius:4px;height:8px;">
                    <div style="background:{color};width:{bar_w}%;height:8px;border-radius:4px;"></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

st.write("")

# Convergence chart
st.markdown("**GA Convergence**")
st.pyplot(draw_fitness_log(log))

# Generation log (collapsed)
with st.expander("Generation Log", expanded=False):
    h1, h2, h3 = st.columns([1, 2, 2])
    h1.markdown("**Gen**"); h2.markdown("**Best Fitness**"); h3.markdown("**Avg Fitness**")
    for entry in log[::max(1, len(log) // 50)]:
        r1, r2, r3 = st.columns([1, 2, 2])
        r1.markdown(f"`{entry['gen']}`")
        r2.markdown(f"`{entry['best_fit']:,.0f}`")
        r3.markdown(f"`{entry['average_fit']:,.0f}`")

st.write("")

# Download
csv_lines = [
    ",".join(CROP_NAMES[best_grid[i, j]] for j in range(grid_n))
    for i in range(grid_n)
]
st.download_button(
    label="⬇  Download crop map (CSV)",
    data="\n".join(csv_lines),
    file_name="sowsmart_crop_map.csv",
    mime="text/csv",
)

# Cleanup
os.unlink(tmp_path)