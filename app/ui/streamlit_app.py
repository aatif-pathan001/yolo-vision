"""
YOLO Vision — Streamlit UI
Professional real-time object detection interface.
"""

import io
import cv2
import time
import tempfile
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

from app.core.detector import YOLODetector
from app.utils.helpers import (
    bgr_to_pil, ensure_output_dirs, build_output_path,
    summarize_detections, frame_to_jpeg_bytes, is_video,
)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YOLO Vision",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Dark Terminal / Cyberpunk Aesthetic ─────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background: #080c10;
    color: #c8d8e8;
}

/* ── Main container ── */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #00d4ff 0%, #00ff9d 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 4px;
    color: #00d4ff88;
    text-transform: uppercase;
    margin: 0.5rem 0 0;
}
.hero-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff44, #00ff9d44, transparent);
    margin: 1.5rem auto;
    max-width: 600px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1c2a3a;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}
.sidebar-section {
    border: 1px solid #1c2a3a;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: #0a0f14;
}
.sidebar-label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00d4ff99;
    margin-bottom: 0.5rem;
}

/* ── Cards ── */
.detect-card {
    background: #0d1117;
    border: 1px solid #1c2a3a;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.detect-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00d4ff, #00ff9d);
}

/* ── Detection Tags ── */
.det-tag {
    display: inline-block;
    background: #0d1117;
    border: 1px solid #00d4ff44;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #00d4ff;
    margin: 2px 3px;
    font-family: 'Space Mono', monospace;
}
.det-tag-conf {
    color: #00ff9d;
    border-color: #00ff9d44;
}

/* ── Metric Boxes ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 100px;
    background: #0a0f14;
    border: 1px solid #1c2a3a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #00d4ff;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a7a9a;
    margin-top: 4px;
}

/* ── Progress / Status ── */
.status-ok {
    color: #00ff9d;
    font-size: 0.75rem;
}
.status-warn {
    color: #ff6b35;
    font-size: 0.75rem;
}

/* ── Streamlit widget overrides ── */
.stButton button {
    background: transparent;
    border: 1px solid #00d4ff;
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.5rem 1.5rem;
    border-radius: 4px;
    transition: all 0.2s;
    width: 100%;
}
.stButton button:hover {
    background: #00d4ff12;
    border-color: #00ff9d;
    color: #00ff9d;
}
.stSlider [data-testid="stThumbValue"] {
    color: #00d4ff;
}
div[data-testid="stFileUploader"] {
    border: 1px dashed #1c2a3a;
    border-radius: 8px;
    background: #0a0f14;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #00d4ff44;
}
[data-testid="stSelectbox"] select {
    background: #0d1117;
    color: #c8d8e8;
    border-color: #1c2a3a;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #1c2a3a;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a7a9a;
    padding: 0.6rem 1.5rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
    background: transparent !important;
}
.stSuccess, .stInfo, .stWarning, .stError {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

/* ── Crop Gallery ── */
.crop-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px;
    margin-top: 1rem;
}
.crop-cell {
    border: 1px solid #1c2a3a;
    border-radius: 6px;
    overflow: hidden;
    background: #0a0f14;
    text-align: center;
    padding: 4px;
}
.crop-label {
    font-size: 0.6rem;
    color: #5a7a9a;
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: #1c2a3a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────

def init_session():
    """Initialize all session state keys."""
    defaults = {
        "detector": None,
        "model_loaded": False,
        "model_path": "model/clothdetector_v2.pt",
        "webcam_running": False,
        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()
OUTPUT_DIRS = ensure_output_dirs("outputs")

# ─── Helper: Load / Re-use Detector ──────────────────────────────────────────

def get_detector():
    """Return the cached detector or None if not loaded."""
    return st.session_state.get("detector")


def load_detector(model_path: str, conf: float, iou: float) -> bool:
    """Load a new detector and cache it in session state."""
    try:
        d = YOLODetector(model_path=model_path, confidence_threshold=conf, iou_threshold=iou)
        d.load_model()
        st.session_state["detector"] = d
        st.session_state["model_loaded"] = True
        st.session_state["model_path"] = model_path
        return True
    except Exception as e:
        st.error(f"⛔ {e}")
        return False

# ─── Hero Header ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <p class="hero-sub">// ultralytics · powered · real-time · detection</p>
    <h1 class="hero-title">YOLO VISION</h1>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-label">⬡ Model Configuration</p>', unsafe_allow_html=True)
    with st.container():
        model_path_input = st.text_input(
            "Model Path (.pt)",
            value=st.session_state.get("model_path") or "yolov8n.pt",
            placeholder="e.g. models/yolov8n.pt",
            help="Absolute or relative path to your YOLO .pt weights file.",
        )
        col1, col2 = st.columns(2)
        conf_val = col1.slider("Confidence", 0.05, 1.0, st.session_state["conf_threshold"], 0.05, format="%.2f")
        iou_val  = col2.slider("IOU", 0.05, 1.0, st.session_state["iou_threshold"], 0.05, format="%.2f")

        st.session_state["conf_threshold"] = conf_val
        st.session_state["iou_threshold"] = iou_val

        if st.button("⬢  LOAD MODEL"):
            with st.spinner("Loading weights…"):
                load_detector(model_path_input, conf_val, iou_val)

    # Model status badge
    d = get_detector()
    if d and d.is_loaded:
        st.markdown(f'<p class="status-ok">✔ Model active · {Path(d.model_path).name} · {len(d.class_names)} classes</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-warn">✘ No model loaded</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-label">⬡ About</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#5a7a9a; line-height:1.8;">
    YOLO Vision gives you three modes:<br>
    <span style="color:#00d4ff">▸ Image</span> — detect & crop objects<br>
    <span style="color:#00ff9d">▸ Video</span> — annotate entire clips<br>
    <span style="color:#ff6b35">▸ Webcam</span> — live stream detection<br><br>
    Built with Ultralytics · OpenCV · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─── Gate: require model ──────────────────────────────────────────────────────

detector = get_detector()
if not (detector and detector.is_loaded):
    st.markdown("""
    <div class="detect-card" style="text-align:center; padding: 3rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🎯</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#c8d8e8;">
            Load a model to begin
        </div>
        <div style="font-size:0.75rem; color:#5a7a9a; margin-top:0.5rem;">
            Enter a model path in the sidebar and click LOAD MODEL
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Main Tabs ────────────────────────────────────────────────────────────────

tab_image, tab_video, tab_webcam = st.tabs(["  🖼  IMAGE DETECT  ", "  🎬  VIDEO DETECT  ", "  📡  WEBCAM STREAM  "])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════

with tab_image:
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="detect-card">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-label">Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop an image file here",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            img_bytes = uploaded_file.read()
            pil_img = Image.open(io.BytesIO(img_bytes))
            st.image(pil_img, caption=f"Input · {uploaded_file.name}", use_container_width=True)

            # Save to temp
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            run_detect = st.button("⬢  RUN DETECTION")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="detect-card">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-label">Detection Result</p>', unsafe_allow_html=True)

        if uploaded_file and 'run_detect' in dir() and run_detect:
            with st.spinner("Inferencing…"):
                t0 = time.perf_counter()
                annotated, detections = detector.detect_image(tmp_path)
                elapsed_ms = (time.perf_counter() - t0) * 1000

            # Save annotated
            out_img_path = build_output_path(uploaded_file.name, str(OUTPUT_DIRS["images"]))
            cv2.imwrite(out_img_path, annotated)

            # Save crops
            crop_paths = detector.save_crops(detections, str(OUTPUT_DIRS["crops"]))

            # Show annotated image
            ann_pil = bgr_to_pil(annotated)
            st.image(ann_pil, caption="Annotated Output", use_container_width=True)

            # Metrics
            summary = summarize_detections(detections)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{summary['total_objects']}</div>
                    <div class="metric-lbl">Objects</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{len(summary['class_counts'])}</div>
                    <div class="metric-lbl">Classes</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{summary['avg_confidence']:.0%}</div>
                    <div class="metric-lbl">Avg Conf</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{elapsed_ms:.0f}<span style="font-size:1rem">ms</span></div>
                    <div class="metric-lbl">Inference</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detection tags
            if detections:
                tags_html = "".join(
                    f'<span class="det-tag">{d.class_label}</span>'
                    f'<span class="det-tag det-tag-conf">{d.confidence:.0%}</span>'
                    for d in detections
                )
                st.markdown(tags_html, unsafe_allow_html=True)

            # Download button
            with open(out_img_path, "rb") as f:
                st.download_button(
                    "⬇  Download Annotated Image",
                    f,
                    file_name=Path(out_img_path).name,
                    mime="image/jpeg",
                )

        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#2a3a4a;">
                <div style="font-size:2.5rem">🎯</div>
                <div style="font-size:0.75rem; margin-top:0.5rem; letter-spacing:2px; text-transform:uppercase;">
                    Awaiting input
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Crop Gallery ─────────────────────────────────────────────────────────
    if uploaded_file and 'detections' in dir() and detections and crop_paths:
        st.markdown("---")
        st.markdown('<p class="sidebar-label">⬡ Cropped Detections</p>', unsafe_allow_html=True)

        cols_per_row = 6
        rows = [detections[i:i+cols_per_row] for i in range(0, len(detections), cols_per_row)]
        crop_iter = iter(crop_paths)

        for row_dets in rows:
            row_cols = st.columns(len(row_dets))
            for col, det in zip(row_cols, row_dets):
                try:
                    crop_path = next(crop_iter)
                    crop_img = Image.open(crop_path)
                    with col:
                        st.image(crop_img, use_container_width=True)
                        st.markdown(
                            f'<div style="text-align:center;font-size:0.62rem;color:#00d4ff">{det.class_label}</div>'
                            f'<div style="text-align:center;font-size:0.58rem;color:#5a7a9a">{det.confidence:.0%}</div>',
                            unsafe_allow_html=True
                        )
                except StopIteration:
                    break

    # ── Full Detection Table ──────────────────────────────────────────────────
    if uploaded_file and 'detections' in dir() and detections:
        with st.expander("▸ Full Detection Report", expanded=False):
            import pandas as pd
            rows_data = []
            for i, d in enumerate(detections):
                x1, y1, x2, y2 = d.bbox
                rows_data.append({
                    "#": i + 1,
                    "Class": d.class_label,
                    "Confidence": f"{d.confidence:.4f}",
                    "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                    "W": d.bbox_width, "H": d.bbox_height,
                })
            st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO DETECTION
# ═══════════════════════════════════════════════════════════════════════

with tab_video:
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="detect-card">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Upload Video</p>', unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Drop a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )

    if video_file:
        # Preview original
        st.video(video_file)
        st.markdown(f'<p class="status-ok">✔ Loaded: {video_file.name} · {video_file.size // 1024} KB</p>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 3])
        process_btn = col_a.button("⬢  PROCESS VIDEO")

        if process_btn:
            # Write to temp
            suffix = Path(video_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            out_path = build_output_path(video_file.name, str(OUTPUT_DIRS["videos"]), "_detected")

            progress_bar = st.progress(0, text="Processing frames…")
            status_text = st.empty()

            def update_progress(current, total):
                pct = min(int(current / max(total, 1) * 100), 100)
                progress_bar.progress(pct, text=f"Frame {current} / {total}")

            with st.spinner("Running detection on video…"):
                t0 = time.perf_counter()
                stats = detector.detect_video(tmp_path, out_path, progress_callback=update_progress)
                elapsed = time.perf_counter() - t0

            progress_bar.progress(100, text="Complete!")
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{stats['frames_processed']}</div>
                    <div class="metric-lbl">Frames</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{stats['total_detections']}</div>
                    <div class="metric-lbl">Detections</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{stats['avg_detections_per_frame']}</div>
                    <div class="metric-lbl">Avg/Frame</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{elapsed:.1f}<span style="font-size:1rem">s</span></div>
                    <div class="metric-lbl">Total Time</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show output video
            st.markdown('<p class="sidebar-label" style="margin-top:1rem">Annotated Output</p>', unsafe_allow_html=True)
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)

            st.download_button(
                "⬇  Download Annotated Video",
                video_bytes,
                file_name=Path(out_path).name,
                mime="video/mp4",
            )

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — WEBCAM STREAM
# ═══════════════════════════════════════════════════════════════════════

with tab_webcam:
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="detect-card">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Live Webcam Detection</p>', unsafe_allow_html=True)

    cam_col1, cam_col2 = st.columns([1, 2])
    with cam_col1:
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)
        start_cam = st.button("⬢  START CAMERA")
        stop_cam  = st.button("◼  STOP CAMERA")

    with cam_col2:
        st.markdown("""
        <div style="font-size:0.72rem; color:#5a7a9a; line-height:2;">
        <span style="color:#00d4ff">▸</span> Select camera index (0 = default webcam)<br>
        <span style="color:#00d4ff">▸</span> Click START to begin live detection<br>
        <span style="color:#00d4ff">▸</span> Click STOP to end the stream<br>
        <span style="color:#ff6b35">⚠</span> Webcam must be connected and accessible
        </div>
        """, unsafe_allow_html=True)

    if start_cam:
        st.session_state["webcam_running"] = True
    if stop_cam:
        st.session_state["webcam_running"] = False

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state["webcam_running"]:
        st.markdown('<p class="status-ok" style="margin:0.5rem 0">● LIVE  —  Real-time YOLO detection active</p>', unsafe_allow_html=True)

        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        stop_signal = st.button("◼  STOP STREAM")

        if stop_signal:
            st.session_state["webcam_running"] = False
            st.rerun()

        try:
            for annotated_frame, detections in detector.stream_webcam(camera_index):
                if not st.session_state["webcam_running"]:
                    break

                # Convert to RGB for Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Live detection info
                summary = summarize_detections(detections)
                classes_str = " · ".join(
                    f"{cls} ({cnt})" for cls, cnt in summary["class_counts"].items()
                ) or "none"
                info_placeholder.markdown(
                    f'<p class="status-ok">Objects: {summary["total_objects"]}  '
                    f'| Avg Conf: {summary["avg_confidence"]:.0%}  '
                    f'| Classes: {classes_str}</p>',
                    unsafe_allow_html=True,
                )

        except RuntimeError as e:
            st.error(f"Camera error: {e}")
            st.session_state["webcam_running"] = False
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem; color:#2a3a4a; border: 1px solid #1c2a3a; border-radius:10px; margin-top:1rem;">
            <div style="font-size:3rem">📡</div>
            <div style="font-size:0.75rem; margin-top:0.5rem; letter-spacing:2px; text-transform:uppercase;">
                Camera offline
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-top:1rem; border-top:1px solid #1c2a3a;">
    <span style="font-size:0.65rem; letter-spacing:3px; color:#2a3a4a; text-transform:uppercase;">
        YOLO Vision · Powered by Ultralytics · Built with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
