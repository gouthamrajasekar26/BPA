"""
MASTER FORENSIC BPA DASHBOARD (STREAMLIT)
=========================================
Unified Single-Upload Bloodstain Pattern Analysis (BPA) Platform:
Upload a single crime scene image to automatically execute all 7 analysis modules
and visualize complete forensic outputs end-to-end.
"""

import os
import json
import random
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# MUST BE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="Forensic BPA Master Platform",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import Core Forensic Modules & Master Pipeline
import module1_dataset as m1
import module2_model as m2
import module3_droplet_cv as m3
import module4_origin_resolver as m4
import module5_weapon_heuristic as m5
import module6_spatial_render as m6
import module7_report_exporter as m7
import pipeline as pipe

# Forensic Laboratory — Espresso & Copper Theme
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Global background: #3A2618 ── */
    .stApp {
        background-color: #3A2618;
        color: #FBF6EF;
    }

    .stMarkdown, .stText, p, li, span, label {
        color: #E8DDD0;
    }

    /* ── Header card ── */
    .dexter-header-card {
        background: linear-gradient(135deg, #1E120A 0%, #2E1C10 50%, #3A2618 100%);
        border: 1px solid #A9744F;
        border-top: 5px solid #A9744F;
        border-radius: 10px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.65);
    }

    .dexter-title {
        color: #FBF6EF;
        font-size: 2.0rem;
        font-weight: 900;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
        text-shadow: 0 2px 12px rgba(169, 116, 79, 0.45);
    }

    .dexter-subtitle {
        color: #C49A72;
        font-size: 0.88rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.4px;
        opacity: 0.90;
    }

    .dexter-badge {
        background: rgba(169, 116, 79, 0.22);
        border: 1px solid #A9744F;
        color: #E8C9A8;
        padding: 3px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* ── Section headings ── */
    .dexter-heading {
        color: #A9744F;
        font-size: 1.10rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
        border-bottom: 2px solid #6B4530;
        padding-bottom: 6px;
        text-transform: uppercase;
    }

    /* ── Espresso glass card ── */
    .dexter-card {
        background: rgba(251, 246, 239, 0.04);
        border: 1px solid rgba(169, 116, 79, 0.28);
        border-top: 3px solid #A9744F;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.55);
        color: #FBF6EF;
        backdrop-filter: blur(10px);
    }

    /* ── Upload hero dropzone ── */
    .upload-hero-card {
        background: rgba(251, 246, 239, 0.02);
        border: 2px dashed #6B4530;
        border-radius: 12px;
        padding: 26px;
        margin-bottom: 24px;
        box-shadow: 0 4px 28px rgba(0, 0, 0, 0.50);
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: rgba(169, 116, 79, 0.10);
        border: 1px solid rgba(169, 116, 79, 0.28);
        border-top: 3px solid #A9744F;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    }

    div[data-testid="stMetric"] label {
        color: #C49A72 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FBF6EF !important;
        font-weight: 800 !important;
    }

    /* ── Primary action button ── */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6B4530 0%, #A9744F 100%) !important;
        color: #FBF6EF !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 7px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 18px rgba(169, 116, 79, 0.40) !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #52351F 0%, #8C5F3C 100%) !important;
        box-shadow: 0 6px 24px rgba(169, 116, 79, 0.55) !important;
        transform: translateY(-1px);
    }

    /* ── Standard buttons ── */
    div.stButton > button {
        background-color: #2A1B10 !important;
        color: #E8DDD0 !important;
        border: 1px solid #6B4530 !important;
        font-weight: 600 !important;
        border-radius: 7px !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        border-color: #A9744F !important;
        color: #FBF6EF !important;
        background-color: rgba(169, 116, 79, 0.15) !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #2A1B10;
        border-right: 1px solid #6B4530;
    }

    section[data-testid="stSidebar"] * {
        color: #E8DDD0;
    }

    /* ── Streamlit tab strip ── */
    div[data-testid="stTabs"] button[role="tab"] {
        color: #C49A72 !important;
        font-weight: 600;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #FBF6EF !important;
        border-bottom-color: #A9744F !important;
    }

    /* ── Dataframe / table ── */
    div[data-testid="stDataFrameGlideDataEditor"] {
        background: #2A1B10 !important;
        color: #FBF6EF !important;
    }

    /* ── Info / success / warning boxes ── */
    div[data-testid="stInfo"] {
        background: rgba(169, 116, 79, 0.12) !important;
        border-left-color: #A9744F !important;
        color: #E8DDD0 !important;
    }

    div[data-testid="stSuccess"] {
        background: rgba(30, 60, 40, 0.45) !important;
        color: #A8D5B5 !important;
    }

    /* ── Horizontal rule ── */
    hr {
        border-color: #6B4530;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state():
    """Initializes persistent application session variables."""
    if "case_num" not in st.session_state:
        st.session_state["case_num"] = f"CASE-2026-{random.randint(1000, 9999)}"
    if "investigator_id" not in st.session_state:
        st.session_state["investigator_id"] = "DEXTER-ANALYST-01"
    if "pipeline_results" not in st.session_state:
        st.session_state["pipeline_results"] = None
    if "tail_clip_factor" not in st.session_state:
        st.session_state["tail_clip_factor"] = 0.85


def render_header():
    """Renders Dexter-inspired clinical laboratory top header."""
    st.markdown(
        """
        <div class="dexter-header-card">
            <div class="dexter-title">🩸 FORENSIC BLOODSTAIN PATTERN ANALYSIS // CLINICAL LABORATORY</div>
            <div class="dexter-subtitle">Dexter Forensic Suite | Spatter Classifier (Mod 2) | Tail-Clipping CV Engine (Mod 3) | 2D/3D Origin Solver (Mod 4)</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    init_session_state()
    render_header()

    # Sidebar Controls & System Status
    with st.sidebar:
        st.markdown("<h3 style='color:#A9744F;'>⚙️ Case & Laboratory Controls</h3>", unsafe_allow_html=True)
        st.session_state["case_num"] = st.text_input("Case Reference ID", value=st.session_state["case_num"])
        st.session_state["investigator_id"] = st.text_input("Lead Analyst ID", value=st.session_state["investigator_id"])
        
        st.markdown("---")
        st.markdown("<h3 style='color:#A9744F;'>📐 Droplet Engine Parameters</h3>", unsafe_allow_html=True)
        tail_clip = st.slider(
            "Tail Clipping Reduction Factor",
            min_value=0.50,
            max_value=1.00,
            value=st.session_state["tail_clip_factor"],
            step=0.01,
            help="Default 0.85 (15% reduction) strips directional spine/tail noise from major axis before calculating theta = arcsin(w / h_clipped)."
        )
        st.session_state["tail_clip_factor"] = tail_clip
        
        st.markdown("---")
        st.markdown("<h3 style='color:#A9744F;'>🧪 Benchmark Dataset Actions</h3>", unsafe_allow_html=True)
        if st.button("📥 Download Attinger Dataset", use_container_width=True):
            with st.spinner("Downloading dataset from Figshare..."):
                if m1.download_and_extract_attinger_dataset():
                    st.success("Downloaded Figshare dataset!")
                else:
                    m1.generate_synthetic_dataset(num_samples_per_class=25)
                    st.warning("Generated synthetic dataset fallback!")
                m1.sort_and_stratify_dataset()

        if st.button("⚡ Generate Synthetic Dataset", use_container_width=True):
            with st.spinner("Generating synthetic spatter benchmark images..."):
                m1.generate_synthetic_dataset(num_samples_per_class=25)
                m1.sort_and_stratify_dataset()
                st.success("Generated 100 synthetic benchmark images!")

        st.markdown("---")
        st.caption("Active Forensic Modules:")
        st.markdown("- ✅ Module 1: Dataset & Environment Setup")
        st.markdown("- ✅ Module 2: ResNet-50 Spatter Classifier")
        st.markdown("- ✅ Module 3: OpenCV Droplet & Tail Clip Engine")
        st.markdown("- ✅ Module 4: 2D Ray Convergence & 3D Origin")

    # =========================================================================
    # HERO SECTION: UNIFIED SINGLE IMAGE UPLOAD & PIPELINE LAUNCHER
    # =========================================================================
    st.markdown('<div class="upload-hero-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#A9744F; margin-top:0;'>📷 Crime Scene Bloodstain Canvas</h3>", unsafe_allow_html=True)
    st.caption("Upload a crime scene bloodstain image below to execute Phase 1 forensic analysis (Modules 1 - 4) end-to-end.")
    
    col_up, col_preview = st.columns([1.2, 1], gap="large")

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Crime Scene Image (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="master_crime_image_upload"
        )
        use_sample = st.checkbox("🧪 Use Sample Crime Scene Bloodstain Image", value=(uploaded_file is None), key="master_use_sample")

        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            img_source_desc = f"Uploaded File: {uploaded_file.name}"
        elif use_sample:
            # Generate sample image from CV synthetic generator
            _, canvas = m3.generate_synthetic_droplet_samples(count=16, tail_clip_factor=st.session_state["tail_clip_factor"])
            image_to_process = Image.fromarray(canvas)
            img_source_desc = "Synthetic Crime Scene Spatter Sample Canvas"
        else:
            image_to_process = None
            img_source_desc = "No image selected"

        run_btn = st.button("🚀 Execute Phase 1 Forensic Pipeline (Modules 1 - 4)", use_container_width=True, type="primary")

    with col_preview:
        if image_to_process is not None:
            st.image(image_to_process, caption=img_source_desc, use_container_width=True)
        else:
            st.info("Please upload an image or check 'Use Sample Crime Scene Pattern Image' to begin.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Automatic execution trigger when user uploads an image or clicks the run button
    if (run_btn or uploaded_file is not None or use_sample) and image_to_process is not None:
        if run_btn or st.session_state["pipeline_results"] is None:
            with st.spinner("Executing Phase 1 Forensic Pipeline (Modules 1 ➔ 2 ➔ 3 ➔ 4)..."):
                pipeline_output = pipe.run_full_forensic_pipeline(
                    image_input=image_to_process,
                    case_number=st.session_state["case_num"],
                    investigator_id=st.session_state["investigator_id"],
                    tail_clip_factor=st.session_state["tail_clip_factor"]
                )
                st.session_state["pipeline_results"] = pipeline_output
                st.success(f"Phase 1 Forensic Pipeline completed successfully for Case {st.session_state['case_num']}!")

    # =========================================================================
    # DISPLAY GENERATED OUTPUTS FROM PHASE 1 MODULES (1 - 4)
    # =========================================================================
    if st.session_state["pipeline_results"] is not None:
        res = st.session_state["pipeline_results"]
        spatter = res["spatter_classification"]
        drops_summary = res["droplet_summary"]
        drops_data = res["droplets_data"]
        annotated_img = res["annotated_rgb"]
        origin_3d = res["origin_3d_result"]

        st.markdown("---")
        st.markdown("<h2 style='color:#A9744F;'>📊 Laboratory Forensic Reports & Analysis Outputs</h2>", unsafe_allow_html=True)

        tab_exec, tab_mod2, tab_mod3, tab_mod4 = st.tabs([
            "📄 Executive Summary",
            "🧠 Spatter & Velocity Classifier (Mod 2)",
            "🔍 Droplets & Impact Angles (Mod 3)",
            "🎯 2D/3D Point of Origin (Mod 4)"
        ])

        # ---------------------------------------------------------------------
        # TAB 1: EXECUTIVE PHASE 1 SUMMARY
        # ---------------------------------------------------------------------
        with tab_exec:
            st.markdown('<div class="dexter-heading">📄 Forensic Case Executive Summary</div>', unsafe_allow_html=True)
            st.caption(f"Case ID: {res['case_number']} | Lead Analyst: {res['investigator_id']} | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Spatter Type", spatter["predicted_class"])
            with m_col2:
                st.metric("Impact Velocity", spatter["impact_velocity"]["range_ms"])
            with m_col3:
                x_o, y_o, z_o = origin_3d["point_of_origin_3d"]
                st.metric("3D Origin [X, Y, Z]", f"[{x_o:.1f}, {y_o:.1f}, {z_o:.1f}] cm")
            with m_col4:
                st.metric("Margin of Error", f"±{origin_3d['margin_of_error_cm']:.2f} cm")

            st.markdown('<div class="dexter-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color:#A9744F;'>📝 Physical Analysis & Trajectory Findings</h4>", unsafe_allow_html=True)
            st.write(f"- **Primary Spatter Category:** {spatter['predicted_class']} (Confidence: {spatter['confidence']*100:.1f}%)")
            st.write(f"- **Physical Impact Mechanism:** {spatter['impact_velocity']['physical_mechanism']}")
            st.write(f"- **Droplet Trigonometry Analysis:** Extracted {drops_summary['total_droplets']} droplets with mean width {drops_summary['mean_width_mm']}mm and mean impact angle {drops_summary['mean_impact_angle_deg']}° (applied {st.session_state['tail_clip_factor']*100:.0f}% tail clipping).")
            st.write(f"- **Spatial Origin:** Solved 2D area of convergence at (X={x_o:.1f} cm, Y={y_o:.1f} cm) with outward 3D origin height Z={z_o:.1f} cm.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # TAB 2: DEEP LEARNING SPATTER CLASSIFIER (MODULE 2)
        # ---------------------------------------------------------------------
        with tab_mod2:
            st.markdown('<div class="dexter-heading">🧠 Module 2: PyTorch ResNet-50 Spatter Classifier</div>', unsafe_allow_html=True)
            
            c_m2_left, c_m2_right = st.columns([1, 1.2], gap="large")
            with c_m2_left:
                st.image(image_to_process, caption="Input Crime Scene Pattern Canvas", use_container_width=True)
            
            with c_m2_right:
                st.markdown('<div class="dexter-card">', unsafe_allow_html=True)
                st.markdown(f"#### Predicted Class: <span style='color:#A9744F;'>{spatter['predicted_class']}</span>", unsafe_allow_html=True)
                st.metric("Model Classification Confidence", f"{spatter['confidence']*100:.2f}%")
                
                v = spatter["impact_velocity"]
                st.markdown(f"**Impact Velocity Category:** {v['category']}")
                st.markdown(f"**Velocity Range:** {v['range_ms']}")
                st.markdown(f"**Droplet Size Profile:** {v['droplet_size_range']}")
                st.info(f"**Physical Impact Mechanism:** {v['physical_mechanism']}")

                st.markdown("##### Categorical Probability Distribution:")
                for cat_name, prob in spatter["class_probabilities"].items():
                    lbl_c, bar_c, pct_c = st.columns([2.5, 4, 1])
                    with lbl_c:
                        st.caption(f"**{cat_name}**")
                    with bar_c:
                        st.progress(float(prob))
                    with pct_c:
                        st.caption(f"{prob*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # TAB 3: DROPLET EXTRACTION & TAIL CLIPPING (MODULE 3)
        # ---------------------------------------------------------------------
        with tab_mod3:
            st.markdown('<div class="dexter-heading">🔍 Module 3: OpenCV Droplet Extraction & Tail-Clipping Engine</div>', unsafe_allow_html=True)
            
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.image(image_to_process, caption="Raw Input Image Canvas", use_container_width=True)
            with v_col2:
                st.image(annotated_img, caption=f"Segmented Droplets & Flight Vectors (Extracted: {len(drops_data)})", use_container_width=True)

            if drops_data:
                st.markdown("#### 📋 Extracted Droplet Metrics Data Table")
                df_drops = pd.DataFrame(drops_data)
                disp_cols = df_drops[[
                    "id", "centroid", "width_mm", "length_raw_mm", "length_clipped_mm",
                    "impact_angle_deg", "orientation_deg", "aspect_ratio", "tail_clip_applied"
                ]].copy()
                disp_cols.columns = [
                    "ID", "Centroid (X, Y)", "Width w (mm)", "Raw Length h_raw (mm)",
                    "Clipped Length h_clip (mm)", "Impact Angle θ (°)", "Orientation φ (°)", "Aspect Ratio", "Tail Clipped (mm)"
                ]
                st.dataframe(disp_cols, use_container_width=True, height=260)

                hist_col1, hist_col2 = st.columns(2)
                with hist_col1:
                    import plotly.express as px
                    fig_ang = px.histogram(
                        df_drops, x="impact_angle_deg", nbins=15,
                        title="<b>Impact Angle θ Distribution</b>",
                        labels={"impact_angle_deg": "Impact Angle θ (°)"},
                        color_discrete_sequence=['#A9744F']
                    )
                    fig_ang.update_layout(template="plotly_dark", paper_bgcolor="#3A2618", plot_bgcolor="#2A1B10", font=dict(color="#E8DDD0"))
                    st.plotly_chart(fig_ang, use_container_width=True)

                with hist_col2:
                    fig_w = px.histogram(
                        df_drops, x="width_mm", nbins=15,
                        title="<b>Droplet Width w (mm) Distribution</b>",
                        labels={"width_mm": "Droplet Width w (mm)"},
                        color_discrete_sequence=['#C49A72']
                    )
                    fig_w.update_layout(template="plotly_dark", paper_bgcolor="#3A2618", plot_bgcolor="#2A1B10", font=dict(color="#E8DDD0"))
                    st.plotly_chart(fig_w, use_container_width=True)

        # ---------------------------------------------------------------------
        # TAB 4: 2D CONVERGENCE & 3D POINT OF ORIGIN (MODULE 4)
        # ---------------------------------------------------------------------
        with tab_mod4:
            st.markdown('<div class="dexter-heading">🎯 Module 4: 2D Convergence & 3D Point of Origin Solver</div>', unsafe_allow_html=True)
            
            p2d_col, p3d_col = st.columns([1, 1.2], gap="large")
            with p2d_col:
                st.markdown("#### 📐 2D Convergence Line Ray Tracing")
                st.plotly_chart(res["fig_2d_convergence"], use_container_width=True)
            with p3d_col:
                st.markdown("#### 🌐 3D Spatial Point of Origin Resolution")
                st.plotly_chart(res["fig_3d_origin"], use_container_width=True)

            st.markdown("##### 📊 Individual Droplet Outward Depth Z Estimates ($d = h \\cdot \\tan\\theta$):")
            z_table = pd.DataFrame({
                "Droplet ID": [d["id"] for d in drops_data],
                "Impact Angle θ (°)": [round(d["impact_angle_deg"], 1) for d in drops_data],
                "Distance h to Conv (cm)": [round(np.sqrt((d["centroid"][0]-x_o)**2 + (d["centroid"][1]-y_o)**2), 1) for d in drops_data],
                "Calculated Depth Z (cm)": origin_3d["individual_z_estimates"]
            })
            st.dataframe(z_table, use_container_width=True, height=220)


if __name__ == "__main__":
    main()


