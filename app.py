"""
Coach Lens - Next-Gen AI Gym Trainer Dashboard
==============================================
Modern, interactive computer-vision sports analytics dashboard for CSE Capstone.
Features:
- Plotly Biomechanical Radar / Spider Telemetry Chart
- Interactive DTW Time-Series Alignment Curves with Hover Tooltips
- Circular Score Indicator Gauge
- Interactive Skeleton Frame Scrubber (Scrub through any moment of the rep)
- Executive Form Diagnostic Card & CSV Telemetry Export
"""

import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Project modules
from config import (
    DATASET_DIR,
    DATA_DIR,
    SAMPLES_DIR,
    CACHE_DIR,
    TEMPLATES_DIR,
    OUTPUTS_DIR,
    EXERCISE_CONFIGS_DIR,
)
from pose_extractor import PoseExtractor
from features import FeatureExtractor
from rep_detector import RepDetector
from compare import FormScorer
from coach import CoachingAdvisor

# ==============================================================================
# PAGE CONFIGURATION & NEXT-GEN MODERN STYLING
# ==============================================================================
st.set_page_config(
    page_title="Coach Lens | AI Sports Biomechanics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Modern Styling (Tailwind / Modern SaaS aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Spacing */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #111827 50%, #0B0F19 100%);
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 26px 32px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #34D399);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(56, 189, 248, 0.1);
        color: #38BDF8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin-top: 8px;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 8px;
        max-width: 880px;
        line-height: 1.6;
    }

    /* Metric Cards */
    .stat-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px 22px;
        position: relative;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
        box-shadow: 0 8px 24px -4px rgba(56, 189, 248, 0.15);
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stat-number {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .stat-meta {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 6px;
    }

    /* Section Header */
    .section-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 28px 0 16px 0;
    }
    .section-bar {
        width: 4px;
        height: 22px;
        background: #38BDF8;
        border-radius: 2px;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.01em;
    }
    .section-desc {
        font-size: 0.82rem;
        color: #64748B;
        margin-left: auto;
    }

    /* Diagnostic Callout Card */
    .diagnostic-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* Sidebar Brand Card */
    .sidebar-brand {
        background: linear-gradient(180deg, #1E293B 0%, #111827 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 18px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 8px;
        padding: 8px 18px;
        border: 1px solid #1E293B;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        border-color: #38BDF8 !important;
        color: #F8FAFC !important;
    }

    /* Primary Button Glow */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        border: 1px solid #38BDF8;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.45);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:0.7rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#38BDF8; background:rgba(56,189,248,0.12); padding:2px 8px; border-radius:999px;">SPORTS AI LAB</span>
            <span style="font-size:0.7rem; color:#64748B; font-weight:600;">v2.4</span>
        </div>
        <div style="font-size:1.35rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">Coach Lens</div>
        <div style="font-size:0.8rem; color:#94A3B8; margin-top:2px;">Biomechanical Kinematic Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Target Exercise Selector
    st.markdown("<div style='font-size:0.72rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>01 // Target Exercise</div>", unsafe_allow_html=True)
    exercise_choice = st.selectbox(
        "Select Exercise",
        ["Squat", "Push-up", "Barbell Biceps Curl"],
        index=0,
        label_visibility="collapsed",
    )
    if exercise_choice == "Squat":
        exercise_name = "squat"
        joint_label = "Knee Joint Flexion (Target <= 90°)"
    elif exercise_choice == "Push-up":
        exercise_name = "pushup"
        joint_label = "Elbow Joint Flexion (Target <= 90°)"
    else:
        exercise_name = "curl"
        joint_label = "Biceps Contraction (Target <= 65°)"

    st.markdown(f"""
    <div style="background:rgba(15,23,42,0.6); border:1px solid #1E293B; border-radius:8px; padding:8px 12px; margin-top:6px; margin-bottom:18px;">
        <div style="font-size:0.68rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em;">Tracked Joint Vector</div>
        <div style="font-size:0.82rem; font-weight:600; color:#38BDF8; margin-top:2px;">{joint_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Video Input Selector
    st.markdown("<div style='font-size:0.72rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>02 // Workout Video Input</div>", unsafe_allow_html=True)
    input_source = st.radio(
        "Input Mode",
        ["Dataset Benchmark Sample", "Upload Phone Video (.mp4)"],
        label_visibility="collapsed",
    )

    selected_video_path = None
    if input_source == "Dataset Benchmark Sample":
        # Search candidate locations for exercise video clips
        candidate_dirs = [
            SAMPLES_DIR,
            Path(r"C:\Users\vijay\OneDrive\Documents\Desktop\data set\squats"),
            DATASET_DIR,
            Path(r"C:\Users\vijay\OneDrive\Documents\Desktop\data set"),
        ]
        sample_dict = {}
        for cdir in candidate_dirs:
            try:
                if cdir and cdir.exists():
                    for vfile in sorted(cdir.glob("*.mp4")):
                        if vfile.name not in sample_dict:
                            sample_dict[vfile.name] = vfile
            except Exception:
                continue

        if sample_dict:
            chosen_sample = st.selectbox("Select Sample Clip", list(sample_dict.keys()), index=0)
            selected_video_path = sample_dict[chosen_sample]
            st.caption(f"Benchmark Clip: {chosen_sample} ({selected_video_path.stat().st_size // 1024} KB)")
        else:
            st.warning("No benchmark clips found. Please upload a workout video below.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Workout Video",
            type=["mp4", "mov", "m4v"],
            help="Upload a phone video of someone exercising (side view recommended).",
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            selected_video_path = Path(tfile.name)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyze Biomechanics", type="primary", use_container_width=True)

    st.markdown("""
    <div style="border-top:1px solid #1E293B; padding-top:14px; margin-top:28px;">
        <div style="display:flex; align-items:center; gap:8px; font-size:0.75rem; color:#34D399; font-weight:600;">
            <div style="width:7px; height:7px; background:#34D399; border-radius:50%; box-shadow:0 0 6px #34D399;"></div>
            Kinematic Engine Ready
        </div>
        <div style="font-size:0.72rem; color:#64748B; margin-top:4px;">
            Explainable AI • MediaPipe BlazePose • FastDTW
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# HERO BANNER
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
        <span class="hero-badge">Computer Vision</span>
        <span class="hero-badge" style="color:#A78BFA; background:rgba(167,139,250,0.1); border-color:rgba(167,139,250,0.3);">33 3D BlazePose Landmarks</span>
        <span class="hero-badge" style="color:#34D399; background:rgba(52,211,153,0.1); border-color:rgba(52,211,153,0.3);">FastDTW Kinematic Alignment</span>
    </div>
    <div class="hero-title">Coach Lens: Biomechanical Workout Analyzer</div>
    <div class="hero-desc">
        Real-time motion analysis that translates 33 3D skeletal landmarks into joint kinematics,
        segments repetitions via kinetic inflection filtering, and scores workout form against physical therapy reference standards.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# IDLE STATE: EDUCATIONAL ARCHITECTURE EXPLAINER
# ==============================================================================
if not analyze_btn and selected_video_path is None:
    st.info("Select a sample clip or upload your phone video from the sidebar, then click **'Analyze Biomechanics'**.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="stat-card" style="border-top:3px solid #38BDF8;">
            <div class="stat-label">Phase 01 // Computer Vision</div>
            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin:8px 0 6px 0;">3D Landmark Tracking</div>
            <div style="font-size:0.85rem; color:#94A3B8; line-height:1.5;">
                Extracts 33 human joint coordinates, resolves dominant camera side, and computes scale-invariant angles.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="stat-card" style="border-top:3px solid #818CF8;">
            <div class="stat-label">Phase 02 // Signal Processing</div>
            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin:8px 0 6px 0;">Repetition Segmentation</div>
            <div style="font-size:0.85rem; color:#94A3B8; line-height:1.5;">
                Splits continuous video into distinct reps using peak/valley signal analysis with physics-based filters.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="stat-card" style="border-top:3px solid #34D399;">
            <div class="stat-label">Phase 03 // Biomechanics</div>
            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin:8px 0 6px 0;">FastDTW Form Scoring</div>
            <div style="font-size:0.85rem; color:#94A3B8; line-height:1.5;">
                Aligns repetitions to Golden Reference Templates, generating transparent 0–100 scores and coach tips.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 20px 24px; margin-top: 22px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
            <div>
                <span style="font-size: 0.72rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.08em;">Kinematic Benchmark Database</span>
                <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">Clinical Reference Specifications</div>
            </div>
            <span style="font-size: 0.72rem; color: #34D399; font-weight: 600; background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.25); padding: 3px 10px; border-radius: 999px;">CALIBRATED STANDARDS</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;">
            <div style="background: #182235; border: 1px solid #23324D; border-radius: 8px; padding: 14px 16px;">
                <div style="color: #38BDF8; font-weight: 700; font-size: 0.95rem;">Barbell Squat</div>
                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 6px; line-height: 1.6;">
                    • <b>Target Depth:</b> Knee Flexion &le; 90°<br>
                    • <b>Torso Stability:</b> Forward Lean &le; 35°<br>
                    • <b>Key Plane:</b> Sagittal (Side View)
                </div>
            </div>
            <div style="background: #182235; border: 1px solid #23324D; border-radius: 8px; padding: 14px 16px;">
                <div style="color: #34D399; font-weight: 700; font-size: 0.95rem;">Standard Push-up</div>
                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 6px; line-height: 1.6;">
                    • <b>Chest Depth:</b> Elbow Flexion &le; 90°<br>
                    • <b>Plank Stability:</b> Trunk Deviation &le; 20°<br>
                    • <b>Key Plane:</b> Lateral Arm Flexion
                </div>
            </div>
            <div style="background: #182235; border: 1px solid #23324D; border-radius: 8px; padding: 14px 16px;">
                <div style="color: #A78BFA; font-weight: 700; font-size: 0.95rem;">Barbell Biceps Curl</div>
                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 6px; line-height: 1.6;">
                    • <b>Peak Squeeze:</b> Elbow Flexion &le; 65°<br>
                    • <b>Strict Form:</b> Elbow Forward Drift &le; 15°<br>
                    • <b>Key Plane:</b> Upper Arm Isolation
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PROCESSING & ANALYTICS PIPELINE
# ==============================================================================
if analyze_btn and selected_video_path is not None:
    with st.spinner("Processing video frames through MediaPipe & Biomechanical Kinematic Engine..."):
        try:
            # 1. Pose Landmark Extraction
            extractor = PoseExtractor()
            landmarks_df, annotated_frames, fps, width, height = extractor.extract_landmarks_from_video(
                selected_video_path, save_annotated_frames=True
            )

            # 2. Biomechanical Feature Extraction
            feat_extractor = FeatureExtractor(smoothing_window=9, polyorder=2)
            features_df = feat_extractor.extract_features(landmarks_df, fps=fps)

            # 3. Repetition Boundary Segmentation
            rep_detector = RepDetector(exercise_name=exercise_name)
            reps = rep_detector.detect_reps(features_df, fps=fps)

            # 4. Form Scoring & FastDTW Comparison
            scorer = FormScorer(exercise_name=exercise_name)
            coach = CoachingAdvisor()

            rep_evaluations = []
            for r in reps:
                start_f = r["start_frame"]
                end_f = r["end_frame"]
                rep_slice = features_df.iloc[start_f : end_f + 1].copy()
                eval_res = scorer.evaluate_repetition(rep_slice, rep_metadata=r)
                rep_evaluations.append(eval_res)

        except Exception as err:
            st.error(f"Execution Error: {err}")
            st.stop()

    if not reps:
        st.warning("No valid repetitions detected in this clip. Ensure full range of motion from starting posture to bottom and back.")
        st.stop()

    # Calculate High-Level Averages
    overall_scores = [ev["overall_score"] for ev in rep_evaluations]
    avg_score = round(float(np.mean(overall_scores)), 1)
    min_knees = [ev["gaps"]["user_min_knee"] for ev in rep_evaluations]
    avg_depth = round(float(np.mean(min_knees)), 1)
    peak_torsos = [ev["gaps"]["user_peak_torso"] for ev in rep_evaluations]
    avg_torso = round(float(np.mean(peak_torsos)), 1)

    # --------------------------------------------------------------------------
    # 1. EXECUTIVE KPI METRICS ROW
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="section-wrap">
        <div class="section-bar"></div>
        <div class="section-title">Workout Performance Summary</div>
        <div class="section-desc">Automated Kinematic Telemetry</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    score_color = "#34D399" if avg_score >= 80 else ("#FBBF24" if avg_score >= 65 else "#F87171")
    grade_label = "Grade A" if avg_score >= 90 else ("Grade B" if avg_score >= 80 else ("Grade C" if avg_score >= 70 else "Grade D/F"))

    target_depth = 90.0 if exercise_name in ("squat", "pushup") else 65.0
    if exercise_name == "squat":
        depth_label = "Average Knee Depth"
        depth_status = "Breaking Parallel" if avg_depth <= target_depth else "Above Parallel"
    elif exercise_name == "pushup":
        depth_label = "Average Elbow Depth"
        depth_status = "Breaking 90°" if avg_depth <= target_depth else "Shallow Depth"
    else:
        depth_label = "Biceps Contraction"
        depth_status = "Full Contraction" if avg_depth <= target_depth else "Partial Range"

    depth_color = "#34D399" if avg_depth <= target_depth else "#FBBF24"

    torso_status = "Safe Alignment" if avg_torso <= 35.0 else "Excessive Lean"
    torso_color = "#34D399" if avg_torso <= 35.0 else "#F87171"

    with c1:
        st.markdown(f"""
        <div class="stat-card" style="border-top:3px solid {score_color};">
            <div class="stat-label">Overall Form Score</div>
            <div class="stat-number" style="color:{score_color};">{avg_score} <span style="font-size:1.05rem; color:#64748B; font-weight:600;">/ 100</span></div>
            <div class="stat-meta"><span style="background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:4px; font-weight:700; color:{score_color};">{grade_label}</span> • Biomechanical Mean</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card" style="border-top:3px solid #38BDF8;">
            <div class="stat-label">Reps Completed</div>
            <div class="stat-number" style="color:#F8FAFC;">{len(reps)}</div>
            <div class="stat-meta">Duration: {len(landmarks_df)/fps:.1f}s • {fps:.1f} FPS</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card" style="border-top:3px solid {depth_color};">
            <div class="stat-label">{depth_label}</div>
            <div class="stat-number" style="color:#F8FAFC;">{avg_depth}°</div>
            <div class="stat-meta"><span style="color:{depth_color}; font-weight:600;">{depth_status}</span> • Target: &le; {target_depth:.0f}°</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card" style="border-top:3px solid {torso_color};">
            <div class="stat-label">Peak Trunk Lean</div>
            <div class="stat-number" style="color:#F8FAFC;">{avg_torso}°</div>
            <div class="stat-meta"><span style="color:{torso_color}; font-weight:600;">{torso_status}</span> • Limit: &le; 35°</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate metrics, best/worst rep, coaching advice and figures
    worst_rep_idx = int(np.argmin(overall_scores))
    worst_eval = rep_evaluations[worst_rep_idx]
    best_rep_idx = int(np.argmax(overall_scores))
    best_eval = rep_evaluations[best_rep_idx]
    tips = coach.get_coaching_feedback(worst_eval)
    worst_abs_frame = worst_eval["worst_moment"]["absolute_frame_idx"]

    # Biomechanical Form Score Circular Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score,
        number={"suffix": " / 100", "font": {"size": 32, "color": "#F8FAFC", "family": "Plus Jakarta Sans"}},
        title={"text": "Biomechanical Form Score", "font": {"size": 13, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
            "bar": {"color": score_color, "thickness": 0.28},
            "bgcolor": "#1E293B",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 60], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [60, 80], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [80, 100], "color": "rgba(16, 185, 129, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#38BDF8", "width": 3},
                "thickness": 0.75,
                "value": 90.0,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=265,
        margin=dict(l=20, r=20, t=35, b=20),
    )

    # Biomechanical Spider Radar Chart
    depth_score = max(0, min(100, 100 - (max(0, avg_depth - target_depth) * 2.0)))
    posture_score = max(0, min(100, 100 - (max(0, avg_torso - 30.0) * 2.5)))
    dtw_distances = [ev["gaps"]["dtw_distance"] for ev in rep_evaluations]
    smoothness_score = max(0, min(100, 100 - (np.mean(dtw_distances) * 2.5)))
    rom_scores = [min(100, (r["rom_degrees"] / 140.0) * 100.0) for r in reps]
    avg_rom = np.mean(rom_scores)
    tempo_consistency = max(50, 100 - (np.std([r["duration_sec"] for r in reps]) * 20.0))

    radar_categories = ["Depth", "Posture", "Smoothness", "ROM", "Tempo"]
    radar_values = [depth_score, posture_score, smoothness_score, avg_rom, tempo_consistency]
    radar_categories_closed = radar_categories + [radar_categories[0]]
    radar_values_closed = radar_values + [radar_values[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values_closed,
        theta=radar_categories_closed,
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.2)",
        line=dict(color="#38BDF8", width=2.5),
        name="Lifter Performance",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8, color="#64748B"), gridcolor="#1E293B"),
            angularaxis=dict(tickfont=dict(size=10, color="#CBD5E1", family="Plus Jakarta Sans"), gridcolor="#1E293B"),
            bgcolor="#111827",
        ),
        paper_bgcolor="#111827",
        height=265,
        margin=dict(l=30, r=30, t=25, b=20),
        showlegend=False,
    )

    # --------------------------------------------------------------------------
    # MODULAR RENDER FUNCTIONS
    # --------------------------------------------------------------------------
    def render_studio_component(is_unified=True):
        st.markdown("""
        <div style="background:#111827; border:1px solid #1E293B; border-radius:8px 8px 0 0; padding:10px 14px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; font-weight:700; color:#38BDF8; text-transform:uppercase; letter-spacing:0.06em;">Visual Kinematic Studio</span>
            <span style="font-size:0.72rem; color:#64748B;">MediaPipe BlazePose 3D</span>
        </div>
        """, unsafe_allow_html=True)
        
        vis_tab1, vis_tab2 = st.tabs(["BlazePose Joint Overlay", "Raw Workout Video"])
        with vis_tab1:
            scrub_key = f"scrub_{'unified' if is_unified else 'tabbed'}"
            scrub_frame = st.slider(
                "Scrub Video Frames:",
                min_value=0,
                max_value=len(annotated_frames) - 1,
                value=min(worst_abs_frame, len(annotated_frames) - 1),
                step=1,
                key=scrub_key,
                help="Drag slider to inspect joint tracking across any frame.",
            )

            frame_rgb = cv2.cvtColor(annotated_frames[scrub_frame], cv2.COLOR_BGR2RGB)
            st.image(
                frame_rgb,
                caption=f"Frame {scrub_frame} of {len(annotated_frames)} | Timestamp: {scrub_frame/fps:.2f}s",
                use_container_width=True,
            )

            if abs(scrub_frame - worst_abs_frame) <= 3:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.3); border-radius:8px; padding:10px 14px; margin-top:8px;">
                    <span style="color:#F87171; font-weight:700; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em;">Primary Form Defect Detected:</span>
                    <div style="color:#FCA5A5; font-size:0.85rem; margin-top:2px;">{worst_eval['worst_moment']['reason']} (Observed around Frame {worst_abs_frame})</div>
                </div>
                """, unsafe_allow_html=True)

        with vis_tab2:
            st.video(str(selected_video_path))

    def render_analytics_component():
        g_col, r_col = st.columns([1.0, 1.1])
        with g_col:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with r_col:
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("<div style='height: 1px; background: #1E293B; margin: 12px 0 12px 0;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="diagnostic-card" style="border-left: 3px solid #F87171; margin-bottom:8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #F87171; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px;">Primary Technique Observation</div>
            <div style="font-size: 0.92rem; color: #F8FAFC; font-weight: 600;">{worst_eval['worst_moment']['reason']}</div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Observed in Repetition #{worst_eval['rep_num']} (Score: {worst_eval['overall_score']} / 100)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="diagnostic-card" style="border-left: 3px solid #34D399; margin-bottom:8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #34D399; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px;">Peak Execution Benchmark</div>
            <div style="font-size: 0.92rem; color: #F8FAFC; font-weight: 600;">Repetition #{best_eval['rep_num']} Form Standard</div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Highest biomechanical score: {best_eval['overall_score']} / 100 ({best_eval['grade']})</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin: 12px 0 8px 0;'>Kinematic Coaching Directives</div>", unsafe_allow_html=True)
        for t in tips["rule_tips"]:
            st.markdown(f"""
            <div class="diagnostic-card" style="padding: 10px 14px; margin-bottom: 6px; border-left: 3px solid #38BDF8;">
                <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.45;">• {t}</div>
            </div>
            """, unsafe_allow_html=True)

    def render_dtw_component():
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1E293B; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.06em;">Dynamic Time Warping (FastDTW) Trajectory Comparison</div>
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 2px;">
                Each repetition is time-normalized to 100 percentage steps and aligned against the Golden Reference Template corridor (shaded band = ±1 standard deviation).
            </div>
        </div>
        """, unsafe_allow_html=True)

        rep_tabs = st.tabs([f"Repetition #{i+1} ({ev['overall_score']} pts)" for i, ev in enumerate(rep_evaluations)])

        for i, rep_tab in enumerate(rep_tabs):
            with rep_tab:
                eval_data = rep_evaluations[i]
                r_gaps = eval_data["gaps"]
                r_penalties = eval_data["penalties"]
                curves = eval_data["curves"]

                plot_col, details_col = st.columns([1.45, 1.0])

                with plot_col:
                    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>Kinematic Curve Alignment (Plotly)</div>", unsafe_allow_html=True)
                    norm_percent = np.linspace(0, 100, len(curves["user_norm_knee"]))
                    ref_mean = np.array(curves["ref_mean_knee"])
                    ref_std = np.array(curves["ref_std_knee"])
                    user_knee = np.array(curves["user_norm_knee"])

                    fig_curve = go.Figure()

                    fig_curve.add_trace(go.Scatter(
                        x=np.concatenate([norm_percent, norm_percent[::-1]]),
                        y=np.concatenate([ref_mean + ref_std, (ref_mean - ref_std)[::-1]]),
                        fill="toself",
                        fillcolor="rgba(56, 189, 248, 0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=True,
                        name="Acceptable Corridor (±1 Std Dev)",
                    ))

                    fig_curve.add_trace(go.Scatter(
                        x=norm_percent,
                        y=ref_mean,
                        mode="lines",
                        line=dict(color="#38BDF8", width=2.5, dash="dash"),
                        name="Golden Reference Mean",
                        hovertemplate="Reference: %{y:.1f}°<extra></extra>",
                    ))

                    u_color = "#34D399" if eval_data["overall_score"] >= 80 else "#F87171"
                    fig_curve.add_trace(go.Scatter(
                        x=norm_percent,
                        y=user_knee,
                        mode="lines",
                        line=dict(color=u_color, width=3.5),
                        name=f"User Rep #{i+1}",
                        hovertemplate="User Angle: %{y:.1f}° at %{x:.0f}%<extra></extra>",
                    ))

                    fig_curve.add_hline(
                        y=target_depth,
                        line_dash="dot",
                        line_color="#FBBF24",
                        annotation_text=f"Target ({target_depth:.0f}°)",
                        annotation_position="bottom right",
                        annotation_font=dict(color="#FBBF24", size=10),
                    )

                    fig_curve.update_layout(
                        paper_bgcolor="#111827",
                        plot_bgcolor="#111827",
                        xaxis=dict(title="Normalized Repetition Progress (%)", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                        yaxis=dict(title=f"{'Knee' if exercise_name == 'squat' else 'Elbow'} Flexion Angle (°)", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1", size=10)),
                        margin=dict(l=30, r=20, t=30, b=30),
                        height=340,
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)

                with details_col:
                    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>Biomechanical Deductions</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="diagnostic-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:0.72rem; color:#64748B; text-transform:uppercase; font-weight:700; letter-spacing:0.06em;">Repetition Grade</span>
                            <span style="font-size:0.75rem; font-weight:700; color:{u_color}; background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:4px;">{eval_data['grade']}</span>
                        </div>
                        <div style="font-size:2rem; font-weight:800; color:{u_color}; line-height:1.1;">
                            {eval_data['overall_score']} <span style="font-size:1.0rem; color:#64748B; font-weight:600;">/ 100 pts</span>
                        </div>
                        <div style="height:1px; background:#1E293B; margin:12px 0;"></div>
                        <div style="display:flex; flex-direction:column; gap:8px; font-size:0.85rem; color:#E2E8F0;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span>Depth Penalty (Bottom: {r_gaps['user_min_knee']}°):</span>
                                <span style="font-family:'JetBrains Mono',monospace; color:{'#F87171' if r_penalties['depth_penalty'] > 0 else '#34D399'}; font-weight:700;">-{r_penalties['depth_penalty']} pts</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span>Trunk Lean Penalty (Peak: {r_gaps['user_peak_torso']}°):</span>
                                <span style="font-family:'JetBrains Mono',monospace; color:{'#F87171' if r_penalties['torso_penalty'] > 0 else '#34D399'}; font-weight:700;">-{r_penalties['torso_penalty']} pts</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span>Trajectory Jitter (DTW: {r_gaps['dtw_distance']}):</span>
                                <span style="font-family:'JetBrains Mono',monospace; color:{'#F87171' if r_penalties['trajectory_penalty'] > 0 else '#34D399'}; font-weight:700;">-{r_penalties['trajectory_penalty']} pts</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin:14px 0 8px 0;'>Coach Recommendations</div>", unsafe_allow_html=True)
                    rep_tips = coach.get_coaching_feedback(eval_data)
                    for t in rep_tips["rule_tips"]:
                        st.markdown(f"""
                        <div class="diagnostic-card" style="padding:10px 14px; margin-bottom:8px; border-left:3px solid #38BDF8;">
                            <div style="font-size:0.88rem; color:#CBD5E1; line-height:1.5;">{t}</div>
                        </div>
                        """, unsafe_allow_html=True)

    def render_audit_component():
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="stat-card" style="padding: 12px 16px;">
                <div class="stat-label">Total Reps Logged</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">{len(reps)}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="stat-card" style="padding: 12px 16px;">
                <div class="stat-label">Highest Rep Score</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #34D399; margin-top: 2px;">{max(overall_scores)} <span style="font-size: 0.85rem; color: #64748B;">/ 100</span></div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="stat-card" style="padding: 12px 16px;">
                <div class="stat-label">Lowest Rep Score</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #F87171; margin-top: 2px;">{min(overall_scores)} <span style="font-size: 0.85rem; color: #64748B;">/ 100</span></div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            mean_dur = np.mean([r['duration_sec'] for r in reps])
            st.markdown(f"""
            <div class="stat-card" style="padding: 12px 16px;">
                <div class="stat-label">Mean Rep Duration</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #38BDF8; margin-top: 2px;">{mean_dur:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        log_rows = []
        for ev in rep_evaluations:
            log_rows.append({
                "Rep #": ev["rep_num"],
                "Score (0-100)": ev["overall_score"],
                "Grade": ev["grade"],
                "Duration (s)": ev["duration_sec"],
                f"{'Knee' if exercise_name == 'squat' else 'Elbow'} Depth Angle": f"{ev['gaps']['user_min_knee']}°",
                "Torso Forward Lean": f"{ev['gaps']['user_peak_torso']}°",
                "Primary Defect Reason": ev["worst_moment"]["reason"],
            })
        log_df = pd.DataFrame(log_rows)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        csv_data = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Repetition Telemetry CSV Report",
            data=csv_data,
            file_name=f"{selected_video_path.stem}_coach_lens_evaluation.csv",
            mime="text/csv",
        )

    # ==========================================================================
    # 2. WORKSPACE PRESENTATION LAYOUT
    # ==========================================================================
    c_mode_l, c_mode_r = st.columns([1.5, 1.2])
    with c_mode_l:
        st.markdown("""
        <div style="font-size:0.75rem; font-weight:700; color:#38BDF8; text-transform:uppercase; letter-spacing:0.06em; margin-top:6px;">
            Biomechanical Cockpit Layout
        </div>
        """, unsafe_allow_html=True)
    with c_mode_r:
        layout_mode = st.radio(
            "Cockpit Mode",
            ["Unified Dual-Pane Cockpit", "Detailed Tabbed View"],
            horizontal=True,
            label_visibility="collapsed",
        )

    if layout_mode == "Unified Dual-Pane Cockpit":
        # Side-by-Side Dual-Pane Layout: Video/Skeleton on Left, Analytics/Radar on Right
        left_col, right_col = st.columns([1.05, 0.95], gap="medium")
        with left_col:
            render_studio_component(is_unified=True)
        with right_col:
            render_analytics_component()

        st.markdown("<div style='height: 1px; background: #1E293B; margin: 28px 0 20px 0;'></div>", unsafe_allow_html=True)
        render_dtw_component()

        st.markdown("<div style='height: 1px; background: #1E293B; margin: 28px 0 20px 0;'></div>", unsafe_allow_html=True)
        render_audit_component()

    else:
        tab_profile, tab_studio, tab_dtw, tab_audit = st.tabs([
            "Performance Profiler & Coaching",
            "Video & Skeletal Studio",
            "DTW Kinematic Curves",
            "Telemetry Audit & Export",
        ])
        with tab_profile:
            render_analytics_component()
        with tab_studio:
            render_studio_component(is_unified=False)
        with tab_dtw:
            render_dtw_component()
        with tab_audit:
            render_audit_component()
