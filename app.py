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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Modern Styling (Tailwind / Modern SaaS aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: radial-gradient(120% 120% at 50% 0%, #1E293B 0%, #0B0F19 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 26px 32px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .hero-desc {
        font-size: 0.98rem;
        color: #94A3B8;
        margin-top: 6px;
        max-width: 900px;
    }

    /* Metric Cards */
    .stat-card {
        background: #131A2A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        position: relative;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        border-color: #38BDF8;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stat-number {
        font-size: 1.9rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
        line-height: 1.1;
    }
    .stat-meta {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Diagnostic Callout Card */
    .diagnostic-card {
        background: #131A2A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    .diagnostic-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #38BDF8;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .diagnostic-text {
        font-size: 0.9rem;
        color: #CBD5E1;
        line-height: 1.55;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #131A2A;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        border: 1px solid #1E293B;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        border-color: #38BDF8 !important;
        color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏋️ Coach Lens")
    st.caption("AI-Powered Gym Form & Biomechanics Engine")
    st.markdown("---")

    # 1. Target Exercise Selector
    exercise_choice = st.selectbox(
        "Select Exercise",
        ["Squat", "Push-up", "Barbell Biceps Curl"],
        index=0,
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

    st.caption(f"🎯 **Tracked Vector:** {joint_label}")
    st.markdown("---")

    # 2. Video Input Selector
    st.markdown("**Workout Video Input**")
    input_source = st.radio(
        "Input Mode",
        ["Dataset Benchmark Sample", "Upload Phone Video (.mp4)"],
        label_visibility="collapsed",
    )

    selected_video_path = None
    if input_source == "Dataset Benchmark Sample":
        sample_options = ["squat_10.mp4", "squat_13.mp4", "squat_15.mp4", "squat_14.mp4"]
        chosen_sample = st.selectbox("Select Sample Clip", sample_options, index=0)
        potential_path = DATASET_DIR / chosen_sample
        if potential_path.exists():
            selected_video_path = potential_path
        else:
            st.warning("Sample not found on local disk. Please upload video.")
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

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Biomechanics", type="primary", use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("🎓 **CSE Final Year Project**")
    st.caption("Explainable AI • MediaPipe BlazePose • FastDTW")


# ==============================================================================
# HERO BANNER
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ Computer Vision & Kinematic Intelligence</div>
    <div class="hero-title">Coach Lens: Biomechanical Workout Analyzer</div>
    <div class="hero-desc">
        Real-time motion analysis that translates 33 3D skeletal landmarks into joint kinematics,
        segments repetitions, and scores workout form against physical therapy reference standards.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# IDLE STATE: EDUCATIONAL ARCHITECTURE EXPLAINER
# ==============================================================================
if not analyze_btn and selected_video_path is None:
    st.info("👈 Select a sample clip or upload your phone video from the sidebar, then click **'Analyze Biomechanics'**.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Module 1 & 2</div>
            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin:6px 0;">3D Landmark Tracking</div>
            <div style="font-size:0.85rem; color:#94A3B8;">
                Extracts 33 human joint coordinates, resolves dominant camera side, and computes scale-invariant angles.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Module 3 & 4</div>
            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin:6px 0;">Repetition Segmentation</div>
            <div style="font-size:0.85rem; color:#94A3B8;">
                Splits continuous video into distinct reps using peak/valley signal analysis with physics-based filters.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Module 5 & 7</div>
            <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC; margin:6px 0;">FastDTW Form Scoring</div>
            <div style="font-size:0.85rem; color:#94A3B8;">
                Aligns repetitions to Golden Reference Templates, generating transparent 0–100 scores and coach tips.
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
        st.warning("⚠️ No valid repetitions detected in this clip. Ensure full range of motion from starting posture to bottom and back.")
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
    st.markdown("### 📊 Workout Performance Summary")
    c1, c2, c3, c4 = st.columns(4)

    score_color = "#34D399" if avg_score >= 80 else ("#FBBF24" if avg_score >= 65 else "#F87171")
    grade_label = "Grade A" if avg_score >= 90 else ("Grade B" if avg_score >= 80 else ("Grade C" if avg_score >= 70 else "Grade D/F"))

    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Overall Form Score</div>
            <div class="stat-number" style="color:{score_color};">{avg_score} <span style="font-size:1.1rem; color:#64748B;">/ 100</span></div>
            <div class="stat-meta">{grade_label} (Biomechanical Average)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Reps Completed</div>
            <div class="stat-number">{len(reps)}</div>
            <div class="stat-meta">Clip Length: {len(landmarks_df)/fps:.1f}s ({fps:.1f} FPS)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Average Depth Angle</div>
            <div class="stat-number">{avg_depth}°</div>
            <div class="stat-meta">Standard: &le; 90.0° Parallel Depth</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Peak Trunk Lean</div>
            <div class="stat-number">{avg_torso}°</div>
            <div class="stat-meta">Safe Threshold: &le; 35.0° from vertical</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. DUAL RADAR & GAUGE VISUALIZATION (PLOTLY)
    # --------------------------------------------------------------------------
    st.markdown("### 🎯 Athletic Biomechanical Profiler")
    gauge_col, radar_col = st.columns([1.0, 1.25])

    with gauge_col:
        # Radial Gauge Indicator for Overall Form Score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            number={"suffix": " / 100", "font": {"size": 32, "color": "#F8FAFC", "family": "Plus Jakarta Sans"}},
            title={"text": "Form Score Metric Gauge", "font": {"size": 15, "color": "#94A3B8"}},
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
            paper_bgcolor="#131A2A",
            plot_bgcolor="#131A2A",
            height=280,
            margin=dict(l=25, r=25, t=40, b=25),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with radar_col:
        # Biomechanical Spider / Radar Chart (5 Key Athletic Dimensions)
        # Compute dimension scores normalized to 100
        depth_score = max(0, min(100, 100 - (max(0, avg_depth - 90.0) * 2.0)))
        posture_score = max(0, min(100, 100 - (max(0, avg_torso - 30.0) * 2.5)))
        dtw_distances = [ev["gaps"]["dtw_distance"] for ev in rep_evaluations]
        smoothness_score = max(0, min(100, 100 - (np.mean(dtw_distances) * 2.5)))
        rom_scores = [min(100, (r["rom_degrees"] / 140.0) * 100.0) for r in reps]
        avg_rom = np.mean(rom_scores)
        tempo_consistency = max(50, 100 - (np.std([r["duration_sec"] for r in reps]) * 20.0))

        radar_categories = ["Depth Accuracy", "Trunk Posture", "Kinematic Smoothness", "Range of Motion", "Tempo Consistency"]
        radar_values = [depth_score, posture_score, smoothness_score, avg_rom, tempo_consistency]
        # Close polygon
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
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#64748B"), gridcolor="#334155"),
                angularaxis=dict(tickfont=dict(size=11, color="#CBD5E1", family="Plus Jakarta Sans"), gridcolor="#334155"),
                bgcolor="#131A2A",
            ),
            paper_bgcolor="#131A2A",
            height=280,
            margin=dict(l=35, r=35, t=30, b=25),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3. INTERACTIVE SKELETON SCRUBBER & VIDEO TELEMETRY
    # --------------------------------------------------------------------------
    st.markdown("### 🎥 Interactive Video & Pose Landmark Scrubber")
    st.write("Examine the raw video or scrub through the motion frame-by-frame with the 33-point skeletal overlay.")

    col_vid, col_pose = st.columns([1.0, 1.0])

    with col_vid:
        st.markdown("**Recorded Workout Video**")
        st.video(str(selected_video_path))

    with col_pose:
        worst_rep_idx = int(np.argmin(overall_scores))
        worst_eval = rep_evaluations[worst_rep_idx]
        worst_abs_frame = worst_eval["worst_moment"]["absolute_frame_idx"]

        st.markdown(f"**Annotated Skeleton Overlay** *(Worst Defect at Frame {worst_abs_frame})*")
        
        # Interactive Frame Scrubber Slider
        scrub_frame = st.slider(
            "Scrub Video Frames:",
            min_value=0,
            max_value=len(annotated_frames) - 1,
            value=min(worst_abs_frame, len(annotated_frames) - 1),
            step=1,
            help="Drag slider to inspect joint tracking across any frame.",
        )

        frame_rgb = cv2.cvtColor(annotated_frames[scrub_frame], cv2.COLOR_BGR2RGB)
        st.image(
            frame_rgb,
            caption=f"Frame {scrub_frame} of {len(annotated_frames)} | Defect Context: {worst_eval['worst_moment']['reason']}",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. INTERACTIVE PLOTLY REPETITION DEEP DIVE
    # --------------------------------------------------------------------------
    st.markdown("### 📈 Interactive Repetition Telemetry & Dynamic Time Warping")
    tabs = st.tabs([f"Repetition #{i+1} ({ev['overall_score']} pts)" for i, ev in enumerate(rep_evaluations)])

    for i, tab in enumerate(tabs):
        with tab:
            eval_data = rep_evaluations[i]
            r_gaps = eval_data["gaps"]
            r_penalties = eval_data["penalties"]
            curves = eval_data["curves"]

            plot_col, details_col = st.columns([1.45, 1.0])

            with plot_col:
                st.markdown("**Interactive Kinematic Curve Alignment (Plotly)**")
                norm_percent = np.linspace(0, 100, len(curves["user_norm_knee"]))
                ref_mean = np.array(curves["ref_mean_knee"])
                ref_std = np.array(curves["ref_std_knee"])
                user_knee = np.array(curves["user_norm_knee"])

                fig_curve = go.Figure()

                # Golden Corridor (±1 Standard Deviation Shading)
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

                # Golden Reference Mean Line
                fig_curve.add_trace(go.Scatter(
                    x=norm_percent,
                    y=ref_mean,
                    mode="lines",
                    line=dict(color="#38BDF8", width=2.5, dash="dash"),
                    name="Golden Reference Mean",
                    hovertemplate="Reference: %{y:.1f}°<extra></extra>",
                ))

                # User's Actual Repetition Curve
                u_color = "#34D399" if eval_data["overall_score"] >= 80 else "#F87171"
                fig_curve.add_trace(go.Scatter(
                    x=norm_percent,
                    y=user_knee,
                    mode="lines",
                    line=dict(color=u_color, width=3.5),
                    name=f"User Rep #{i+1}",
                    hovertemplate="User Angle: %{y:.1f}° at %{x:.0f}%<extra></extra>",
                ))

                # Highlight Parallel Threshold (90 degrees)
                fig_curve.add_hline(
                    y=90.0,
                    line_dash="dot",
                    line_color="#FBBF24",
                    annotation_text="Parallel Depth (90°)",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#FBBF24", size=10),
                )

                fig_curve.update_layout(
                    paper_bgcolor="#131A2A",
                    plot_bgcolor="#131A2A",
                    xaxis=dict(title="Normalized Repetition Progress (%)", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                    yaxis=dict(title="Knee Flexion Angle (°)", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1", size=10)),
                    margin=dict(l=30, r=20, t=30, b=30),
                    height=340,
                )
                st.plotly_chart(fig_curve, use_container_width=True)

            with details_col:
                st.markdown("**Explainable Form Deduction Breakdown**")
                st.markdown(f"""
                <div class="diagnostic-card">
                    <div style="font-size:0.8rem; color:#64748B; text-transform:uppercase; font-weight:600;">Mathematical Deduction</div>
                    <div style="font-size:1.6rem; font-weight:800; color:{score_color}; margin:4px 0;">
                        {eval_data['overall_score']} <span style="font-size:0.95rem; color:#94A3B8;">/ 100 pts ({eval_data['grade']})</span>
                    </div>
                    <hr style="border:0; border-top:1px solid #1E293B; margin:8px 0;">
                    <div style="font-size:0.88rem; color:#E2E8F0; line-height:1.7;">
                        • <b>Depth Penalty:</b> <code>-{r_penalties['depth_penalty']} pts</code> (Bottom: {r_gaps['user_min_knee']}° | Standard: &le; 90°)<br>
                        • <b>Trunk Lean:</b> <code>-{r_penalties['torso_penalty']} pts</code> (Peak Lean: {r_gaps['user_peak_torso']}° | Safe: &le; 35°)<br>
                        • <b>Smoothness:</b> <code>-{r_penalties['trajectory_penalty']} pts</code> (DTW Warp Distance: {r_gaps['dtw_distance']})
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Coach Lens Recommendations:**")
                tips = coach.get_coaching_feedback(eval_data)
                for t in tips["rule_tips"]:
                    st.markdown(f"""
                    <div class="diagnostic-card" style="padding:10px 14px; margin-bottom:8px; border-left:3px solid #38BDF8;">
                        <div class="diagnostic-text">💡 {t}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 5. DETAILED AUDIT TELEMETRY LOG
    # --------------------------------------------------------------------------
    st.markdown("### 📋 Repetition Telemetry Audit Log")
    log_rows = []
    for ev in rep_evaluations:
        log_rows.append({
            "Rep #": ev["rep_num"],
            "Score (0-100)": ev["overall_score"],
            "Grade": ev["grade"],
            "Duration (s)": ev["duration_sec"],
            "Knee Depth Angle": f"{ev['gaps']['user_min_knee']}°",
            "Torso Forward Lean": f"{ev['gaps']['user_peak_torso']}°",
            "Primary Defect Reason": ev["worst_moment"]["reason"],
        })
    log_df = pd.DataFrame(log_rows)
    st.dataframe(log_df, use_container_width=True, hide_index=True)

    csv_data = log_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Repetition Telemetry CSV Report",
        data=csv_data,
        file_name=f"{selected_video_path.stem}_coach_lens_evaluation.csv",
        mime="text/csv",
    )
