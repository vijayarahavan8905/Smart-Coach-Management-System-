"""
Coach Lens - Module 2 Demo: Feature Extraction & Angle Trajectory Plotting
=========================================================================
Extracts joint angles and torso lean from a real squat video, applies
Savitzky-Golay smoothing, automatically chooses the camera-facing side,
and exports a multi-panel visual plot to data/outputs/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATASET_DIR, OUTPUTS_DIR, CACHE_DIR
from pose_extractor import PoseExtractor
from features import FeatureExtractor


def run_module2_demo():
    print("\n=======================================================")
    print("   COACH LENS - MODULE 2: FEATURE EXTRACTION DEMO")
    print("=======================================================")

    test_video_path = DATASET_DIR / "squat_10.mp4"
    if not test_video_path.exists():
        candidates = list(DATASET_DIR.glob("*.mp4")) + list(DATASET_DIR.glob("*.MOV"))
        if not candidates:
            raise FileNotFoundError(f"No video files found in {DATASET_DIR}")
        test_video_path = sorted(candidates)[0]

    print(f"Target Video: {test_video_path.name}")

    # Step 1: Load landmarks (instant cache hit from Module 1)
    extractor = PoseExtractor()
    landmarks_df, fps, width, height = extractor.get_or_extract(
        test_video_path, cache_dir=CACHE_DIR
    )
    print(f"Loaded {len(landmarks_df)} frames of landmarks at {fps:.1f} fps.")

    # Step 2: Extract kinematic features
    feat_extractor = FeatureExtractor(smoothing_window=9, polyorder=2)
    features_df = feat_extractor.extract_features(landmarks_df, fps=fps)

    dominant_side = features_df["dominant_side"].iloc[0]
    left_vis = features_df["left_vis_score"].iloc[0]
    right_vis = features_df["right_vis_score"].iloc[0]

    print("\n--- Side Visibility Analysis ---")
    print(f"  * Left side average visibility:  {left_vis:.3f}")
    print(f"  * Right side average visibility: {right_vis:.3f}")
    print(f"  --> Automatically Selected Dominant Side: [{dominant_side.upper()}]")

    # Step 3: Biomechanical Metrics Summary
    knee_min = features_df["primary_knee_smooth"].min()
    knee_max = features_df["primary_knee_smooth"].max()
    hip_min = features_df["primary_hip_smooth"].min()
    hip_max = features_df["primary_hip_smooth"].max()
    torso_max = features_df["torso_lean_smooth"].max()

    # Find deepest frame (minimum knee angle)
    deepest_idx = int(features_df["primary_knee_smooth"].idxmin())
    deepest_time = features_df["timestamp_sec"].iloc[deepest_idx]

    print("\n--- Biomechanical Angle Trajectory Summary ---")
    print(f"  * Knee Angle Range:       {knee_min:.1f} deg (deepest bottom) -> {knee_max:.1f} deg (standing upright)")
    print(f"  * Deepest Squat Moment:   Frame {deepest_idx} at {deepest_time:.2f} seconds")
    print(f"  * Hip Angle Range:        {hip_min:.1f} deg -> {hip_max:.1f} deg")
    print(f"  * Peak Torso Forward Lean: {torso_max:.1f} deg relative to vertical")

    # Step 4: Multi-panel Trajectory Plotting
    print("\n--- Generating Multi-Panel Kinematic Trajectory Plot ---")
    time_axis = features_df["timestamp_sec"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.patch.set_facecolor("#0F172A")  # Modern slate dark background

    # Styling helper for dark mode plots
    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")
        ax.tick_params(colors="#E2E8F0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Panel 1: Primary Knee Angle (Raw vs. Smoothed)
    ax1 = axes[0]
    ax1.plot(time_axis, features_df["primary_knee_raw"], color="#F87171", alpha=0.5, linewidth=1.5, label="Raw Knee Angle (Jittery)")
    ax1.plot(time_axis, features_df["primary_knee_smooth"], color="#38BDF8", linewidth=2.5, label="Savitzky-Golay Smoothed")
    ax1.scatter([deepest_time], [knee_min], color="#FACC15", s=90, zorder=5, label=f"Deepest Bottom ({knee_min:.1f}° at {deepest_time:.2f}s)")
    ax1.axhline(90, color="#4ADE80", linestyle=":", linewidth=1.5, label="Parallel Depth Threshold (90°)")
    ax1.set_ylabel("Knee Angle (deg)", color="#F1F5F9", fontsize=11, fontweight="bold")
    ax1.set_title(f"Coach Lens - Kinematic Angle Trajectories ({test_video_path.name})", color="#F8FAFC", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="#F8FAFC", fontsize=9)

    # Panel 2: Primary Hip Angle
    ax2 = axes[1]
    ax2.plot(time_axis, features_df["primary_hip_smooth"], color="#A78BFA", linewidth=2.5, label="Primary Hip Flexion Angle")
    ax2.set_ylabel("Hip Angle (deg)", color="#F1F5F9", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="#F8FAFC", fontsize=9)

    # Panel 3: Torso Forward Lean Angle
    ax3 = axes[2]
    ax3.plot(time_axis, features_df["torso_lean_smooth"], color="#FB923C", linewidth=2.5, label="Torso Forward Lean from Vertical")
    ax3.axhline(45, color="#EF4444", linestyle=":", linewidth=1.5, label="Excessive Lean Alert (>45°)")
    ax3.set_ylabel("Torso Lean (deg)", color="#F1F5F9", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="#F8FAFC", fontsize=9)

    # Panel 4: Left vs Right Side Visibility
    ax4 = axes[3]
    ax4.plot(time_axis, features_df["left_vis_score"], color="#34D399", linewidth=2, label="Left Side Joints Visibility")
    ax4.plot(time_axis, features_df["right_vis_score"], color="#F472B6", linewidth=2, label="Right Side Joints Visibility (Occluded)")
    ax4.set_ylabel("Visibility (0-1)", color="#F1F5F9", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time (seconds)", color="#F1F5F9", fontsize=11, fontweight="bold")
    ax4.set_ylim(-0.05, 1.1)
    ax4.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="#F8FAFC", fontsize=9)

    plt.tight_layout()
    plot_path = OUTPUTS_DIR / f"{test_video_path.stem}_angle_trajectories.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED PLOT] Output saved to: {plot_path.name}")

    # Step 5: Save Features to CSV for Next Modules
    features_csv_path = CACHE_DIR / f"{test_video_path.stem}_features.csv"
    features_df.to_csv(features_csv_path, index=False)
    print(f"  [SAVED FEATURES] Exported features table to: {features_csv_path.name}")

    print("\n=======================================================")
    print("   MODULE 2 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    return features_df, plot_path


if __name__ == "__main__":
    run_module2_demo()
