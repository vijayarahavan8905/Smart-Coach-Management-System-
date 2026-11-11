"""
Coach Lens - Module 5: Comparison & 0-100 Form Scoring Engine
============================================================
Compares user exercise repetitions against the reference template using
Dynamic Time Warping (FastDTW) and computes a 100% explainable 0-100 Form Score.

Key Concepts:
1. Dynamic Time Warping (DTW): Aligns movement curves across time so fast and
   slow repetitions are compared fairly without penalizing natural pacing.
2. Metric-Specific Error Gaps: Measures exact physical deviations in degrees:
   - Squat Depth Gap (degrees above parallel depth)
   - Torso Lean Gap (degrees leaning excessively forward)
   - Trajectory Consistency Gap (overall shape deviation via DTW)
3. Explainable Jury-Ready Scoring Formula:
   Score = 100 - (Depth Penalty + Torso Penalty + Trajectory Penalty)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from config import TEMPLATES_DIR, OUTPUTS_DIR


class FormScorer:
    """
    Evaluates individual repetition kinematics against a golden reference template.
    """

    def __init__(self, template_path: Optional[Path] = None, exercise_name: str = "squat"):
        if template_path is None:
            self.template_path = TEMPLATES_DIR / f"{exercise_name.lower()}_template.json"
        else:
            self.template_path = Path(template_path)

        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Reference template not found at: {self.template_path}. "
                "Run build_template.py first to generate reference templates."
            )

        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template = json.load(f)

        self.exercise_name = self.template.get("exercise_name", exercise_name)
        self.standard_length = self.template.get("standard_length", 100)
        self.primary_metric = self.template.get("primary_metric", "primary_knee_smooth")
        
        # Extract golden curves
        primary_data = self.template["metrics"][self.primary_metric]
        self.ref_mean_knee = np.array(primary_data["mean"])
        self.ref_std_knee = np.array(primary_data["std"])
        self.ref_bottom_angle = float(np.min(self.ref_mean_knee))

        torso_data = self.template["metrics"].get("torso_lean_smooth", {})
        self.ref_mean_torso = np.array(torso_data.get("mean", np.full(self.standard_length, 20.0)))

    def evaluate_repetition(
        self,
        rep_feature_df: pd.DataFrame,
        rep_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Evaluates a single segmented repetition against the golden template.
        
        Args:
            rep_feature_df: Feature DataFrame containing frames for this single rep.
            rep_metadata: Optional metadata dictionary from RepDetector.
            
        Returns:
            Dictionary containing:
            - overall_score (0-100)
            - metric_scores breakdown
            - physical gaps in degrees
            - aligned curves for visualization
            - worst frame index & defect description
        """
        if rep_feature_df.empty or len(rep_feature_df) < 3:
            return {
                "overall_score": 0.0,
                "grade": "F",
                "feedback": "Repetition too short or invalid to analyze.",
                "gaps": {},
                "penalties": {},
            }

        # 1. Resample User Rep to 100 Normalized Steps
        from build_template import resample_repetition

        user_raw_knee = rep_feature_df[self.primary_metric].to_numpy()
        user_norm_knee = resample_repetition(user_raw_knee, self.standard_length)

        user_raw_torso = (
            rep_feature_df["torso_lean_smooth"].to_numpy()
            if "torso_lean_smooth" in rep_feature_df.columns
            else np.full(len(rep_feature_df), 15.0)
        )
        user_norm_torso = resample_repetition(user_raw_torso, self.standard_length)

        # 2. Dynamic Time Warping (DTW) Alignment
        # Aligns user_norm_knee to ref_mean_knee with absolute difference metric
        dtw_distance, warp_path = fastdtw(
            user_norm_knee,
            self.ref_mean_knee,
            dist=lambda a, b: abs(float(a) - float(b))
        )
        normalized_dtw_distance = round(float(dtw_distance / self.standard_length), 2)

        # 3. Biomechanical Feature Gaps
        user_min_knee = float(np.min(user_norm_knee))
        parallel_threshold = 90.0  # Standard parallel squat depth

        # Depth Gap: Positive if user stayed above parallel (shallow)
        depth_gap_degrees = max(0.0, user_min_knee - parallel_threshold)
        user_peak_torso = float(np.max(user_norm_torso))
        lean_threshold = 35.0      # Acceptable forward lean ceiling

        # Torso Lean Gap: Positive if user leaned forward past threshold
        torso_gap_degrees = max(0.0, user_peak_torso - lean_threshold)

        # 4. Form Scoring Formula (Explainable Point Deduction System)
        # -------------------------------------------------------------
        # Maximum Score: 100 points
        # 1. Depth Penalty (Max 40 pts): 1.5 pts deducted per degree above parallel
        depth_penalty = min(40.0, depth_gap_degrees * 1.5)

        # 2. Torso Lean Penalty (Max 30 pts): 1.5 pts deducted per degree past 35 deg lean
        torso_penalty = min(30.0, torso_gap_degrees * 1.5)

        # 3. Trajectory Consistency Penalty (Max 30 pts): derived from DTW distance
        trajectory_penalty = min(30.0, max(0.0, (normalized_dtw_distance - 4.0) * 2.5))

        # Overall Form Score
        total_penalty = depth_penalty + torso_penalty + trajectory_penalty
        overall_score = max(0.0, min(100.0, round(100.0 - total_penalty, 1)))

        # Assign Letter Grade
        if overall_score >= 90:
            grade = "A (Excellent Form)"
        elif overall_score >= 80:
            grade = "B (Good Form)"
        elif overall_score >= 70:
            grade = "C (Fair - Needs Improvement)"
        elif overall_score >= 50:
            grade = "D (Poor Form - Injury Risk)"
        else:
            grade = "F (Critical Errors)"

        # 5. Worst Frame & Moment Identification
        # Find the frame where deviation was greatest
        # Depth error occurs at bottom; lean error occurs at peak lean
        if depth_penalty >= torso_penalty and depth_gap_degrees > 0:
            # Worst moment was at bottom depth
            worst_local_idx = int(np.argmin(user_raw_knee))
            worst_reason = f"Shallow Depth: Knee angle was {user_min_knee:.1f}° (target <= 90°)"
        elif torso_penalty > 0:
            worst_local_idx = int(np.argmax(user_raw_torso))
            worst_reason = f"Excessive Forward Lean: Chest angled {user_peak_torso:.1f}° forward (target <= 35°)"
        else:
            worst_local_idx = int(np.argmin(user_raw_knee))
            worst_reason = "Good execution! Minor trajectory variance."

        # Map local rep frame index to absolute video frame index
        start_frame_offset = rep_metadata.get("start_frame", 0) if rep_metadata else 0
        worst_absolute_frame = start_frame_offset + worst_local_idx

        result = {
            "exercise_name": self.exercise_name,
            "overall_score": overall_score,
            "grade": grade,
            "rep_num": rep_metadata.get("rep_num", 1) if rep_metadata else 1,
            "duration_sec": rep_metadata.get("duration_sec", 0.0) if rep_metadata else round(len(rep_feature_df) / 30.0, 2),
            "gaps": {
                "depth_gap_degrees": round(depth_gap_degrees, 1),
                "user_min_knee": round(user_min_knee, 1),
                "target_depth": parallel_threshold,
                "torso_gap_degrees": round(torso_gap_degrees, 1),
                "user_peak_torso": round(user_peak_torso, 1),
                "target_lean_max": lean_threshold,
                "dtw_distance": normalized_dtw_distance,
            },
            "penalties": {
                "depth_penalty": round(depth_penalty, 1),
                "torso_penalty": round(torso_penalty, 1),
                "trajectory_penalty": round(trajectory_penalty, 1),
            },
            "worst_moment": {
                "local_frame_idx": worst_local_idx,
                "absolute_frame_idx": worst_absolute_frame,
                "reason": worst_reason,
            },
            "curves": {
                "user_norm_knee": [round(float(x), 1) for x in user_norm_knee],
                "user_norm_torso": [round(float(x), 1) for x in user_norm_torso],
                "ref_mean_knee": [round(float(x), 1) for x in self.ref_mean_knee],
                "ref_std_knee": [round(float(x), 1) for x in self.ref_std_knee],
            },
        }
        return result

    def plot_comparison(
        self,
        eval_result: Dict,
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Renders a diagnostic comparison chart showing the user's rep against
        the reference template corridor with labeled penalty breakdown.
        """
        import matplotlib.pyplot as plt

        rep_num = eval_result["rep_num"]
        if save_path is None:
            save_path = OUTPUTS_DIR / f"{self.exercise_name}_rep{rep_num}_comparison.png"
        else:
            save_path = Path(save_path)
            
        save_path.parent.mkdir(parents=True, exist_ok=True)

        norm_time = np.linspace(0, 100, self.standard_length)
        curves = eval_result["curves"]
        gaps = eval_result["gaps"]
        penalties = eval_result["penalties"]

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")
        ax.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")
        ax.tick_params(colors="#E2E8F0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#475569")

        # Plot Golden Corridor
        ref_mean = np.array(curves["ref_mean_knee"])
        ref_std = np.array(curves["ref_std_knee"])
        ax.plot(norm_time, ref_mean, color="#38BDF8", linewidth=2.5, label="Golden Reference Form (Mean)")
        ax.fill_between(
            norm_time,
            ref_mean - ref_std,
            ref_mean + ref_std,
            color="#38BDF8",
            alpha=0.2,
            label="Ideal Tolerance Band (±1 Std)",
        )

        # Plot User Curve
        user_knee = np.array(curves["user_norm_knee"])
        user_color = "#4ADE80" if eval_result["overall_score"] >= 80 else "#F87171"
        ax.plot(norm_time, user_knee, color=user_color, linewidth=3, label=f"User Rep {rep_num} (Score: {eval_result['overall_score']})")

        # Highlight parallel depth
        ax.axhline(90.0, color="#FACC15", linestyle=":", linewidth=1.5, label="Parallel Depth (90°)")

        # Annotation Box with Scoring Breakdown
        info_text = (
            f"Form Score: {eval_result['overall_score']} / 100 ({eval_result['grade']})\n"
            f"------------------------------------\n"
            f"Knee Depth: {gaps['user_min_knee']}° (Penalty: -{penalties['depth_penalty']} pts)\n"
            f"Torso Lean: {gaps['user_peak_torso']}° (Penalty: -{penalties['torso_penalty']} pts)\n"
            f"Trajectory DTW: {gaps['dtw_distance']} (Penalty: -{penalties['trajectory_penalty']} pts)"
        )
        ax.text(
            0.03, 0.05, info_text,
            transform=ax.transAxes,
            fontsize=10,
            family="monospace",
            color="#F8FAFC",
            bbox=dict(boxstyle="round,pad=0.6", fc="#334155", ec="#64748B", lw=1.5),
        )

        ax.set_title(
            f"Coach Lens - Form Scoring & DTW Alignment (Rep {rep_num})\n"
            f"Overall Score: {eval_result['overall_score']}/100 | {eval_result['worst_moment']['reason']}",
            color="#F8FAFC", fontsize=12, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Normalized Repetition Progress (%)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.set_ylabel("Knee Angle (degrees)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="#F8FAFC", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        return save_path
