"""
Coach Lens - Module 3: Repetition Detection & Boundary Segmentation
===================================================================
A generic exercise-agnostic repetition segmentation engine driven by
per-exercise JSON configuration files.

Key Features:
1. Config-Driven: Works for squats, push-ups, or bicep curls without changing code.
2. Robust Boundary Detection: Pinpoints exact Start, Deepest Inflection, and
   Completion frames for each repetition.
3. Physics-Based Filtering: Rejects small fidgets and twitches using Range-of-Motion
   (ROM), minimum duration, and standing threshold validation.
4. Visual Diagnostics: Generates multi-rep trajectory plots with shaded repetition
   bands and labeled key moments.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config import EXERCISE_CONFIGS_DIR, OUTPUTS_DIR


class RepDetector:
    """
    Detects and segments individual repetitions from kinematic feature time-series
    based on exercise configuration parameters.
    """

    def __init__(self, config_path: Optional[Path] = None, exercise_name: str = "squat"):
        if config_path is None:
            config_path = EXERCISE_CONFIGS_DIR / f"{exercise_name.lower()}.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Exercise config file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.exercise_name = self.config.get("exercise_name", exercise_name)
        self.primary_metric = self.config.get("primary_metric", "primary_knee_smooth")
        self.inflection_type = self.config.get("inflection_type", "valley")
        
        rep_params = self.config.get("rep_detection", {})
        self.standing_threshold = rep_params.get("standing_threshold", 150.0)
        self.inflection_threshold = rep_params.get("inflection_threshold", 120.0)
        self.min_rom_degrees = rep_params.get("min_rom_degrees", 35.0)
        self.min_duration_sec = rep_params.get("min_duration_sec", 1.0)
        self.max_duration_sec = rep_params.get("max_duration_sec", 8.0)
        self.peak_prominence = rep_params.get("peak_prominence", 25.0)

    def detect_reps(
        self,
        features_df: pd.DataFrame,
        fps: float = 30.0,
    ) -> List[Dict]:
        """
        Segments all valid repetitions from the feature DataFrame.
        
        Returns:
            List of repetition dictionaries with start, inflection, and end metadata.
        """
        if features_df.empty or self.primary_metric not in features_df.columns:
            return []

        series = features_df[self.primary_metric].to_numpy()
        n_frames = len(series)
        min_distance = max(1, int(fps * self.min_duration_sec * 0.5))

        # For exercises like squats, inflection (bottom) is a valley (local minimum).
        # We invert the signal so find_peaks can locate valleys as peaks.
        if self.inflection_type == "valley":
            search_signal = -series
            prominence = self.peak_prominence
        else:
            search_signal = series
            prominence = self.peak_prominence

        candidate_inflections, properties = find_peaks(
            search_signal,
            prominence=prominence,
            distance=min_distance,
        )

        reps: List[Dict] = []
        last_end_frame = 0

        for inf_idx in candidate_inflections:
            inf_angle = series[inf_idx]

            # Validate that the inflection reached deep enough
            if self.inflection_type == "valley" and inf_angle > self.inflection_threshold:
                continue
            elif self.inflection_type == "peak" and inf_angle < self.inflection_threshold:
                continue

            # 1. Search Backwards for Rep Start (local plateau or standing threshold)
            start_idx = inf_idx
            search_back_limit = max(last_end_frame, 0)
            
            # Walk backwards until the angle stops increasing or reaches standing threshold
            for i in range(inf_idx - 1, search_back_limit - 1, -1):
                if series[i] >= self.standing_threshold:
                    start_idx = i
                    break
                if i > search_back_limit and series[i] < series[i + 1] and (inf_idx - i) > 5:
                    # Angle started dropping again; local peak reached
                    start_idx = i
                    break
                start_idx = i

            # Refine start to local peak between search_back_limit and inf_idx
            if inf_idx > start_idx:
                local_start = start_idx + np.argmax(series[start_idx:inf_idx])
                start_idx = local_start

            # 2. Search Forwards for Rep End (completion of ascent)
            end_idx = inf_idx
            for i in range(inf_idx + 1, n_frames):
                if series[i] >= self.standing_threshold:
                    end_idx = i
                    break
                if i < (n_frames - 1) and series[i] < series[i - 1] and (i - inf_idx) > 5:
                    # Angle started dropping again; new descent started
                    end_idx = i
                    break
                end_idx = i

            # Refine end to local peak between inf_idx and min(end_idx + 10, n_frames)
            forward_window = min(end_idx + 15, n_frames)
            if forward_window > inf_idx:
                local_end = inf_idx + np.argmax(series[inf_idx:forward_window])
                end_idx = max(end_idx, local_end)

            # 3. Validation Criteria
            start_angle = series[start_idx]
            end_angle = series[end_idx]
            peak_standing = max(start_angle, end_angle)
            rom = abs(peak_standing - inf_angle)
            duration_sec = (end_idx - start_idx) / fps

            # Filter out non-reps (small twitches or unrealistic durations)
            if rom < self.min_rom_degrees:
                continue
            if not (self.min_duration_sec <= duration_sec <= self.max_duration_sec):
                continue
            if start_idx >= inf_idx or inf_idx >= end_idx:
                continue

            # Measure peak torso lean during this rep (if available)
            peak_torso = 0.0
            if "torso_lean_smooth" in features_df.columns:
                rep_slice = features_df["torso_lean_smooth"].iloc[start_idx:end_idx + 1]
                peak_torso = round(float(rep_slice.max()), 1) if not rep_slice.empty else 0.0

            rep_data = {
                "rep_num": len(reps) + 1,
                "start_frame": int(start_idx),
                "inflection_frame": int(inf_idx),
                "end_frame": int(end_idx),
                "start_time_sec": round(float(start_idx / fps), 2),
                "inflection_time_sec": round(float(inf_idx / fps), 2),
                "end_time_sec": round(float(end_idx / fps), 2),
                "duration_sec": round(float(duration_sec), 2),
                "inflection_angle": round(float(inf_angle), 1),
                "start_angle": round(float(start_angle), 1),
                "end_angle": round(float(end_angle), 1),
                "rom_degrees": round(float(rom), 1),
                "peak_torso_lean": peak_torso,
            }
            reps.append(rep_data)
            last_end_frame = end_idx

        return reps

    def plot_detected_reps(
        self,
        features_df: pd.DataFrame,
        reps: List[Dict],
        video_name: str,
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Renders a diagnostic chart showing the tracked angle trajectory
        with labeled repetition intervals, start/inflection/end markers.
        """
        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = OUTPUTS_DIR / f"{Path(video_name).stem}_detected_reps.png"
        else:
            save_path = Path(save_path)
            
        save_path.parent.mkdir(parents=True, exist_ok=True)

        time_axis = features_df["timestamp_sec"].to_numpy()
        angle_series = features_df[self.primary_metric].to_numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")
        ax.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")
        ax.tick_params(colors="#E2E8F0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#475569")

        # Plot full smoothed trajectory
        ax.plot(
            time_axis, angle_series,
            color="#38BDF8", linewidth=2.5,
            label=f"{self.exercise_name.title()} Angle ({self.primary_metric})",
        )

        # Highlight threshold lines
        ax.axhline(
            self.standing_threshold, color="#A78BFA",
            linestyle=":", linewidth=1.5, label=f"Standing Threshold ({self.standing_threshold}°)",
        )
        ax.axhline(
            90.0, color="#4ADE80",
            linestyle="--", linewidth=1.5, label="Parallel Depth (90°)",
        )

        # Highlight each detected repetition
        colors = ["#3B82F6", "#8B5CF6", "#EC4899", "#10B981", "#F59E0B"]

        for rep in reps:
            color = colors[(rep["rep_num"] - 1) % len(colors)]
            t_start = rep["start_time_sec"]
            t_inf = rep["inflection_time_sec"]
            t_end = rep["end_time_sec"]
            ang_inf = rep["inflection_angle"]

            # Shaded interval
            ax.axvspan(t_start, t_end, color=color, alpha=0.15)

            # Markers: Start (green dot), Bottom (gold star), End (blue square)
            ax.scatter([t_start], [rep["start_angle"]], color="#22C55E", s=70, zorder=5)
            ax.scatter([t_inf], [ang_inf], color="#FACC15", marker="*", s=160, zorder=6)
            ax.scatter([t_end], [rep["end_angle"]], color="#60A5FA", marker="s", s=70, zorder=5)

            # Text annotation above bottom
            ax.annotate(
                f"Rep {rep['rep_num']}\n{ang_inf:.1f}° ({rep['duration_sec']}s)",
                xy=(t_inf, ang_inf),
                xytext=(t_inf, ang_inf - 15),
                fontsize=9, fontweight="bold", color="#F8FAFC",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#334155", ec=color, lw=1.5),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

        ax.set_title(
            f"Coach Lens - Automated Repetition Segmentation ({video_name})\nTotal Reps Detected: {len(reps)}",
            color="#F8FAFC", fontsize=13, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Time (seconds)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.set_ylabel("Angle (degrees)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="#F8FAFC", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        return save_path
