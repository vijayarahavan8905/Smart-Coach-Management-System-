"""
Coach Lens - Module 4: Reference Template Builder
=================================================
Constructs high-quality reference form templates ("Golden Rulers") for exercises
by aggregating and time-normalizing clean repetitions extracted from good video clips.

Key Concepts:
1. Time Normalization: Resamples repetitions of varying durations (e.g. 1.2s vs 3.5s)
   to a standardized 100-step time grid (0% start -> 50% bottom -> 100% finish).
2. Statistical Modeling: Computes point-wise Mean (ideal trajectory) and Standard
   Deviation (acceptable biological buffer zone).
3. Multi-Metric Tracking: Stores templates for primary angle (knee), secondary
   angle (hip), and torso lean.
4. JSON Export: Writes lightweight, portable templates to data/templates/ for
   instant client-side evaluation without needing heavy video files.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

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


def resample_repetition(
    signal: np.ndarray,
    target_length: int = 100,
) -> np.ndarray:
    """
    Resamples a 1D time-series slice onto a uniform grid of `target_length` points.
    
    Uses linear interpolation across normalized time tau in [0.0, 1.0].
    """
    n_orig = len(signal)
    if n_orig < 2:
        return np.full(target_length, signal[0] if n_orig == 1 else 0.0)

    orig_tau = np.linspace(0.0, 1.0, num=n_orig)
    target_tau = np.linspace(0.0, 1.0, num=target_length)
    
    interpolator = interp1d(orig_tau, signal, kind="linear", bounds_error=False, fill_value="extrapolate")
    return interpolator(target_tau)


class TemplateBuilder:
    """
    Builds and manages biomechanical reference templates for gym exercises.
    """

    def __init__(self, exercise_name: str = "squat"):
        self.exercise_name = exercise_name.lower()
        self.config_path = EXERCISE_CONFIGS_DIR / f"{self.exercise_name}.json"
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.primary_metric = self.config.get("primary_metric", "primary_knee_smooth")
        self.tracked_metrics = [
            self.primary_metric,
            "primary_hip_smooth",
            "torso_lean_smooth",
        ]
        
        self.output_template_path = TEMPLATES_DIR / f"{self.exercise_name}_template.json"
        self.pose_extractor = PoseExtractor()
        self.feature_extractor = FeatureExtractor()
        self.rep_detector = RepDetector(config_path=self.config_path)

    def extract_reps_from_video(
        self,
        video_path: Path,
    ) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """
        Extracts all valid repetitions from a single video clip and slices
        their feature DataFrames.
        """
        landmarks_df, fps, width, height = self.pose_extractor.get_or_extract(
            video_path, cache_dir=CACHE_DIR
        )
        if landmarks_df.empty:
            return [], []

        features_df = self.feature_extractor.extract_features(landmarks_df, fps=fps)
        reps_metadata = self.rep_detector.detect_reps(features_df, fps=fps)

        rep_slices = []
        for rep in reps_metadata:
            start_f = rep["start_frame"]
            end_f = rep["end_frame"]
            rep_df = features_df.iloc[start_f : end_f + 1].copy().reset_index(drop=True)
            rep_slices.append(rep_df)

        return rep_slices, reps_metadata

    def build_template_from_videos(
        self,
        reference_video_names: List[str],
        standard_length: int = 100,
    ) -> Dict:
        """
        Processes a curated list of good-form video clips, extracts all clean reps,
        time-normalizes each rep to `standard_length`, and computes mean & std curves.
        """
        print(f"\n=======================================================")
        print(f"   BUILDING REFERENCE TEMPLATE: {self.exercise_name.upper()}")
        print(f"=======================================================")
        print(f"Reference videos pool: {reference_video_names}")

        collected_reps: Dict[str, List[np.ndarray]] = {m: [] for m in self.tracked_metrics}
        reps_log = []

        for v_name in reference_video_names:
            v_path = DATASET_DIR / v_name
            if not v_path.exists():
                print(f"[SKIP] Video not found: {v_name}")
                continue

            print(f"  * Analyzing reference candidate: {v_name} ...")
            rep_slices, metadata = self.extract_reps_from_video(v_path)
            print(f"    -> Found {len(rep_slices)} candidate reps.")

            for r_idx, (rep_df, meta) in enumerate(zip(rep_slices, metadata)):
                # Form quality filter: must reach legitimate depth and moderate torso lean
                min_knee = meta["inflection_angle"]
                peak_torso = meta["peak_torso_lean"]

                # Ensure only good/valid reps contribute to the reference template
                if min_knee > 105.0:  # Too shallow to be a golden example
                    print(f"       [REJECTED] Rep {meta['rep_num']} depth was shallow ({min_knee:.1f}°)")
                    continue
                if peak_torso > 50.0:  # Excessive torso lean
                    print(f"       [REJECTED] Rep {meta['rep_num']} torso lean excessive ({peak_torso:.1f}°)")
                    continue

                reps_log.append({
                    "video": v_name,
                    "rep_num": meta["rep_num"],
                    "duration_sec": meta["duration_sec"],
                    "inflection_angle": min_knee,
                    "peak_torso": peak_torso,
                })

                # Resample each tracked metric to standard_length
                for metric in self.tracked_metrics:
                    if metric in rep_df.columns:
                        resampled = resample_repetition(rep_df[metric].to_numpy(), standard_length)
                        collected_reps[metric].append(resampled)

        num_valid_reps = len(reps_log)
        print(f"\nTotal Clean Reps Accepted for Reference Template: {num_valid_reps}")
        if num_valid_reps < 2:
            raise ValueError(
                f"Insufficient clean reps ({num_valid_reps}) found in selected videos. "
                "Provide videos with clear, full range of motion."
            )

        # Compute point-wise Mean and Standard Deviation for each metric
        template_metrics = {}
        for metric in self.tracked_metrics:
            matrix = np.array(collected_reps[metric])  # Shape: (num_reps, standard_length)
            mean_vals = np.mean(matrix, axis=0)
            std_vals = np.std(matrix, axis=0)

            # Enforce a minimum realistic biological tolerance band of at least 5 degrees
            std_vals = np.maximum(std_vals, 5.0)

            template_metrics[metric] = {
                "mean": [round(float(x), 2) for x in mean_vals],
                "std": [round(float(x), 2) for x in std_vals],
            }

        # Build final serializable template payload
        template_payload = {
            "exercise_name": self.exercise_name,
            "version": "1.0",
            "standard_length": standard_length,
            "primary_metric": self.primary_metric,
            "num_reference_reps_used": num_valid_reps,
            "reference_reps_metadata": reps_log,
            "metrics": template_metrics,
            "inflection_index": standard_length // 2,
            "tolerances": self.config.get("form_metrics", {}),
            "coaching_tips": self.config.get("coaching_tips", {}),
        }

        # Save to templates directory
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.output_template_path, "w", encoding="utf-8") as f:
            json.dump(template_payload, f, indent=2)

        print(f"[SAVED TEMPLATE] Successfully exported template to: {self.output_template_path}")
        return template_payload

    def plot_template(self, template_data: Optional[Dict] = None, save_path: Optional[Path] = None) -> Path:
        """
        Renders a diagnostic chart showing the ideal mean trajectory with
        its shaded standard deviation tolerance corridor.
        """
        import matplotlib.pyplot as plt

        if template_data is None:
            if not self.output_template_path.exists():
                raise FileNotFoundError(f"Template file not found at {self.output_template_path}")
            with open(self.output_template_path, "r", encoding="utf-8") as f:
                template_data = json.load(f)

        if save_path is None:
            save_path = OUTPUTS_DIR / f"{self.exercise_name}_reference_template.png"
        else:
            save_path = Path(save_path)
            
        save_path.parent.mkdir(parents=True, exist_ok=True)

        norm_time = np.linspace(0, 100, template_data["standard_length"])
        metrics_dict = template_data["metrics"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        fig.patch.set_facecolor("#0F172A")

        for ax in (ax1, ax2):
            ax.set_facecolor("#1E293B")
            ax.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")
            ax.tick_params(colors="#E2E8F0", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#475569")

        # Top Panel: Knee Angle Corridor
        knee_data = metrics_dict[self.primary_metric]
        knee_mean = np.array(knee_data["mean"])
        knee_std = np.array(knee_data["std"])

        ax1.plot(norm_time, knee_mean, color="#38BDF8", linewidth=3, label="Golden Reference Mean (Knee)")
        ax1.fill_between(
            norm_time,
            knee_mean - knee_std,
            knee_mean + knee_std,
            color="#38BDF8",
            alpha=0.25,
            label="Acceptable Form Corridor (±1 Std Dev)",
        )
        ax1.axhline(90.0, color="#4ADE80", linestyle=":", linewidth=1.5, label="Parallel Depth (90°)")
        ax1.set_ylabel("Knee Angle (deg)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"Coach Lens - Biomechanical Golden Reference Template ({self.exercise_name.upper()})\n"
            f"Built from {template_data['num_reference_reps_used']} textbook repetitions across dataset",
            color="#F8FAFC", fontsize=13, fontweight="bold", pad=12,
        )
        ax1.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="#F8FAFC", fontsize=9)

        # Bottom Panel: Torso Forward Lean Corridor
        torso_data = metrics_dict["torso_lean_smooth"]
        torso_mean = np.array(torso_data["mean"])
        torso_std = np.array(torso_data["std"])

        ax2.plot(norm_time, torso_mean, color="#FB923C", linewidth=3, label="Ideal Torso Lean Trajectory")
        ax2.fill_between(
            norm_time,
            np.maximum(0, torso_mean - torso_std),
            torso_mean + torso_std,
            color="#FB923C",
            alpha=0.25,
            label="Acceptable Torso Buffer",
        )
        ax2.axhline(45.0, color="#EF4444", linestyle="--", linewidth=1.5, label="Excessive Lean Ceiling (45°)")
        ax2.set_xlabel("Normalized Repetition Progress (% from Start to Finish)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Torso Lean (deg)", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="#F8FAFC", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SAVED PLOT] Reference template visualization saved to: {save_path.name}")
        return save_path


if __name__ == "__main__":
    builder = TemplateBuilder(exercise_name="squat")
    
    # We select 4 clean clips from our audit that perform full-depth squats
    training_videos = ["squat_10.mp4", "squat_13.mp4", "squat_14.mp4", "squat_15.mp4"]
    template = builder.build_template_from_videos(training_videos)
    builder.plot_template(template)
