"""
Coach Lens - Module 3 Demo: Repetition Detection on Multiple Real Videos
========================================================================
Runs the config-driven RepDetector across 3 real squat video clips,
reports per-rep segmentation boundaries, and generates annotated plots.
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd

from config import DATASET_DIR, CACHE_DIR, OUTPUTS_DIR
from pose_extractor import PoseExtractor
from features import FeatureExtractor
from rep_detector import RepDetector


def run_module3_demo():
    print("\n=======================================================")
    print("   COACH LENS - MODULE 3: REPETITION DETECTION DEMO")
    print("=======================================================")

    test_video_names = ["squat_10.mp4", "squat_13.mp4", "squat_15.mp4"]

    extractor = PoseExtractor()
    feat_extractor = FeatureExtractor(smoothing_window=9, polyorder=2)
    rep_detector = RepDetector(exercise_name="squat")

    results_summary: List[Dict] = []

    for idx, v_name in enumerate(test_video_names, 1):
        video_path = DATASET_DIR / v_name
        if not video_path.exists():
            print(f"[SKIP] Video not found: {video_path}")
            continue

        print(f"\n[{idx}/3] Processing Video: {v_name}")
        
        # 1. Pose landmarks (cached or extracted)
        landmarks_df, fps, width, height = extractor.get_or_extract(
            video_path, cache_dir=CACHE_DIR
        )
        duration_total = round(len(landmarks_df) / fps, 2)
        print(f"  * Total Duration: {duration_total}s ({len(landmarks_df)} frames at {fps:.1f} fps)")

        # 2. Features
        features_df = feat_extractor.extract_features(landmarks_df, fps=fps)
        dominant_side = features_df["dominant_side"].iloc[0]
        print(f"  * Dominant Side Tracked: [{dominant_side.upper()}]")

        # 3. Detect Repetitions
        reps = rep_detector.detect_reps(features_df, fps=fps)
        print(f"  * Detected Reps Count: {len(reps)}")

        for r in reps:
            print(
                f"    -> Rep {r['rep_num']}: {r['start_time_sec']}s -> "
                f"bottom at {r['inflection_time_sec']}s ({r['inflection_angle']}°) -> "
                f"finish at {r['end_time_sec']}s (dur: {r['duration_sec']}s, ROM: {r['rom_degrees']}°, torso: {r['peak_torso_lean']}°)"
            )

        # 4. Save Visualization Plot
        plot_path = OUTPUTS_DIR / f"{Path(v_name).stem}_detected_reps.png"
        rep_detector.plot_detected_reps(features_df, reps, v_name, save_path=plot_path)
        print(f"  * Saved Rep Plot: {plot_path.name}")

        results_summary.append({
            "video_name": v_name,
            "duration_sec": duration_total,
            "total_frames": len(landmarks_df),
            "fps": fps,
            "reps_detected": len(reps),
            "reps_data": reps,
            "plot_file": str(plot_path.name),
        })

    print("\n=======================================================")
    print("   AUTOMATED REPETITION DETECTION SUMMARY TABLE")
    print("=======================================================")
    for item in results_summary:
        print(f"Video: {item['video_name']:<14} | Duration: {item['duration_sec']:<6}s | Algorithmic Reps: {item['reps_detected']}")

    return results_summary


if __name__ == "__main__":
    run_module3_demo()
