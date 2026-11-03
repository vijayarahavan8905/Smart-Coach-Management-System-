"""
Coach Lens - End-to-End Command-Line Demo
==========================================
Runs the complete Coach Lens computer-vision & biomechanical pipeline on a real
exercise video and outputs rep-by-rep telemetry, scores, and coaching tips.

Usage:
    python demo.py
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import DATASET_DIR, CACHE_DIR, OUTPUTS_DIR
from pose_extractor import PoseExtractor
from features import FeatureExtractor
from rep_detector import RepDetector
from compare import FormScorer
from coach import CoachingAdvisor


def run_full_demo(video_name: str = "squat_10.mp4"):
    print("\n" + "=" * 65)
    print("      COACH LENS: END-TO-END PIPELINE DEMONSTRATION")
    print("=" * 65)

    video_path = DATASET_DIR / video_name
    # Check cache first or check video existence safely
    cache_file = CACHE_DIR / f"{Path(video_name).stem}_landmarks.csv"
    if not cache_file.exists():
        try:
            if not video_path.exists():
                print(f"[ERROR] Video not found at {video_path}")
                sys.exit(1)
        except Exception:
            pass

    print(f"Target Video: {video_path.name}")

    # Step 1: Pose Extraction & Caching
    print("\n[Step 1/5] Extracting 33 Body Landmarks via MediaPipe Pose...")
    extractor = PoseExtractor()
    landmarks_df, fps, width, height = extractor.get_or_extract(video_path, cache_dir=CACHE_DIR)
    total_sec = round(len(landmarks_df) / fps, 2)
    print(f"  -> Extracted {len(landmarks_df)} frames at {fps:.1f} fps (Duration: {total_sec}s)")

    # Step 2: Feature Engineering & Joint Angles
    print("\n[Step 2/5] Computing Biomechanical Joint Angles & Side Visibility...")
    feat_extractor = FeatureExtractor(smoothing_window=9, polyorder=2)
    features_df = feat_extractor.extract_features(landmarks_df, fps=fps)
    dom_side = features_df["dominant_side"].iloc[0]
    print(f"  -> Dominant Camera-Facing Side: [{dom_side.upper()}]")
    print(f"  -> Knee Angle Range: {features_df['primary_knee_smooth'].min():.1f} deg to {features_df['primary_knee_smooth'].max():.1f} deg")
    print(f"  -> Torso Lean Peak:  {features_df['torso_lean_smooth'].max():.1f} deg from vertical")

    # Step 3: Repetition Detection
    print("\n[Step 3/5] Segmenting Individual Repetitions...")
    rep_detector = RepDetector(exercise_name="squat")
    reps = rep_detector.detect_reps(features_df, fps=fps)
    print(f"  -> Successfully Detected: {len(reps)} complete repetition(s)")

    # Step 4: DTW Alignment & Form Scoring
    print("\n[Step 4/5] Evaluating Form against Golden Reference Template (FastDTW)...")
    scorer = FormScorer(exercise_name="squat")
    coach = CoachingAdvisor()

    eval_results = []
    for r in reps:
        rep_slice = features_df.iloc[r["start_frame"] : r["end_frame"] + 1].copy()
        eval_res = scorer.evaluate_repetition(rep_slice, rep_metadata=r)
        eval_results.append(eval_res)

        print(f"\n  [+] Repetition #{r['rep_num']} Telemetry:")
        print(f"     * Duration:           {r['duration_sec']}s ({r['start_time_sec']}s -> {r['end_time_sec']}s)")
        print(f"     * Lowest Knee Angle:  {eval_res['gaps']['user_min_knee']} deg (Target: <= 90 deg parallel)")
        print(f"     * Peak Torso Lean:    {eval_res['gaps']['user_peak_torso']} deg (Target: <= 35 deg)")
        print(f"     * DTW Warp Distance:  {eval_res['gaps']['dtw_distance']}")
        print(f"     * Form Score:         {eval_res['overall_score']} / 100 [{eval_res['grade']}]")
        print(f"     * Worst Identified:   {eval_res['worst_moment']['reason']}")

        # Step 5: Coach Feedback
        tips = coach.get_coaching_feedback(eval_res)
        print(f"     * Coaching Advice:")
        for tip in tips["rule_tips"]:
            print(f"       -> {tip}")

    # Generate visual summary plot
    if eval_results:
        summary_plot = scorer.plot_comparison(eval_results[0], save_path=OUTPUTS_DIR / f"{video_path.stem}_demo_scored.png")
        print(f"\n[Saved Plot] Visual telemetry exported to: {summary_plot.name}")

    print("\n" + "=" * 65)
    print("      END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_full_demo()
