"""
Coach Lens - Module 5 Demo: Scoring & DTW Validation
=====================================================
Tests the FormScorer engine on:
1. Real repetitions from the dataset
2. Controlled synthetic test cases (Ideal Rep, Shallow Depth, Excessive Lean)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import DATASET_DIR, CACHE_DIR, OUTPUTS_DIR
from pose_extractor import PoseExtractor
from features import FeatureExtractor
from rep_detector import RepDetector
from compare import FormScorer


def run_module5_demo():
    print("\n=======================================================")
    print("   COACH LENS - MODULE 5: DTW SCORING VALIDATION")
    print("=======================================================")

    scorer = FormScorer(exercise_name="squat")
    print(f"Loaded Golden Template for [{scorer.exercise_name.upper()}]")
    print(f"Reference Baseline Depth: {scorer.ref_bottom_angle:.1f}°")

    # -------------------------------------------------------------
    # Test 1: Real Repetition from Dataset (squat_10.mp4, Rep 1)
    # -------------------------------------------------------------
    print("\n--- Test 1: Evaluating Real Repetition (squat_10.mp4) ---")
    feat_df = pd.read_csv(CACHE_DIR / "squat_10_features.csv")
    rep_detector = RepDetector(exercise_name="squat")
    reps = rep_detector.detect_reps(feat_df, fps=24.0)

    if reps:
        rep1_meta = reps[0]
        rep1_df = feat_df.iloc[rep1_meta["start_frame"] : rep1_meta["end_frame"] + 1].copy()
        eval1 = scorer.evaluate_repetition(rep1_df, rep_metadata=rep1_meta)
        plot_path = scorer.plot_comparison(eval1, save_path=OUTPUTS_DIR / "real_squat10_rep1_scored.png")

        print(f"  * Rep 1 Score:       {eval1['overall_score']} / 100 ({eval1['grade']})")
        print(f"  * Measured Knee Min: {eval1['gaps']['user_min_knee']}° (Target <= 90°)")
        print(f"  * Measured Torso:    {eval1['gaps']['user_peak_torso']}° (Target <= 35°)")
        print(f"  * Penalties:         Depth: -{eval1['penalties']['depth_penalty']}, Torso: -{eval1['penalties']['torso_penalty']}, DTW: -{eval1['penalties']['trajectory_penalty']}")
        print(f"  * Saved Plot:        {plot_path.name}")

    # -------------------------------------------------------------
    # Test 2: Synthetic Controlled Test Cases
    # -------------------------------------------------------------
    print("\n--- Test 2: Sanity Checks on Synthetic Form Distortions ---")
    time_pts = 100

    # Case A: Textbook Ideal Rep (Matches reference exactly)
    synth_good = pd.DataFrame({
        "frame_idx": np.arange(time_pts),
        "primary_knee_smooth": np.array(scorer.ref_mean_knee),
        "torso_lean_smooth": np.array(scorer.ref_mean_torso),
    })
    res_good = scorer.evaluate_repetition(synth_good)
    print(f"  [A] Ideal Textbook Rep: Score = {res_good['overall_score']}/100 ({res_good['grade']}) [EXPECTED: ~100]")

    # Case B: Shallow Depth Mistake (Only descends to 112°, stops way above parallel)
    shallow_knee = np.array(scorer.ref_mean_knee)
    # Lift the bottom from ~30° up to 112°
    shallow_knee = 175.0 - (175.0 - shallow_knee) * 0.45
    synth_shallow = pd.DataFrame({
        "frame_idx": np.arange(time_pts),
        "primary_knee_smooth": shallow_knee,
        "torso_lean_smooth": np.array(scorer.ref_mean_torso),
    })
    res_shallow = scorer.evaluate_repetition(synth_shallow)
    print(f"  [B] Shallow Squat (112°): Score = {res_shallow['overall_score']}/100 ({res_shallow['grade']})")
    print(f"      Depth Penalty: -{res_shallow['penalties']['depth_penalty']} pts | Worst Reason: {res_shallow['worst_moment']['reason']}")

    # Case C: Excessive Torso Forward Lean (Chest drops to 58° at bottom)
    bad_lean = np.array(scorer.ref_mean_torso) + 38.0  # Excessive lean
    synth_lean = pd.DataFrame({
        "frame_idx": np.arange(time_pts),
        "primary_knee_smooth": np.array(scorer.ref_mean_knee),
        "torso_lean_smooth": bad_lean,
    })
    res_lean = scorer.evaluate_repetition(synth_lean)
    print(f"  [C] Chest Collapse (58° Lean): Score = {res_lean['overall_score']}/100 ({res_lean['grade']})")
    print(f"      Torso Penalty: -{res_lean['penalties']['torso_penalty']} pts | Worst Reason: {res_lean['worst_moment']['reason']}")

    print("\n=======================================================")
    print("   MODULE 5 DTW SCORING TEST COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    run_module5_demo()

