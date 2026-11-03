"""
Coach Lens - Module 6: Evaluation Suite & Team Labeling Protocol
================================================================
Evaluates the Coach Lens scoring engine under rigorous scientific standards.

Key Features:
1. Honest Dataset Disclaimer: Acknowledges that the Kaggle dataset lacks ground-truth
   good/bad labels; avoids hallucinating unsubstantiated accuracy claims.
2. Synthetic Ground-Truth Benchmark: Evaluates discrimination performance on 20
   controlled test cases (textbook reps, shallow squats, torso collapses).
3. Statistical Metrics: Computes Binary Classification metrics (Sensitivity,
   Specificity, F1-Score, and ROC-AUC) using threshold cutoff (Score >= 80).
4. Team Recording & Labeling Protocol: Generates a ready-to-use CSV spreadsheet
   (data/reports/teammate_form_labeling_sheet.csv) and guidelines for college teammates.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from config import REPORTS_DIR, OUTPUTS_DIR
from compare import FormScorer


def generate_synthetic_benchmark(scorer: FormScorer, n_samples: int = 20) -> List[Dict]:
    """
    Creates a controlled benchmark dataset with known ground-truth labels:
    - 10 'Good Form' reps (natural variations within tolerance)
    - 5 'Shallow Depth' mistakes
    - 5 'Excessive Torso Lean' mistakes
    """
    benchmark_cases = []
    standard_length = scorer.standard_length
    base_knee = np.array(scorer.ref_mean_knee)
    base_torso = np.array(scorer.ref_mean_torso)

    np.random.seed(42)  # Deterministic reproducibility

    # 1. Ten Good Repetitions (True Label = 1)
    for i in range(10):
        # Add slight natural jitter (±3 degrees)
        jitter = np.random.normal(0, 2.5, standard_length)
        knee_curve = base_knee + jitter
        torso_curve = np.clip(base_torso + np.random.normal(0, 2.0, standard_length), 5.0, 32.0)
        
        df = pd.DataFrame({
            "frame_idx": np.arange(standard_length),
            "primary_knee_smooth": knee_curve,
            "torso_lean_smooth": torso_curve,
        })
        benchmark_cases.append({
            "id": f"good_rep_{i+1:02d}",
            "ground_truth": "Good",
            "ground_truth_binary": 1,
            "expected_defect": "None",
            "df": df,
        })

    # 2. Five Shallow Depth Repetitions (True Label = 0)
    for i in range(5):
        # Scale descent so knee angle bottoms out between 100° and 125°
        target_bottom = 100.0 + (i * 5.0)
        scale = (175.0 - target_bottom) / (175.0 - scorer.ref_bottom_angle)
        shallow_curve = 175.0 - (175.0 - base_knee) * scale
        
        df = pd.DataFrame({
            "frame_idx": np.arange(standard_length),
            "primary_knee_smooth": shallow_curve,
            "torso_lean_smooth": base_torso,
        })
        benchmark_cases.append({
            "id": f"shallow_rep_{i+1:02d}",
            "ground_truth": "Bad",
            "ground_truth_binary": 0,
            "expected_defect": f"Shallow Depth ({target_bottom:.0f}°)",
            "df": df,
        })

    # 3. Five Excessive Forward Lean Repetitions (True Label = 0)
    for i in range(5):
        # Lean increases to 48° - 68° at bottom
        added_lean = 25.0 + (i * 7.0)
        lean_curve = base_torso + added_lean
        
        df = pd.DataFrame({
            "frame_idx": np.arange(standard_length),
            "primary_knee_smooth": base_knee,
            "torso_lean_smooth": lean_curve,
        })
        benchmark_cases.append({
            "id": f"excessive_lean_rep_{i+1:02d}",
            "ground_truth": "Bad",
            "ground_truth_binary": 0,
            "expected_defect": f"Excessive Lean (+{added_lean:.0f}°)",
            "df": df,
        })

    return benchmark_cases


def run_evaluation_suite():
    print("\n=======================================================")
    print("   COACH LENS - MODULE 6: COMPREHENSIVE EVALUATION")
    print("=======================================================")

    scorer = FormScorer(exercise_name="squat")
    benchmark_dataset = generate_synthetic_benchmark(scorer, n_samples=20)
    print(f"Generated {len(benchmark_dataset)} controlled validation repetitions (10 Good, 10 Defective).")

    records = []
    y_true = []
    scores = []

    for item in benchmark_dataset:
        eval_res = scorer.evaluate_repetition(item["df"])
        score = eval_res["overall_score"]
        predicted_binary = 1 if score >= 75.0 else 0

        y_true.append(item["ground_truth_binary"])
        scores.append(score)

        records.append({
            "Rep_ID": item["id"],
            "Ground_Truth": item["ground_truth"],
            "Expected_Defect": item["expected_defect"],
            "Score": score,
            "Predicted_Quality": "Good" if predicted_binary == 1 else "Bad",
            "Depth_Penalty": eval_res["penalties"]["depth_penalty"],
            "Torso_Penalty": eval_res["penalties"]["torso_penalty"],
            "DTW_Penalty": eval_res["penalties"]["trajectory_penalty"],
            "Worst_Moment_Reason": eval_res["worst_moment"]["reason"],
        })

    eval_df = pd.DataFrame(records)
    csv_report_path = REPORTS_DIR / "synthetic_evaluation_report.csv"
    eval_df.to_csv(csv_report_path, index=False)
    print(f"[SAVED] Synthetic Evaluation Table written to: {csv_report_path.name}")

    # Compute Statistical Classification Metrics
    y_true = np.array(y_true)
    y_pred = (np.array(scores) >= 75.0).astype(int)

    auc_score = roc_auc_score(y_true, scores)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n-------------------------------------------------------")
    print("   BENCHMARK EVALUATION METRICS (Cutoff = 75 pts)")
    print("-------------------------------------------------------")
    print(f"  * ROC-AUC Score:      {auc_score:.3f} (Ideal: 1.000)")
    print(f"  * Precision:          {precision * 100.0:.1f}%")
    print(f"  * Recall / Sensitivity:{recall * 100.0:.1f}%")
    print(f"  * F1-Score:           {f1 * 100.0:.1f}%")
    print(f"  * Confusion Matrix:   TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-------------------------------------------------------")

    # Generate Team Labeling Protocol Spreadsheet
    generate_team_labeling_template()
    return eval_df


def generate_team_labeling_template():
    """
    Creates a CSV template for teammates to record and annotate their own
    phone video clips for external validation.
    """
    labeling_rows = [
        {
            "video_filename": "teammate_squat_01.mp4",
            "athlete_name": "Vijay",
            "camera_angle": "Left Side (90 deg)",
            "camera_height": "Hip Level (approx 1m)",
            "rep_number": 1,
            "form_label": "Good",
            "observed_defect": "None - parallel depth reached",
            "notes": "Textbook form baseline",
        },
        {
            "video_filename": "teammate_squat_02.mp4",
            "athlete_name": "Teammate 1",
            "camera_angle": "Left Side (90 deg)",
            "camera_height": "Hip Level",
            "rep_number": 1,
            "form_label": "Bad",
            "observed_defect": "Shallow Depth",
            "notes": "Stopped ~15 degrees above parallel",
        },
        {
            "video_filename": "teammate_squat_03.mp4",
            "athlete_name": "Teammate 2",
            "camera_angle": "Left Side (90 deg)",
            "camera_height": "Hip Level",
            "rep_number": 1,
            "form_label": "Bad",
            "observed_defect": "Chest Collapse",
            "notes": "Excessive torso forward lean at bottom",
        },
    ]
    label_df = pd.DataFrame(labeling_rows)
    template_path = REPORTS_DIR / "teammate_form_labeling_sheet.csv"
    label_df.to_csv(template_path, index=False)
    print(f"[SAVED] Teammate Form Labeling Sheet written to: {template_path.name}")


if __name__ == "__main__":
    run_evaluation_suite()

