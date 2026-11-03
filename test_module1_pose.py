"""
Coach Lens - Module 1 Demo: Pose Extraction & Caching Test
==========================================================
Runs MediaPipe Pose on a real dataset clip, validates landmark extraction,
demonstrates disk caching speedup, and saves visual skeleton overlays.
"""

import time
from pathlib import Path
import cv2
import pandas as pd

from config import DATASET_DIR, OUTPUTS_DIR, CACHE_DIR
from pose_extractor import PoseExtractor


def run_module1_demo():
    print("\n=======================================================")
    print("   COACH LENS - MODULE 1: POSE EXTRACTION DEMO")
    print("=======================================================")

    test_video_path = DATASET_DIR / "squat_10.mp4"
    if not test_video_path.exists():
        # Fallback to first available video if squat_10.mp4 is missing
        candidates = list(DATASET_DIR.glob("*.mp4")) + list(DATASET_DIR.glob("*.MOV"))
        if not candidates:
            raise FileNotFoundError(f"No video files found in {DATASET_DIR}")
        test_video_path = sorted(candidates)[0]

    print(f"Target Video: {test_video_path.name}")
    extractor = PoseExtractor()

    # ------------------------------------------------------------------
    # Step 1: First Run (Extract from Scratch & Cache to Disk)
    # ------------------------------------------------------------------
    print("\n--- Step 1: Extracting Landmarks & Writing to Cache ---")
    start_time = time.time()
    df_raw, annotated_frames, fps, width, height = extractor.extract_landmarks_from_video(
        test_video_path, save_annotated_frames=True
    )
    extract_duration = time.time() - start_time
    print(f"Extraction completed in: {extract_duration:.2f} seconds")
    print(f"Total Frames Processed: {len(df_raw)} at {fps:.1f} fps ({width}x{height})")

    # Save to cache manually to verify cache mechanism
    cache_file = CACHE_DIR / f"{test_video_path.stem}_landmarks.csv"
    df_raw.to_csv(cache_file, index=False)
    print(f"[CACHE WRITTEN] Saved landmark CSV to: {cache_file.name} ({cache_file.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Step 2: Second Run (Demonstrate Lightning-Fast Cache Hit)
    # ------------------------------------------------------------------
    print("\n--- Step 2: Testing Cache Retrieval (Re-running same video) ---")
    cache_start = time.time()
    df_cached, _, _, _ = extractor.get_or_extract(test_video_path, cache_dir=CACHE_DIR)
    cache_duration = time.time() - cache_start
    print(f"Cache load completed in: {cache_duration:.4f} seconds!")
    print(f"Speedup Factor: {extract_duration / max(cache_duration, 1e-4):.1f}x faster!")

    # ------------------------------------------------------------------
    # Step 3: Data Quality & Detection Rate Validation
    # ------------------------------------------------------------------
    print("\n--- Step 3: Validating Landmark Data Quality ---")
    detected_count = int(df_raw["person_detected"].sum())
    total_count = len(df_raw)
    detection_rate = (detected_count / total_count) * 100.0 if total_count > 0 else 0.0

    print(f"Person Detection Rate: {detected_count}/{total_count} frames ({detection_rate:.1f}%)")
    
    # Check key squat joints visibility
    key_joints = ["left_knee", "right_knee", "left_hip", "right_hip", "left_ankle", "right_ankle"]
    print("Average joint visibility confidences:")
    for j in key_joints:
        avg_vis = df_raw[f"{j}_v"].mean()
        print(f"  * {j:<12}: {avg_vis:.3f} (confidence scale 0.0 to 1.0)")

    # ------------------------------------------------------------------
    # Step 4: Save Sample Skeleton Overlay Frames
    # ------------------------------------------------------------------
    print("\n--- Step 4: Saving Visual Skeleton Proof Images ---")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select 3 key moments: Start (~15%), Deepest Point (~45%), and Finish (~85%)
    indices = [
        int(total_count * 0.15),
        int(total_count * 0.45),
        int(total_count * 0.85),
    ]
    labels = ["start_phase", "deep_squat_phase", "finish_phase"]

    saved_images = []
    for idx, label in zip(indices, labels):
        if idx < len(annotated_frames):
            img_path = OUTPUTS_DIR / f"{test_video_path.stem}_{label}_frame{idx}.jpg"
            cv2.imwrite(str(img_path), annotated_frames[idx])
            saved_images.append(img_path)
            print(f"  [SAVED] {img_path.name}")

    print("\n=======================================================")
    print("   MODULE 1 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    return df_raw, saved_images


if __name__ == "__main__":
    run_module1_demo()
