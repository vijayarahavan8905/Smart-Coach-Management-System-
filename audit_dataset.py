"""
Coach Lens - Module 0: Dataset Audit Script
===========================================
This script inspects the gym exercise dataset, validates every video file,
extracts core video metadata (resolution, fps, duration, codecs), checks
for corrupt or unreadable files, and exports a detailed CSV audit report.

It supports both:
1. Subfolder-based datasets (e.g., dataset/squat/clip1.mp4)
2. Flat filename-prefixed datasets (e.g., dataset/squat_1.MOV)
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import pandas as pd

# Import project configuration
from config import (
    DATASET_DIR,
    REPORTS_DIR,
    SUPPORTED_VIDEO_EXTENSIONS,
    MIN_RECOMMENDED_CLIPS,
)


def infer_exercise_name(file_path: Path, dataset_root: Path) -> str:
    """
    Determines the exercise name from the folder hierarchy or filename.
    
    If the video is inside a subdirectory under the dataset root:
        e.g., 'dataset/squat/video1.mp4' -> exercise is 'squat'
    If all videos are in the root directory:
        e.g., 'dataset/squat_1.MOV' -> exercise is 'squat'
    """
    relative_parent = file_path.parent.relative_to(dataset_root)
    if str(relative_parent) != ".":
        # First-level folder under dataset root
        return relative_parent.parts[0].lower().strip()
    
    # Flat directory: extract word before number or underscore
    # e.g., 'squat_1.MOV' -> 'squat'
    stem = file_path.stem.lower()
    match = re.match(r"^([a-zA-Z\s\-]+)", stem)
    if match:
        name = match.group(1).rstrip("_ -")
        if name:
            return name
    return "unknown"


def audit_single_video(file_path: Path, dataset_root: Path) -> Dict:
    """
    Opens a single video file with OpenCV, inspects its metadata,
    and tests if the first frame can be successfully decoded.
    """
    exercise = infer_exercise_name(file_path, dataset_root)
    file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
    
    record = {
        "file_name": file_path.name,
        "exercise": exercise,
        "file_path": str(file_path),
        "extension": file_path.suffix.lower(),
        "file_size_mb": file_size_mb,
        "is_readable": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_sec": 0.0,
        "error_message": "",
    }
    
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        record["error_message"] = "OpenCV failed to open video file"
        return record
    
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Test decoding the first frame to guarantee codec compatibility
        ret, frame = cap.read()
        if not ret or frame is None:
            record["error_message"] = "Cannot decode first frame (possible codec issue)"
            return record
        
        duration = round(frame_count / fps, 2) if fps > 0 else 0.0
        
        record.update({
            "is_readable": True,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_sec": duration,
            "error_message": "None",
        })
    except Exception as exc:
        record["error_message"] = str(exc)
    finally:
        cap.release()
        
    return record


def run_dataset_audit(dataset_dir: Path, output_csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scans the dataset directory, collects metadata for all supported video files,
    generates summary statistics per exercise, and writes the CSV report.
    """
    print(f"\n=======================================================")
    print(f"   COACH LENS - MODULE 0: DATASET AUDIT")
    print(f"=======================================================")
    print(f"Target dataset directory: {dataset_dir}")
    
    if not dataset_dir.exists():
        print(f"\n[ERROR] Dataset directory not found at: {dataset_dir}")
        print("Please verify the path in config.py.")
        sys.exit(1)
        
    # Discover all video files recursively
    video_files = [
        f for f in dataset_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    
    total_found = len(video_files)
    print(f"Total video files detected: {total_found}")
    
    if total_found == 0:
        print("[WARNING] No video files found matching supported extensions.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Audit each video
    records: List[Dict] = []
    for idx, video_path in enumerate(video_files, 1):
        rec = audit_single_video(video_path, dataset_dir)
        records.append(rec)
        status_flag = "[OK]" if rec["is_readable"] else "[FAIL]"
        print(f"  [{idx:02d}/{total_found:02d}] {status_flag} {rec['file_name']:<18} "
              f"| {rec['exercise']:<10} | {rec['width']}x{rec['height']} "
              f"| {rec['fps']} fps | {rec['duration_sec']}s")
        
    df = pd.DataFrame(records)
    
    # Ensure reports directory exists and save CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\n[SAVED] Detailed audit report written to: {output_csv_path}")
    
    # Generate aggregated summary table per exercise
    summary_rows = []
    for exercise_name, group in df.groupby("exercise"):
        total_clips = len(group)
        readable_clips = int(group["is_readable"].sum())
        unreadable_clips = total_clips - readable_clips
        
        readable_group = group[group["is_readable"]]
        if not readable_group.empty:
            avg_duration = round(readable_group["duration_sec"].mean(), 2)
            min_duration = round(readable_group["duration_sec"].min(), 2)
            max_duration = round(readable_group["duration_sec"].max(), 2)
            resolutions = ", ".join(
                f"{w}x{h} ({cnt})" 
                for (w, h), cnt in readable_group.groupby(["width", "height"]).size().items()
            )
            fps_values = ", ".join(
                f"{fps}fps ({cnt})" 
                for fps, cnt in readable_group.groupby("fps").size().items()
            )
        else:
            avg_duration = min_duration = max_duration = 0.0
            resolutions = "N/A"
            fps_values = "N/A"
            
        status = "Sufficient" if readable_clips >= MIN_RECOMMENDED_CLIPS else "Insufficient (<10)"
        
        summary_rows.append({
            "Exercise": exercise_name,
            "Total Clips": total_clips,
            "Readable": readable_clips,
            "Unreadable": unreadable_clips,
            "Avg Duration (s)": avg_duration,
            "Duration Range (s)": f"{min_duration}s - {max_duration}s",
            "Resolutions": resolutions,
            "Frame Rates": fps_values,
            "Sample Sufficiency": status,
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_csv_path.parent / "dataset_summary_report.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"\n=======================================================")
    print(f"   DATASET AUDIT SUMMARY")
    print(f"=======================================================")
    for _, row in summary_df.iterrows():
        print(f"Exercise:           {row['Exercise'].upper()}")
        print(f"  Total Clips:      {row['Total Clips']} (Readable: {row['Readable']}, Unreadable: {row['Unreadable']})")
        print(f"  Duration:         Average {row['Avg Duration (s)']}s ({row['Duration Range (s)']})")
        print(f"  Resolutions:      {row['Resolutions']}")
        print(f"  Frame Rates:      {row['Frame Rates']}")
        print(f"  Status:           {row['Sample Sufficiency']} for template building")
        print(f"-------------------------------------------------------")
        
    return df, summary_df


if __name__ == "__main__":
    report_file = REPORTS_DIR / "dataset_audit_report.csv"
    run_dataset_audit(DATASET_DIR, report_file)
