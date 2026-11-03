"""
Coach Lens - Module 1: Pose Extraction & Landmark Caching
=========================================================
This module extracts 3D human pose landmarks from video frames using Google
MediaPipe Pose. It converts video files into structured DataFrames containing
normalized (0-1) coordinates, pixel coordinates, and visibility confidences.

Key Features:
1. Disk Caching: Saves extracted landmark data to CSV in data/cache/ to avoid
   re-running slow ML inferences on every execution.
2. Graceful Detection Handling: Robustly records missing person detections as NaN
   without crashing.
3. Visual Overlay: Renders clean, color-coded skeleton connections and joints on
   video frames and saves debug snapshots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from config import (
    PROJECT_ROOT,
    CACHE_DIR,
    OUTPUTS_DIR,
)

# MediaPipe 33 Landmark Names mapping (standard BlazePose topology)
ALL_LANDMARKS = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Primary skeletal connections for gym exercise tracking & visualization
SKELETON_CONNECTIONS = [
    # Head & Shoulders
    (0, 11), (0, 12), (11, 12),
    # Torso (Shoulders to Hips)
    (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15),
    # Right Arm
    (12, 14), (14, 16),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (30, 32),
]


class PoseExtractor:
    """
    Extracts 33 body landmarks per video frame using MediaPipe PoseLandmarker
    and manages disk caching for lightning-fast subsequent runs.
    """

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            self.model_path = PROJECT_ROOT / "pose_landmarker_full.task"
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Pose model not found at {self.model_path}. "
                "Please place pose_landmarker_full.task in the project root."
            )

        # Import modern MediaPipe Tasks Vision API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        self._options = vision.PoseLandmarkerOptions(
            base_options=self._base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

    def extract_landmarks_from_video(
        self,
        video_path: Path,
        save_annotated_frames: bool = False,
    ) -> Tuple[pd.DataFrame, List[np.ndarray], float, int, int]:
        """
        Processes a video file frame-by-frame and extracts all 33 pose landmarks.

        Returns:
            df: DataFrame with normalized coords, pixel coords, and visibility scores.
            annotated_frames: List of OpenCV frames with skeleton drawn (if requested).
            fps: Video frames per second.
            width: Frame pixel width.
            height: Frame pixel height.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Fallback to standard frame rate
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        from mediapipe.tasks.python import vision

        records: List[Dict] = []
        annotated_frames: List[np.ndarray] = []
        frame_idx = 0

        with vision.PoseLandmarker.create_from_options(self._options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp_ms = int(frame_idx * (1000.0 / fps))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                row = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "person_detected": False,
                }

                has_person = (
                    detection_result.pose_landmarks is not None
                    and len(detection_result.pose_landmarks) > 0
                )

                if has_person:
                    row["person_detected"] = True
                    landmarks = detection_result.pose_landmarks[0]

                    for idx, name in enumerate(ALL_LANDMARKS):
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            v = float(getattr(lm, "visibility", 1.0))
                            row[f"{name}_x"] = float(lm.x)
                            row[f"{name}_y"] = float(lm.y)
                            row[f"{name}_z"] = float(lm.z)
                            row[f"{name}_v"] = v
                            row[f"{name}_px"] = round(float(lm.x * width), 1)
                            row[f"{name}_py"] = round(float(lm.y * height), 1)
                        else:
                            row[f"{name}_x"] = np.nan
                            row[f"{name}_y"] = np.nan
                            row[f"{name}_z"] = np.nan
                            row[f"{name}_v"] = 0.0
                            row[f"{name}_px"] = np.nan
                            row[f"{name}_py"] = np.nan

                    if save_annotated_frames:
                        self.draw_skeleton_overlay(frame, landmarks, width, height)
                else:
                    # No person detected in this frame: fill with NaNs
                    for name in ALL_LANDMARKS:
                        row[f"{name}_x"] = np.nan
                        row[f"{name}_y"] = np.nan
                        row[f"{name}_z"] = np.nan
                        row[f"{name}_v"] = 0.0
                        row[f"{name}_px"] = np.nan
                        row[f"{name}_py"] = np.nan

                if save_annotated_frames:
                    # Stamp frame information on top-left
                    status_text = "PERSON DETECTED" if has_person else "NO PERSON"
                    color = (0, 255, 0) if has_person else (0, 0, 255)
                    cv2.putText(
                        frame,
                        f"Frame: {frame_idx:03d} | {status_text}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    annotated_frames.append(frame)

                records.append(row)
                frame_idx += 1

        cap.release()
        df = pd.DataFrame(records)
        return df, annotated_frames, fps, width, height

    def get_or_extract(
        self,
        video_path: Path,
        cache_dir: Path = CACHE_DIR,
        force_recompute: bool = False,
    ) -> Tuple[pd.DataFrame, float, int, int]:
        """
        Checks if cached CSV already exists.
        If cache exists and force_recompute=False: Loads from disk instantly.
        Otherwise: Runs MediaPipe extraction, saves cache, and returns DataFrame.
        """
        video_path = Path(video_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_filename = f"{video_path.stem}_landmarks.csv"
        cache_file = cache_dir / cache_filename

        if cache_file.exists() and not force_recompute:
            print(f"[CACHE HIT] Loaded pre-extracted landmarks from: {cache_file.name}")
            df = pd.read_csv(cache_file)
            
            # Read metadata from first row or video header
            cap = cv2.VideoCapture(str(video_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 720
            cap.release()
            return df, fps, width, height

        print(f"[EXTRACTING] Processing video with MediaPipe: {video_path.name} ...")
        df, _, fps, width, height = self.extract_landmarks_from_video(
            video_path, save_annotated_frames=False
        )

        df.to_csv(cache_file, index=False)
        print(f"[SAVED CACHE] Saved {len(df)} frames of landmarks to: {cache_file.name}")
        return df, fps, width, height

    @staticmethod
    def draw_skeleton_overlay(
        frame: np.ndarray, landmarks, width: int, height: int
    ) -> None:
        """
        Renders sleek skeleton bones and joint keypoints on top of an OpenCV BGR frame.
        """
        # 1. Draw Bones (Lime Green / Cyan lines)
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]
                v1 = getattr(p1, "visibility", 1.0) or 1.0
                v2 = getattr(p2, "visibility", 1.0) or 1.0

                # Only draw if both joint endpoints are reasonably visible
                if v1 > 0.4 and v2 > 0.4:
                    pt1 = (int(p1.x * width), int(p1.y * height))
                    pt2 = (int(p2.x * width), int(p2.y * height))
                    cv2.line(frame, pt1, pt2, (0, 240, 120), 3, cv2.LINE_AA)

        # 2. Draw Joint Keypoints (Vibrant Orange with White Border)
        for lm in landmarks:
            v = getattr(lm, "visibility", 1.0) or 1.0
            if v > 0.4:
                cx, cy = int(lm.x * width), int(lm.y * height)
                # Outer white rim
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 2, cv2.LINE_AA)
                # Inner glowing orange center
                cv2.circle(frame, (cx, cy), 4, (0, 140, 255), -1, cv2.LINE_AA)