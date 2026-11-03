"""
Coach Lens - Module 2: Feature Engineering & Joint Angles
=========================================================
This module converts raw 2D/3D skeletal landmark coordinates into physically
meaningful biomechanical angles across time.

Key Features:
1. Vector Kinematics: Calculates joint angles (knee, hip, elbow) and torso lean
   relative to the vertical plane using stable vector geometry.
2. Automatic Dominant Side Selection: Analyzes left vs. right visibility scores
   to track the camera-facing side and ignore occluded limbs.
3. Gap Filling (Linear Interpolation): Heals short occlusions and missing frames.
4. Savitzky-Golay Smoothing: Removes camera jitter while preserving true peaks
   and movement depth (unlike naive rolling averages which flatten peaks).
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def calculate_joint_angle_2d(
    a_x: np.ndarray, a_y: np.ndarray,
    b_x: np.ndarray, b_y: np.ndarray,
    c_x: np.ndarray, c_y: np.ndarray,
) -> np.ndarray:
    """
    Computes the 2D interior angle at vertex B formed by segments BA and BC.
    
    Formula:
        ba = A - B
        bc = C - B
        cos(theta) = (ba . bc) / (||ba|| * ||bc||)
        theta = arccos(clip(cos(theta), -1.0, 1.0)) in degrees [0, 180]
        
    Args:
        a_x, a_y: Coordinates of proximal joint A (e.g. Hip)
        b_x, b_y: Coordinates of center joint B (e.g. Knee)
        c_x, c_y: Coordinates of distal joint C (e.g. Ankle)
        
    Returns:
        NumPy array of angles in degrees.
    """
    # Vector from center B to A
    ba_x = a_x - b_x
    ba_y = a_y - b_y
    
    # Vector from center B to C
    bc_x = c_x - b_x
    bc_y = c_y - b_y
    
    dot_product = (ba_x * bc_x) + (ba_y * bc_y)
    norm_ba = np.sqrt((ba_x ** 2) + (ba_y ** 2))
    norm_bc = np.sqrt((bc_x ** 2) + (bc_y ** 2))
    
    # Avoid division by zero with small epsilon
    denominator = (norm_ba * norm_bc) + 1e-7
    cosine = dot_product / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    
    return np.degrees(np.arccos(cosine))


def calculate_torso_lean_angle(
    shoulder_x: np.ndarray, shoulder_y: np.ndarray,
    hip_x: np.ndarray, hip_y: np.ndarray,
) -> np.ndarray:
    """
    Calculates the trunk (torso) forward lean angle relative to vertical.
    
    In computer vision coordinates:
        (0, 0) is top-left, so +Y points straight DOWN.
        A vertical vector pointing straight UP has direction: dx = 0, dy = -1.
        
    Torso vector (from Hip to Shoulder):
        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y (negative when upright)
        
    Lean Angle:
        arctan2(|dx|, -dy) converted to degrees.
        0 deg  = perfectly upright torso
        90 deg = completely bent horizontal torso
    """
    dx = np.abs(shoulder_x - hip_x)
    dy = -(shoulder_y - hip_y)  # Invert so upright is positive
    
    lean = np.degrees(np.arctan2(dx, dy))
    return np.clip(lean, 0.0, 90.0)


def select_dominant_body_side(df: pd.DataFrame) -> Tuple[str, float, float]:
    """
    Determines whether the person's left side or right side faces the camera.
    
    Computes the mean visibility score of key joints on both sides:
        Left:  left_shoulder, left_hip, left_knee, left_ankle
        Right: right_shoulder, right_hip, right_knee, right_ankle
        
    Returns:
        dominant_side ('left' or 'right'), left_avg_vis, right_avg_vis
    """
    left_vis_cols = ["left_shoulder_v", "left_hip_v", "left_knee_v", "left_ankle_v"]
    right_vis_cols = ["right_shoulder_v", "right_hip_v", "right_knee_v", "right_ankle_v"]
    
    # Ensure columns exist in DataFrame
    valid_left = [c for c in left_vis_cols if c in df.columns]
    valid_right = [c for c in right_vis_cols if c in df.columns]
    
    left_mean = float(df[valid_left].mean().mean()) if valid_left else 0.5
    right_mean = float(df[valid_right].mean().mean()) if valid_right else 0.5
    
    dominant_side = "left" if left_mean >= right_mean else "right"
    return dominant_side, left_mean, right_mean


def fill_gaps_and_smooth(
    series: pd.Series,
    window_length: int = 9,
    polyorder: int = 2,
    max_gap_limit: int = 5,
) -> np.ndarray:
    """
    Fills short missing gaps using linear interpolation and applies a
    Savitzky-Golay filter to eliminate noise without clipping motion peaks.
    
    Args:
        series: Raw angle series (with possible NaNs).
        window_length: Size of smoothing filter window (must be odd integer).
        polyorder: Polynomial order for local fitting.
        max_gap_limit: Max consecutive missing frames to interpolate.
        
    Returns:
        Cleaned, smoothed 1D NumPy array.
    """
    # 1. Fill missing gaps linearly
    interpolated = series.interpolate(
        method="linear", limit=max_gap_limit, limit_direction="both"
    )
    # Forward-fill / back-fill any edge NaNs
    interpolated = interpolated.bfill().ffill()
    
    # If all values are NaN, return zeros
    if interpolated.isna().all():
        return np.zeros(len(series))
    
    values = interpolated.to_numpy()
    n_samples = len(values)
    
    # 2. Savitzky-Golay Smoothing
    # Filter window must be odd and <= number of samples
    if n_samples < 5:
        # Too short for polynomial filtering; return interpolated
        return values
        
    effective_window = min(window_length, n_samples)
    if effective_window % 2 == 0:
        effective_window -= 1  # Window length must be odd
        
    if effective_window > polyorder:
        try:
            smoothed = savgol_filter(values, effective_window, polyorder)
            return smoothed
        except Exception:
            # Fallback to rolling average if Savitzky-Golay fails
            return pd.Series(values).rolling(window=3, min_periods=1, center=True).mean().to_numpy()
            
    return values


class FeatureExtractor:
    """
    High-level biomechanical feature extractor that produces clean joint angle
    trajectories from landmark dataframes.
    """

    def __init__(self, smoothing_window: int = 9, polyorder: int = 2):
        self.smoothing_window = smoothing_window
        self.polyorder = polyorder

    def extract_features(
        self,
        landmarks_df: pd.DataFrame,
        fps: float = 30.0,
        preferred_side: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Transforms raw landmark DataFrame into kinematic angle features.
        
        Args:
            landmarks_df: Output from PoseExtractor.
            fps: Video frames per second.
            preferred_side: Explicitly force 'left' or 'right' side, or None for auto.
            
        Returns:
            DataFrame with raw and smoothed angle trajectories.
        """
        if landmarks_df.empty:
            return pd.DataFrame()

        df = landmarks_df.copy()
        n_frames = len(df)

        # 1. Determine Dominant Side
        auto_side, left_vis, right_vis = select_dominant_body_side(df)
        dominant_side = preferred_side.lower() if preferred_side in ("left", "right") else auto_side

        features = pd.DataFrame()
        features["frame_idx"] = df["frame_idx"]
        features["timestamp_sec"] = df.get("timestamp_sec", np.round(df["frame_idx"] / fps, 3))
        features["dominant_side"] = dominant_side
        features["left_vis_score"] = round(left_vis, 3)
        features["right_vis_score"] = round(right_vis, 3)

        # 2. Compute Left Side Joint Angles
        left_knee_raw = calculate_joint_angle_2d(
            df["left_hip_x"].values, df["left_hip_y"].values,
            df["left_knee_x"].values, df["left_knee_y"].values,
            df["left_ankle_x"].values, df["left_ankle_y"].values,
        )
        left_hip_raw = calculate_joint_angle_2d(
            df["left_shoulder_x"].values, df["left_shoulder_y"].values,
            df["left_hip_x"].values, df["left_hip_y"].values,
            df["left_knee_x"].values, df["left_knee_y"].values,
        )
        left_elbow_raw = calculate_joint_angle_2d(
            df["left_shoulder_x"].values, df["left_shoulder_y"].values,
            df["left_elbow_x"].values, df["left_elbow_y"].values,
            df["left_wrist_x"].values, df["left_wrist_y"].values,
        )
        left_torso_raw = calculate_torso_lean_angle(
            df["left_shoulder_x"].values, df["left_shoulder_y"].values,
            df["left_hip_x"].values, df["left_hip_y"].values,
        )

        # 3. Compute Right Side Joint Angles
        right_knee_raw = calculate_joint_angle_2d(
            df["right_hip_x"].values, df["right_hip_y"].values,
            df["right_knee_x"].values, df["right_knee_y"].values,
            df["right_ankle_x"].values, df["right_ankle_y"].values,
        )
        right_hip_raw = calculate_joint_angle_2d(
            df["right_shoulder_x"].values, df["right_shoulder_y"].values,
            df["right_hip_x"].values, df["right_hip_y"].values,
            df["right_knee_x"].values, df["right_knee_y"].values,
        )
        right_elbow_raw = calculate_joint_angle_2d(
            df["right_shoulder_x"].values, df["right_shoulder_y"].values,
            df["right_elbow_x"].values, df["right_elbow_y"].values,
            df["right_wrist_x"].values, df["right_wrist_y"].values,
        )
        right_torso_raw = calculate_torso_lean_angle(
            df["right_shoulder_x"].values, df["right_shoulder_y"].values,
            df["right_hip_x"].values, df["right_hip_y"].values,
        )

        # Store side-specific raw angles
        features["left_knee_angle"] = np.round(left_knee_raw, 2)
        features["right_knee_angle"] = np.round(right_knee_raw, 2)
        features["left_hip_angle"] = np.round(left_hip_raw, 2)
        features["right_hip_angle"] = np.round(right_hip_raw, 2)
        features["left_elbow_angle"] = np.round(left_elbow_raw, 2)
        features["right_elbow_angle"] = np.round(right_elbow_raw, 2)

        # 4. Assign Primary Tracked Angles based on Dominant Side
        if dominant_side == "left":
            primary_knee_raw = left_knee_raw
            primary_hip_raw = left_hip_raw
            primary_elbow_raw = left_elbow_raw
            primary_torso_raw = left_torso_raw
        else:
            primary_knee_raw = right_knee_raw
            primary_hip_raw = right_hip_raw
            primary_elbow_raw = right_elbow_raw
            primary_torso_raw = right_torso_raw

        features["primary_knee_raw"] = np.round(primary_knee_raw, 2)
        features["primary_hip_raw"] = np.round(primary_hip_raw, 2)
        features["primary_elbow_raw"] = np.round(primary_elbow_raw, 2)
        features["torso_lean_raw"] = np.round(primary_torso_raw, 2)

        # 5. Gap Filling and Smoothing on Primary Features
        features["primary_knee_smooth"] = np.round(
            fill_gaps_and_smooth(
                features["primary_knee_raw"],
                window_length=self.smoothing_window,
                polyorder=self.polyorder,
            ),
            2,
        )
        features["primary_hip_smooth"] = np.round(
            fill_gaps_and_smooth(
                features["primary_hip_raw"],
                window_length=self.smoothing_window,
                polyorder=self.polyorder,
            ),
            2,
        )
        features["primary_elbow_smooth"] = np.round(
            fill_gaps_and_smooth(
                features["primary_elbow_raw"],
                window_length=self.smoothing_window,
                polyorder=self.polyorder,
            ),
            2,
        )
        features["torso_lean_smooth"] = np.round(
            fill_gaps_and_smooth(
                features["torso_lean_raw"],
                window_length=self.smoothing_window,
                polyorder=self.polyorder,
            ),
            2,
        )

        return features