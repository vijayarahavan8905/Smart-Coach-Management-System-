"""
Coach Lens - Global Configuration
=================================
Central configuration file holding project paths, dataset location,
and common defaults.

All paths are handled via pathlib.Path to support spaces and Windows/Linux paths seamlessly.
"""

from pathlib import Path

# ==============================================================================
# BASE DIRECTORIES
# ==============================================================================
# Root directory of the Smart Coach project
PROJECT_ROOT = Path(__file__).resolve().parent

# External dataset location (as provided on your Windows Desktop)
# Supports the root 'data set' folder and the 'data set/squats' subfolder
DATASET_ROOT = Path(r"C:\Users\vijay\OneDrive\Documents\Desktop\data set")
try:
    DATASET_DIR = DATASET_ROOT / "squats" if (DATASET_ROOT / "squats").exists() else DATASET_ROOT
except Exception:
    DATASET_DIR = DATASET_ROOT

# Local data directories (inside the project, not committing large files to git)
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"          # Local benchmark sample clips
CACHE_DIR = DATA_DIR / "cache"              # Pose extraction parquet / csv caches
REPORTS_DIR = DATA_DIR / "reports"          # Audit and evaluation reports
TEMPLATES_DIR = DATA_DIR / "templates"      # Reference form templates (Module 4)
OUTPUTS_DIR = DATA_DIR / "outputs"          # Visual debug outputs, plots, skeleton overlays

# Exercise configuration folder (holds JSON configs for squat, pushup, etc.)
EXERCISE_CONFIGS_DIR = PROJECT_ROOT / "configs" / "exercises"

# ==============================================================================
# AUDIT & VIDEO CONSTANTS
# ==============================================================================
# Video file extensions recognized during dataset scanning
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Threshold for considering an exercise dataset "sufficient" for reference building
MIN_RECOMMENDED_CLIPS = 10

# Ensure essential local directories exist
for directory in (DATA_DIR, CACHE_DIR, REPORTS_DIR, TEMPLATES_DIR, OUTPUTS_DIR, EXERCISE_CONFIGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)