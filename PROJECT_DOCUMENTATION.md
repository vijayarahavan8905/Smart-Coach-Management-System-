# 🏋️ Smart Coach Management System (Coach Lens)
## Complete Technical Project Documentation & System Specification
**Academic Department:** Department of Computer Science & Engineering  
**Project Title:** Coach Lens: AI-Powered Gym Exercise Form Analysis & Feedback System  
**Framework:** MediaPipe BlazePose • Dynamic Time Warping (FastDTW) • Kinematic Goniometry • Streamlit  

---

## 1. Executive Summary & Abstract

Traditional resistance training carries a substantial risk of musculoskeletal injury when performed with improper biomechanical form. While elite athletes have continuous access to personal trainers, everyday fitness enthusiasts rely on unsupervised training or unverified online advice. 

**Smart Coach Management System (Coach Lens)** is an automated, explainable computer-vision framework that transforms any standard smartphone video into real-time biomechanical telemetry. The system:
1. Detects and tracks 33 3D skeletal body landmarks per video frame using Google's MediaPipe BlazePose.
2. Computes scale-invariant joint angles (knee flexion, hip hinge, elbow flexion) and torso inclination relative to gravity vertical.
3. Automatically identifies the camera-facing dominant side of the body, discarding occluded limbs.
4. Segments continuous video into discrete, valid repetitions using topological peak/valley detection and physics-based validation filters.
5. Compares each repetition against a golden reference template using **Fast Dynamic Time Warping (FastDTW)** to eliminate pacing differences.
6. Generates an explainable **0–100 Form Score**, pinpoints the exact frame of greatest defect with annotated skeleton visuals, and delivers actionable coaching advice.
7. Extends to new exercises (e.g., Push-ups, Barbell Biceps Curls) purely through declarative JSON configuration files with zero machine-learning retraining.

---

## 2. System Architecture & Workflow

The system is organized into a modular pipeline where each stage processes and refines the kinematic signal:

```
                  ┌──────────────────────────────────────┐
                  │      Input Video (.mp4 / .mov)       │
                  │   Recorded via Smartphone Camera     │
                  └──────────────────┬───────────────────┘
                                     │
                                     ▼
                  ┌──────────────────────────────────────┐
                  │    Module 1: Pose Landmark Engine    │
                  │    MediaPipe BlazePose 33 Joints     │
                  │   Cached to data/cache/*.csv (85x)   │
                  └──────────────────┬───────────────────┘
                                     │
                                     ▼
                  ┌──────────────────────────────────────┐
                  │   Module 2: Biomechanical Feature   │
                  │        Vector Angle Goniometry       │
                  │    Dominant Camera Side Selection    │
                  │    Savitzky-Golay Noise Filtering    │
                  └──────────────────┬───────────────────┘
                                     │
                                     ▼
                  ┌──────────────────────────────────────┐
                  │   Module 3: Repetition Segmentation  │
                  │   Peak/Valley Inflection Detection   │
                  │    Monotonic Phase Boundary Search   │
                  │   ROM (>=35°) & Duration (1-8s) Gate │
                  └──────────────────┬───────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
            ┌─────────────────────────┐ ┌─────────────────────────┐
            │   Module 4: Reference   │ │   Module 5: Comparison  │
            │     Template Builder    │ │     & Scoring Engine    │
            │   Time Normalization    │ │   FastDTW Temporal Warp │
            │   (100 Steps Uniform)   │ │  0–100 Explainable Score│
            │  Mean & Std Corridors   │ │   Worst Frame Detection │
            └─────────────────────────┘ └────────────┬────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │ Module 7: Coach Advisor │
                                        │  Rule-Based Bio-Tips +  │
                                        │   Optional Gemini AI    │
                                        └────────────┬────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │ Module 8: Web Dashboard │
                                        │   Streamlit + Plotly    │
                                        │ Radar Telemetry, Gauge, │
                                        │ Skeleton Frame Scrubber │
                                        └─────────────────────────┘
```

---

## 3. Mathematical Foundations & Formulations

### 3.1 2D Scale-Invariant Vector Joint Angle
To evaluate joint motion regardless of athlete height or distance from the lens, interior angles are computed via the vector dot product. For joint $B$ (vertex) between proximal segment $A$ and distal segment $C$:

$$\vec{u} = A - B = (x_A - x_B, \, y_A - y_B)$$
$$\vec{v} = C - B = (x_C - x_B, \, y_C - y_B)$$

$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\| + \epsilon} = \frac{(x_A - x_B)(x_C - x_B) + (y_A - y_B)(y_C - y_B)}{\sqrt{(x_A - x_B)^2 + (y_A - y_B)^2} \sqrt{(x_C - x_B)^2 + (y_C - y_B)^2} + 10^{-7}}$$

$$\theta = \arccos\left(\text{clip}\left(\cos(\theta), -1.0, 1.0\right)\right) \times \frac{180^\circ}{\pi}$$

* Standing extended leg: $\theta \approx 175^\circ - 180^\circ$
* Parallel squat depth: $\theta \le 90^\circ$

### 3.2 Torso Forward Lean Relative to Gravity
In computer vision coordinate space, the vertical $+Y$ axis points downwards. A vector pointing upward parallel to gravity is $\vec{g} = (0, -1)$. The torso vector from Hip $H$ to Shoulder $S$ is $\vec{T} = (x_S - x_H, \, y_S - y_H)$. The forward inclination angle $\theta_{\text{lean}}$ is:

$$\theta_{\text{lean}} = \arctan2(|x_S - x_H|, \, -(y_S - y_H)) \times \frac{180^\circ}{\pi}$$

* Upright standing posture: $0^\circ - 15^\circ$
* Safe squat descent: $20^\circ - 35^\circ$
* Hazardous forward collapse (injury risk): $> 35^\circ$

### 3.3 Savitzky-Golay Digital Filter
To remove camera tracking jitter without dampening true peak depth (which occurs when using standard moving averages), local polynomials of order $p=2$ are fitted across sliding odd-sized windows of length $m=9$:

$$\hat{y}_k = \sum_{i=-\frac{m-1}{2}}^{\frac{m-1}{2}} c_i \, y_{k+i}$$

### 3.4 Dynamic Time Warping (FastDTW)
Given a normalized user repetition sequence $X = [x_1, x_2, \dots, x_N]$ and golden reference sequence $Y = [y_1, y_2, \dots, y_M]$ where $N = M = 100$, DTW solves for the optimal alignment path $W = [w_1, w_2, \dots, w_K]$ minimizing cumulative distance:

$$D(i, j) = |x_i - y_j| + \min \begin{cases} D(i-1, j) & \text{(Insertion)} \\ D(i, j-1) & \text{(Deletion)} \\ D(i-1, j-1) & \text{(Match)} \end{cases}$$

### 3.5 Explainable 0–100 Form Scoring Formula
$$\text{Score} = \max\left(0, \, \min\left(100, \, 100 - (\text{Penalty}_{\text{depth}} + \text{Penalty}_{\text{torso}} + \text{Penalty}_{\text{trajectory}})\right)\right)$$

* **Depth Penalty (Max 40 points):**
  $$\text{Penalty}_{\text{depth}} = \min\left(40.0, \, \max(0.0, \, \theta_{\text{knee\_min}} - 90.0) \times 1.5\right)$$
* **Torso Lean Penalty (Max 30 points):**
  $$\text{Penalty}_{\text{torso}} = \min\left(30.0, \, \max(0.0, \, \theta_{\text{torso\_peak}} - 35.0) \times 1.5\right)$$
* **Trajectory Smoothness Penalty (Max 30 points):**
  $$\text{Penalty}_{\text{trajectory}} = \min\left(30.0, \, \max(0.0, \, (\text{Normalized\_DTW} - 4.0) \times 2.5)\right)$$

---

## 4. Module-by-Module Technical Specification

### Module 0: Dataset Audit & Health Validation
* **Source:** `audit_dataset.py`
* **Purpose:** Programmatically checks every video file in the target repository for valid codec decoding, resolution, frame rate, duration, and corruption.
* **Audit Results:**
  * Total files scanned: 29 clips
  * Codec readability: 29 / 29 (100% readable, 0 corrupt files)
  * Resolutions: 1280x720 (23 clips), 1920x1080 (6 clips)
  * Frame rates: 23.98 fps, 25.0 fps, 29.97 fps
  * Durations: 1.96s to 22.15s (Mean: 10.24s)

### Module 1: Pose Landmark Extraction & Disk Caching
* **Source:** `pose_extractor.py`
* **Features:**
  * Uses Google's MediaPipe Tasks `vision.PoseLandmarker` (`pose_landmarker_full.task`).
  * Extracts normalized coordinates $(x, y, z)$, pixel coordinates $(px, py)$, and visibility confidences $(v)$ for 33 joints per frame.
  * Implements disk caching in `data/cache/{video_name}_landmarks.csv`. Re-running on pre-analyzed video drops execution time from **8.8s to 0.10s (85x speedup)**.
  * Automatic network fallback: If `pose_landmarker_full.task` is missing on a cloned machine, it downloads automatically from Google Cloud Storage.

### Module 2: Biomechanical Kinematic Feature Engine
* **Source:** `features.py`
* **Features:**
  * Computes 2D vector angles and forward trunk inclination.
  * **Automatic Dominant Side Selection:** Compares left-side joint confidence (`left_shoulder_v`, `left_hip_v`, `left_knee_v`, `left_ankle_v`) against right-side confidence. Whichever side faces the camera is automatically tracked, preventing occluded body parts from generating corrupt angles.
  * Linear interpolation bridges brief tracking gaps up to 5 consecutive frames.
  * Savitzky-Golay polynomial smoothing eliminates high-frequency camera vibration.

### Module 3: Repetition Segmentation & Boundary Detection
* **Source:** `rep_detector.py`
* **Features:**
  * Inverts joint trajectories ($\text{search\_signal} = -\theta$) to locate inflection points (bottom of squat/push-up) via `scipy.signal.find_peaks`.
  * Walks backward and forward from inflection points to find exact start and finish boundaries.
  * Gated by physics constraints: Range of Motion must be $\ge 35^\circ$; duration must be between $1.0\text{s}$ and $8.0\text{s}$.
  * Verified across dataset clips:
    * `squat_10.mp4`: 2 reps detected
    * `squat_13.mp4`: 2 reps detected
    * `squat_15.mp4`: 3 reps detected

### Module 4: Golden Reference Template Builder
* **Source:** `build_template.py`
* **Features:**
  * Normalizes repetitions of varying duration onto a standardized 100-step time grid ($0\% \to 50\% \to 100\%$).
  * Computes point-wise Mean $\mu(t)$ (target trajectory) and Standard Deviation $\sigma(t)$ (tolerance corridor).
  * Enforces a minimum biological tolerance buffer of $\pm 5^\circ$.
  * Generated templates: `squat_template.json`, `pushup_template.json`, `curl_template.json`.

### Module 5: Comparison & 0–100 Form Scoring Engine
* **Source:** `compare.py`
* **Features:**
  * FastDTW non-linear temporal alignment against the golden reference template.
  * Computes Depth Gap, Torso Gap, and Trajectory Variance.
  * Pinpoints the **Worst Moment** (exact frame number and physical reason) for focused user improvement.

### Module 6: Benchmark Evaluation & Teammate Labeling Protocol
* **Source:** `evaluate.py`
* **Features:**
  * Acknowledges that the Kaggle dataset contains no ground-truth good/bad labels.
  * Evaluated on a 20-sample controlled synthetic benchmark (10 Textbook Reps, 5 Shallow Squats, 5 Torso Collapses):
    * **ROC-AUC Score:** `1.000` (Perfect discrimination)
    * **Precision:** `83.3%`
    * **Recall / Sensitivity:** `100.0%`
    * **F1-Score:** `90.9%`
  * Generates `teammate_form_labeling_sheet.csv` for standardized team collection.

### Module 7: Actionable Coaching Advisor
* **Source:** `coach.py`
* **Features:**
  * **Tier 1 (Deterministic Rule Engine):** Maps measured degree gaps to biomechanical coaching tips with zero latency and 100% uptime.
  * **Tier 2 (Optional Gemini AI Rephraser):** Uses Google's free-tier Gemini API to rephrase exact measured numbers into friendly coach tone; automatically falls back to Tier 1 if offline.

### Module 8: Interactive Web Application
* **Source:** `app.py`
* **Features:**
  * Clean, academic dashboard built with Streamlit and custom styling.
  * Radial Form Score Gauge (0–100).
  * 5-Axis Plotly Biomechanical Radar / Spider Chart (Depth, Posture, Smoothness, ROM, Tempo).
  * Interactive Skeleton Frame Scrubber: Users can drag a slider to inspect MediaPipe skeleton tracking across any frame.
  * Interactive Plotly hoverable curves with golden reference corridors.
  * 1-Click Telemetry CSV Export.

### Module 9: Multi-Exercise Configs & Extensibility
* **Source:** `configs/exercises/`
* **Features:**
  * Adding Push-up (`pushup.json`) and Bicep Curl (`curl.json`) required **zero lines of code modification** in the core engine.
  * Proves modularity and scalability of the architecture.

---

## 5. File Structure of Repository

```
Smart Coach/
│
├── configs/exercises/          # Declarative Exercise Rulebooks (JSON)
│   ├── squat.json              # Squat thresholds, primary metrics, coaching text
│   ├── pushup.json             # Push-up elbow flexion & plank posture rules
│   └── curl.json               # Bicep curl contraction & torso sway rules
│
├── data/
│   ├── cache/                  # Fast landmark & feature CSV caches (85x reload speedup)
│   ├── outputs/                # Diagnostic skeleton & trajectory plots
│   ├── reports/                # Audit & evaluation CSV reports
│   └── templates/              # Golden reference templates (squat, pushup, curl)
│
├── app.py                      # Interactive Streamlit Web Application
├── demo.py                     # Single-command end-to-end CLI demonstration
├── pose_extractor.py           # Module 1: MediaPipe 33-landmark pose extractor
├── features.py                 # Module 2: Scale-invariant kinematics & digital filter
├── rep_detector.py             # Module 3: Repetition boundary segmenter
├── build_template.py           # Module 4: Time normalization & template generator
├── compare.py                  # Module 5: FastDTW comparison & 0-100 scoring engine
├── evaluate.py                 # Module 6: Synthetic evaluation & team labeling
├── coach.py                    # Module 7: Actionable coaching feedback advisor
├── audit_dataset.py            # Module 0: Dataset scanner & health check
│
├── README.md                   # Repository landing page with architecture & quickstart
├── REVIEW_VIVA_GUIDE.md        # Comprehensive 30-sec pitch, 3-min demo & Viva Q&A
├── PROJECT_DOCUMENTATION.md    # Complete technical documentation (this file)
├── requirements.txt            # Python dependencies (all free & open-source)
└── .gitignore                  # Git hygiene rules (excludes heavy videos & caches)
```

---

## 6. Installation & Execution Guide

### Prerequisites
* Operating System: Windows 10/11, macOS, or Linux
* Python Version: Python 3.10 to 3.13
* Web Browser: Google Chrome, Microsoft Edge, or Firefox

### Step 1: Install Dependencies
Open terminal or PowerShell in the project directory:
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Command Line Demo
```bash
python demo.py
```
*Processes `squat_10.mp4`, segments reps, computes Form Scores, and prints telemetry in 2 seconds.*

### Step 3: Launch Web Application
```bash
streamlit run app.py
```
*Opens [http://localhost:8501](http://localhost:8501) with full interactive telemetry, radar charts, and skeleton frame scrubber.*

---

## 7. Known Limitations & Engineering Assumptions

1. **Sagittal / Diagonal Camera Orientation:** Movement along the sagittal plane (squats, curls, push-ups) requires side-on or 45-degree filming. Direct frontal cameras suffer from perspective foreshortening.
2. **Lighting & Clothing:** Loose, baggy blankets or heavy coats can obscure joint landmarks. Standard gym attire (shorts, track pants, t-shirts) is recommended.
3. **Single Athlete in Frame:** The system assumes a single primary subject in the workout field of view. Multiple individuals crossing between the camera and lifter can cause temporary landmark dropouts.

---

## 8. Conclusion

Coach Lens demonstrates that reliable, injury-preventing gym coaching does not require black-box neural network classifiers or paid proprietary APIs. By combining Google's free MediaPipe pose tracking with rigorous kinematic vector geometry and Fast Dynamic Time Warping, the system provides transparent, explainable feedback grounded in physical sports science.

`