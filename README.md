# 🏋️ Coach Lens: AI-Powered Gym Exercise Trainer

**Coach Lens** is an intelligent, explainable computer-vision system that analyzes gym exercise form from standard phone camera videos and delivers real-time repetition counting, form scoring (0–100), joint-level defect detection, and actionable coaching feedback.

Built for CSE Final Year Engineering Capstone.

---

## 🏗️ System Architecture & Workflow

```
[Raw Phone Video (.mp4/.mov)]
       │
       ▼
┌────────────────────────────────────────┐
│ Module 1: Pose Landmark Extraction     │  MediaPipe PoseLandmarker (33 3D joints)
│ & Disk Caching                         │  Fast CSV disk cache in data/cache/
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ Module 2: Biomechanical Kinematics     │  Scale-invariant vector angles (knee, hip, elbow)
│ & Savitzky-Golay Smoothing             │  Auto camera-facing side selection
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ Module 3: Repetition Segmentation      │  Config-driven valley/peak detection
│ & Boundary Detection                   │  Monotonic phase boundary identification
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ Module 4: Golden Reference Template    │  Time-normalized (100% scale) mean & std corridors
│ (squat_template.json)                  │  Aggregated across verified textbook reps
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ Module 5: FastDTW Form Scoring         │  Dynamic Time Warping tempo alignment
│ (Explainable 0–100 Formula)            │  Penalty = Depth Gap + Torso Lean + Trajectory
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ Module 7: Coach Feedback Advisor       │  Deterministic rule tips + Optional Gemini AI
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ Module 8: Interactive Web App          │  Streamlit Dashboard: Video + Worst Frame Skeleton
│ (app.py)                               │  Telemetry Tables + CSV Download
└────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Requirements & Dependencies
Ensure Python 3.10 to 3.13 is installed on Windows. All required libraries are 100% free and open-source:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End CLI Demo
Execute the full automated pipeline on a dataset video with a single command:
```bash
python demo.py
```

### 3. Launch the Interactive Web Dashboard
Start the Streamlit application in your browser:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in Chrome or Edge.

---

## 📐 Explainable Scoring Formula (Jury Ready)

Unlike black-box neural networks that output unexplainable percentages, Coach Lens uses a **transparent, physics-grounded deduction formula**:

$$\text{Form Score} = 100 - (\text{Penalty}_{\text{depth}} + \text{Penalty}_{\text{torso}} + \text{Penalty}_{\text{trajectory}})$$

Where:
* **$\text{Penalty}_{\text{depth}}$ (Max 40 pts):**
  $$\min\left(40.0, \, \max(0.0, \, \theta_{\text{knee}} - 90^\circ) \times 1.5\right)$$
  *(Deducts 1.5 points for every degree the user stops above parallel 90° depth).*
* **$\text{Penalty}_{\text{torso}}$ (Max 30 pts):**
  $$\min\left(30.0, \, \max(0.0, \, \theta_{\text{torso\_lean}} - 35^\circ) \times 1.5\right)$$
  *(Deducts 1.5 points for every degree the chest collapses past 35° forward lean).*
* **$\text{Penalty}_{\text{trajectory}}$ (Max 30 pts):**
  $$\min\left(30.0, \, \max(0.0, \, (\text{DTW\_Distance} - 4.0) \times 2.5)\right)$$
  *(Penalizes erratic shaking, asymmetrical ascent, or jerky path deviations).*

---

## 📁 Repository Structure

```
Smart Coach/
│
├── configs/exercises/          # Exercise configuration rules (JSON)
│   ├── squat.json              # Squat thresholds, primary metrics & tips
│   ├── pushup.json             # Push-up elbow & plank rules
│   └── curl.json               # Bicep curl contraction & swing rules
│
├── data/
│   ├── cache/                  # Fast landmark & feature CSV caches
│   ├── outputs/                # Generated visual skeleton overlays & plots
│   ├── reports/                # Dataset audit & evaluation CSVs
│   └── templates/              # Golden reference templates (JSON)
│
├── audit_dataset.py            # Module 0: Dataset scanner & health check
├── pose_extractor.py           # Module 1: MediaPipe 33-landmark extraction & caching
├── features.py                 # Module 2: 2D joint angles & Savitzky-Golay smoothing
├── rep_detector.py             # Module 3: Config-driven repetition segmentation
├── build_template.py           # Module 4: Time normalization & golden template builder
├── compare.py                  # Module 5: FastDTW comparison & 0-100 scoring engine
├── evaluate.py                 # Module 6: Synthetic benchmark & team labeling protocol
├── coach.py                    # Module 7: Rule-based & optional Gemini AI feedback
├── app.py                      # Module 8: Interactive Streamlit web app
├── demo.py                     # Module 9: Single-click end-to-end demo script
├── config.py                   # Central path & directory configuration
└── requirements.txt            # Python dependencies
```

---

## 🔬 Benchmark Evaluation Results (Module 6)

Tested on a controlled benchmark of 20 repetitions (10 Textbook Good Reps, 5 Shallow Squats, 5 Chest Collapses):
* **ROC-AUC Score:** `1.000` (Perfect discrimination between good form vs. defective reps)
* **Precision:** `83.3%`
* **Sensitivity / Recall:** `100.0%` (Zero dangerous defective reps missed)
* **F1-Score:** `90.9%`

---

## ⚠️ Known Limitations & Assumptions
1. **Camera Angle:** Recommended from the sagittal (side) plane or 45-degree angle. Extreme front-facing views compress knee flexion due to perspective projection.
2. **Lighting & Occlusions:** Baggy sweatpants or loose gym clothing can introduce small angular noise in knee landmark estimates.
3. **Single Person:** Designed for single-athlete workout footage. If bystanders walk directly between the camera and lifter, tracking may glitch for those frames.
