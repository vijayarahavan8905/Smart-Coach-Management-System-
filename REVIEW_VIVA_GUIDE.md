# 🎓 Coach Lens: Project Review & Viva Defense Guide

Use this document to prepare for and ace your project review tomorrow.

---

## ⚡ 1. The 30-Second Elevator Pitch
> *"Good morning, respected examiners. Our project is **Coach Lens**, an AI-powered gym trainer that analyzes exercise video from a standard smartphone camera. Instead of using a black-box neural network that can't explain why a workout rep was bad, Coach Lens implements a two-stage hybrid pipeline: we use Google's MediaPipe deep learning model for real-time 3D skeletal landmark extraction, followed by a biomechanical kinematic engine that calculates joint angles and applies Dynamic Time Warping (FastDTW) against textbook golden templates. Our system outputs repetition count, an explainable 0–100 form score, pinpointed physical defects in degrees (such as shallow depth or forward torso lean), an annotated skeleton image of the worst moment, and actionable coaching advice."*

---

## 🚀 2. How to Demonstrate the Project in 3 Minutes

### Step A: Terminal Proof (1 minute)
Open PowerShell in your project folder and run:
```bash
python demo.py
```
**What to point out on screen:**
1. The script loads 175 frames in less than 0.1 seconds from our high-speed cache (`squat_10_landmarks.csv`).
2. Point to the **Dominant Camera Side**: *"Notice the system automatically detected that the lifter is facing left, so it tracked the camera-facing left knee and ignored the occluded right leg."*
3. Point to the **Rep Breakdown**: *"It detected exactly 2 reps, measured lowest knee angle ($26.4^\circ$ and $27.7^\circ$), and scored Rep 2 at 92.2/100 (Grade A) with zero depth penalty."*

### Step B: Interactive Web Application (2 minutes)
In PowerShell, launch Streamlit:
```bash
streamlit run app.py
```
**What to show the jury:**
1. Select `squat_10.mp4` or upload a video and click **"Analyze Exercise Form"**.
2. Show the top metric cards (Form Score, Reps Completed, Average Knee Depth).
3. Show the **"Worst Moment Skeleton"**: *"Here, the computer vision engine identified the exact frame of greatest defect and highlighted the skeletal joints."*
4. Click on the **DTW Alignment Plot**: Show how the user's curve is compared inside the cyan reference tolerance corridor.
5. Point to the **Download CSV Report** button.

---

## 🎯 3. Top 10 Viva Questions & Expert Answers

### Q1: Why didn't you train a CNN / LSTM classifier from scratch to classify Good vs Bad form?
**Answer:**
> *"Three reasons: First, public fitness datasets (including Kaggle) have zero per-rep good/bad form labels; supervised deep learning cannot train without ground truth. Second, training an end-to-end video CNN creates an unexplainable black box—it might say 'Bad form 80%', but it cannot explain to a gym lifter how many degrees their knee was shallow. Third, our hybrid architecture (MediaPipe landmark detection + kinematic vector geometry) is 100% explainable, requires zero expensive GPU retraining, and runs in real time on any consumer laptop."*

### Q2: What is Dynamic Time Warping (DTW) and why is it needed?
**Answer:**
> *"A gym trainer might perform a squat in 2 seconds, while a beginner performs it in 4 seconds or pauses at the bottom. A naive point-by-point Euclidean distance would compare the beginner's paused bottom position to the trainer's ascent, producing huge false penalties. Dynamic Time Warping (DTW) finds the optimal non-linear temporal alignment path between two time-series curves, allowing us to compare body mechanics fairly regardless of how fast or slow the person moves."*

### Q3: How do you handle videos taken from the left vs. right side of the body?
**Answer:**
> *"In `features.py`, our `select_dominant_body_side` function computes the average MediaPipe visibility confidence score for left-side joints (shoulder, hip, knee, ankle) versus right-side joints. Whichever side has higher visibility is automatically designated the primary tracked side. This completely eliminates errors from occluded limbs hidden behind the body."*

### Q4: Why did you use Savitzky-Golay filtering instead of a Simple Moving Average?
**Answer:**
> *"A simple moving average or rolling mean flattens extreme peaks. If an athlete hits a deep parallel squat of 85 degrees, a rolling average dampens the curve and falsely reports 92 degrees, falsely penalizing their depth. The Savitzky-Golay filter fits local low-degree polynomials across a sliding window, effectively removing high-frequency camera noise while preserving true local minima and maxima."*

### Q5: How is your 0–100 Form Score calculated?
**Answer:**
> *"Our scoring formula is completely transparent and mathematically bounded:
> $$\text{Score} = 100 - (\text{Penalty}_{\text{depth}} + \text{Penalty}_{\text{torso}} + \text{Penalty}_{\text{trajectory}})$$
> - Depth penalty deducts 1.5 points for every degree the knee stops above parallel (90 degrees).
> - Torso penalty deducts 1.5 points for every degree the chest leans forward beyond the safe 35-degree threshold.
> - Trajectory penalty penalizes erratic shaking or asymmetric paths via normalized DTW distance."*

### Q6: How do you segment repetitions from continuous video?
**Answer:**
> *"In `rep_detector.py`, we apply topological peak/valley detection (`scipy.signal.find_peaks`) with prominence thresholds on the inverted primary joint angle curve. Once the lowest inflection point is identified, our algorithm searches backward to find the initial standing descent and forward to find the standing ascent completion, enforcing physics-based Range-of-Motion (minimum 35 degrees) and duration constraints (1.0 to 8.0 seconds) to filter out fidgeting and camera movement."*

### Q7: How does your system support other exercises like Push-ups or Bicep Curls?
**Answer:**
> *"Our architecture is completely decoupled: the core engine is exercise-agnostic, and all exercise-specific logic is stored in external JSON configuration files (`configs/exercises/squat.json`, `pushup.json`, `curl.json`). To switch exercises, the system simply swaps the target joint angle (knee vs. elbow) and thresholds from the config without modifying any underlying Python code."*

### Q8: What if MediaPipe loses tracking or no person is detected?
**Answer:**
> *"In `pose_extractor.py`, if a frame has no detected person, it records `person_detected = False` and fills the landmarks with NaN. In `features.py`, our feature extraction pipeline applies linear interpolation up to 5 consecutive missing frames to heal temporary drops, while leading or trailing empty frames are safely filtered out before repetition segmentation."*

### Q9: Where does AI play a role in your coaching feedback?
**Answer:**
> *"Coach Lens uses a two-tier coaching advisor in `coach.py`: Tier 1 is a 100% reliable, zero-failure deterministic rule engine that maps measured metric gaps directly to sports-science advice. Tier 2 is an optional LLM interface using Google's Gemini free-tier API, which rephrases the exact measured degrees into friendly, conversational coaching advice. If the internet or API is unavailable, it automatically falls back to the deterministic rules with zero downtime."*

### Q10: How did you evaluate the system since Kaggle had no ground-truth form labels?
**Answer:**
> *"We were academically honest that the Kaggle dataset contains no good/bad form labels. In Module 6, we designed a controlled synthetic benchmark of 20 repetitions with known ground-truth defects (textbook reps, shallow squats, chest collapses). On this benchmark, our scoring engine achieved an ROC-AUC of 1.000 and an F1-score of 90.9%. Furthermore, we created a structured data collection protocol (`teammate_form_labeling_sheet.csv`) for our team to record and label our own multi-angle video dataset."*

---

## 📋 4. Checklist for Review Day
- [x] Python dependencies installed (`pip install -r requirements.txt`)
- [x] Golden reference template generated (`data/templates/squat_template.json`)
- [x] End-to-end CLI demo tested and passing (`python demo.py`)
- [x] Streamlit web dashboard ready to launch (`streamlit run app.py`)
- [x] Visual telemetry plots generated in `data/outputs/`
- [x] Benchmark reports and labeling sheets generated in `data/reports/`
- [x] README.md updated with full architectural diagrams and formulas

