# 🏋️ AI Gym Form Checker

An AI-powered fitness assistant that uses Computer Vision and Deep Learning to analyze workout videos, evaluate exercise form, count repetitions, and generate annotated output videos with visual feedback.

The project leverages YOLOv8 Pose Estimation to track body keypoints and monitor exercise movements, helping users improve workout quality and reduce injury risks.

---

# 🌐 Live Application

**Streamlit Cloud Deployment**

https://rvejvewkrdwuzxraoyddip.streamlit.app/

---

# 📌 Problem Statement

Many fitness enthusiasts and beginners perform exercises incorrectly without realizing it. Poor exercise form can lead to:

* Reduced workout effectiveness
* Muscle imbalances
* Joint stress
* Shoulder injuries
* Lower back pain
* Increased risk of long-term injury

Although personal trainers can help correct form, they are not always available or affordable.

This project aims to solve this problem by providing an AI-powered workout analysis system capable of evaluating exercise form directly from uploaded workout videos.

---

# 💡 Proposed Solution

The AI Gym Form Checker uses Human Pose Estimation to analyze workout videos frame-by-frame.

The system:

1. Detects body keypoints using YOLOv8 Pose Estimation.
2. Calculates joint angles.
3. Tracks exercise movement patterns.
4. Counts repetitions automatically.
5. Evaluates posture and form.
6. Generates annotated output videos.
7. Captures screenshots for review.

This allows users to receive visual feedback without requiring wearable devices or professional supervision.

---

# 🎯 Project Objectives

* Detect human body posture using AI.
* Analyze exercise movement patterns.
* Count exercise repetitions automatically.
* Monitor exercise form.
* Generate visual workout analysis.
* Create a user-friendly web application.
* Demonstrate practical use of Computer Vision in fitness applications.

---

# 🏗️ System Architecture

```text
User Uploads Video
        │
        ▼
Streamlit Web Interface
        │
        ▼
YOLOv8 Pose Detection
        │
        ▼
Body Keypoint Extraction
        │
        ▼
Joint Angle Calculation
        │
        ▼
Exercise Analysis Engine
        │
 ┌──────┴──────┐
 ▼             ▼
Rep Counter   Form Checker
        │
        ▼
Annotated Video Generation
        │
        ▼
Screenshots & Downloadable Results
```

---

# ⚙️ Tech Stack

## Frontend

* Streamlit

## Backend

* Python

## Computer Vision

* OpenCV
* YOLOv8 Pose Estimation

## Deep Learning

* PyTorch
* Ultralytics

## Data Processing

* NumPy

## Deployment

* GitHub
* Streamlit Community Cloud

---

# 🏋️ Supported Exercises

## Dumbbell Exercises

### Bicep Curl

* Tracks elbow movement
* Counts repetitions
* Monitors arm angle

### Hammer Curl

* Tracks arm motion
* Counts repetitions
* Monitors exercise stages

### Shoulder Press

* Tracks shoulder extension
* Counts repetitions
* Evaluates pressing motion

---

## Core Exercise

### Plank

* Tracks shoulder, hip, and knee alignment
* Monitors body posture
* Detects incorrect plank positioning

---

# 🔄 Workflow

### Step 1

User uploads a workout video.

### Step 2

The video is temporarily stored for processing.

### Step 3

YOLOv8 Pose Model loads.

### Step 4

Each video frame is analyzed.

### Step 5

Body keypoints are extracted.

### Step 6

Joint angles are calculated.

### Step 7

Exercise logic determines movement stages.

### Step 8

Repetitions are counted.

### Step 9

Visual annotations are added.

### Step 10

Output video is generated.

### Step 11

Screenshots are captured.

### Step 12

Results are displayed and made available for download.

---

# 📂 Project Structure

```text
Gym_AI_Assistant/
│
├── app.py
├── main.py
├── plank_main.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── models/
│   └── yolov8n-pose.pt
│
├── output_dumbbell/
├── screenshots_dumbbell/
├── output_plank/
├── screenshots_plank/
│
└── videos/
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/soumodip615c/gym_assistant_ai.git

cd gym_assistant_ai
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

```bash
streamlit run app.py
```

The application will launch automatically in your browser.

---

# 🌐 Using the Online Application

Visit:

https://rvejvewkrdwuzxraoyddip.streamlit.app/

### Instructions

1. Upload a workout video.
2. Select analysis type.
3. Select exercise.
4. Click **Analyze Workout**.
5. Wait for processing.
6. Download the generated result.
7. Review screenshots and analysis.

---

# 📸 Output Features

The application generates:

* Rep Counting
* Exercise Stage Detection
* Form Analysis
* Pose Tracking
* Annotated Workout Videos
* Automatic Screenshots
* Downloadable Results

---

# ⚠️ Important Note

Due to browser and cloud video codec limitations, the processed output video may not always play directly inside the Streamlit web interface.

If the video preview is unavailable:

1. Click **Download Result**
2. Save the generated video locally
3. Open it using:

   * VLC Media Player
   * Windows Media Player
   * MPC-HC

The downloaded output video contains the complete analysis and annotations.

---

# 📈 Project Impact

## For Fitness Beginners

Provides workout guidance without requiring a personal trainer.

## For Home Workouts

Allows exercise monitoring using only a recorded video.

## For Fitness Coaches

Provides a visual tool for exercise analysis.

## For Computer Vision Learning

Demonstrates practical applications of:

* Human Pose Estimation
* Deep Learning
* Video Analytics
* AI-Assisted Coaching

---

# 🔮 Future Enhancements

* Squat Detection
* Push-Up Analysis
* Deadlift Analysis
* Lunge Analysis
* Pull-Up Detection
* Real-Time Webcam Support
* AI Voice Feedback
* FastAPI Backend
* PostgreSQL Database Integration
* User Authentication
* Workout History Tracking
* Progress Analytics Dashboard

---

# 🧠 Concepts Demonstrated

* Computer Vision
* Deep Learning
* Human Pose Estimation
* Fitness Analytics
* Video Processing
* Exercise Tracking
* AI-Assisted Coaching

---

# 👨‍💻 Developer

**Soumodip Ghosh**

GitHub:
https://github.com/soumodip615c

---

# 📄 License

This project is licensed under the MIT License License.
