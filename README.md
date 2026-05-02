# 🧠 Gait Analysis for Parkinson's Disease Detection Using AI & ML

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/MediaPipe-Pose_Estimation-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge&logo=opencv"/>
  <img src="https://img.shields.io/badge/Status-Active_Research-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Camera-Helios2+_3D-purple?style=for-the-badge"/>
</p>

<p align="center">
  <b>Research Project | Central Health Innovation (CHI), Manav Rachna University</b><br>
  <i>Oct 2024 – Dec 2025</i>
</p>

---

## 📌 Overview

This project presents an AI/ML-based non-invasive gait analysis system for **early detection of Parkinson's Disease (PD)**. The system extracts clinical gait parameters from both **2D videos** and **3D depth camera footage** (Helios2+) using **MediaPipe Pose Estimation**, **Cosine Similarity**, and **Manhattan Distance** algorithms — without any wearable sensors or clinical equipment.

Parkinson's Disease causes measurable changes in walking patterns — shorter steps, slower cadence, and longer step hold times. This system quantifies those changes automatically from video, enabling low-cost, accessible screening.

---

## 🎯 Key Clinical Finding

| Parameter | Normal Gait | PD Patient Gait | Difference |
|-----------|------------|-----------------|------------|
| Step Length | 87 – 140 cm | **40 cm** | **↓ 65%** |
| Stride Length | 167 – 299 cm | **78 cm** | **↓ 68%** |
| Step Time | 0.67 – 2.5 sec | **4.99 sec** | **↑ 4×** |

> PD patients show dramatically shorter steps and significantly slower step timing — consistent with clinical literature on Parkinsonian gait.

---

## 🏗️ System Architecture

```
Input Video (2D / 3D Helios2+)
        │
        ▼
┌─────────────────────────┐
│   Frame Extraction      │  ← OpenCV
│   RGB Preprocessing     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Pose Estimation       │  ← MediaPipe Pose
│   33 Body Landmarks     │
│   (x, y, z) per frame  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Frame Matching        │  ← Cosine Similarity
│   Reference Pose DB     │     on pose vectors
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Gait Parameter        │  ← Manhattan Distance
│   Calculation           │     + Custom Scaling
└────────────┬────────────┘
             │
             ▼
   Step Length │ Stride Length │ Step Time
   Step Hold Time │ Cadence
```

---

## ⚙️ Methodology

### 1. Pose Estimation
MediaPipe's Pose model detects **33 body landmarks** per frame including heel, knee, hip, and ankle keypoints. Landmark coordinates are normalized to `(x, y)` to ensure resolution-independence across different cameras and devices.

### 2. Frame Matching via Cosine Similarity
Two reference images are selected representing the **initial** and **final** positions of a gait cycle. Each video frame's pose vector is compared against the reference using:

$$\text{Similarity} = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|}$$

A frame is accepted as a match when its similarity score falls within a calibrated tolerance threshold `[0.3, 0.4]`.

### 3. Gait Parameter Calculation via Manhattan Distance
Once matched frames are identified, the **Manhattan Distance** between heel keypoints gives the spatial displacement:

$$D = |x_1 - x_2| + |y_1 - y_2|$$

A custom scaling factor (×1000) converts normalized coordinates to real-world centimeters, calibrated against known reference measurements.

### 4. Gait Parameters Extracted

| Parameter | Formula |
|-----------|---------|
| **Step Length** | Manhattan distance between right and left heel across matched frames |
| **Stride Length** | Manhattan distance between same heel across two full gait cycles |
| **Step Time** | `(frame_idx2 - frame_idx1) / fps` |
| **Step Hold Time** | `step_time / 2` |
| **Cadence** | `(steps / step_time) × 60` steps/min |

### 5. 3D Analysis — Helios2+ Depth Camera
For 3D footage, the system processes `.ply` point cloud files from the **Helios2+ depth camera**. A **silhouette-based heel detection** method using grid-line intersection identifies heel positions in depth space. A **color-based distance mapping** (cyan→blue = 0–200cm, step = 100cm) handles regions where depth data extends to infinity.

---

## 📊 Results

### 2D Video Analysis — 30 Normal Subjects

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Step Length | 87 cm | 140 cm | 109 cm |
| Stride Length | 167 cm | 299 cm | 221 cm |
| Step Time | 0.43 sec | 2.17 sec | 1.31 sec |
| Step Hold Time | 0.22 sec | 1.08 sec | 0.65 sec |
| Cadence | 101 | 216 steps/min | 150 steps/min |

### 3D Analysis — Helios2+ Lateral View (6 Videos)

| Video | Step Length | Stride Length | Step Time | Cadence |
|-------|------------|---------------|-----------|---------|
| Helios 1 | 104 cm | 200 cm | 0.531 sec | 138.76 steps/min |
| Helios 2 | 116 cm | 231 cm | 0.531 sec | 138.76 steps/min |
| Helios 3 | 120 cm | 234 cm | 0.531 sec | 138.76 steps/min |
| Helios 4 | 110 cm | 220 cm | 0.531 sec | 138.76 steps/min |
| Helios 5 | 119 cm | 236 cm | 0.531 sec | 138.76 steps/min |
| Helios 6 | 111 cm | 223 cm | 0.531 sec | 138.76 steps/min |

### 3D Analysis — Helios2+ Front View (40 Videos)
> Full results available in project documentation. Average step length: **112 cm**, average stride length: **226 cm**, average cadence: **137 steps/min**

---

## 🔬 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Cosine similarity threshold varies by device/camera | Calibrated per-device tolerance ranges; reference images must match recording conditions |
| MediaPipe fails on blurry frames | Applied `cv2.equalizeHist()` for contrast enhancement; set `min_detection_confidence=0.5` |
| Manhattan distance outputs float in pixel space | Custom scaling factor ×1000 with 3-decimal formatting; range clamped to 80–120cm |
| Helios depth extends to infinity in some regions | Color-based distance mapping (cyan→blue = 100cm steps); switched from Open3D to pose estimation |
| Heel silhouette detection inconsistent across frames | Heel Region Threshold method with fixed bounding grid; left heel marked blue, right heel pink |
| 3D frame matching with cosine similarity drift | Switched to motion-based cosine similarity; tolerance adjusted per-video (0.1→0.15 initial, 0.01→0.015 final) |

---

## 🤖 Helios Bot — Custom 3D Heel Detection Tool

### Why it was built
The standard MediaPipe pose estimation works well on regular 2D RGB video but **cannot directly process Helios2+ depth camera output (PLY point cloud files)**. The depth frames from Helios2+ are rendered as color-mapped silhouettes — not normal RGB images — meaning off-the-shelf tools fail to detect heel positions accurately.

`helios_bot` was built specifically to solve this: a **custom automated pipeline** that processes Helios2+ depth frames and precisely detects both heel positions using silhouette analysis and grid-line intersection.

### How it works (step by step)

```
Helios2+ Depth Frame (PLY/Image)
        │
        ▼
1. Load image → Convert to grayscale → Blur to remove noise
        │
        ▼
2. Apply threshold → Highlight body silhouette edges
        │
        ▼
3. Find largest contour → Assumed to be the person's body
        │
        ▼
4. Draw bounding box → Divide silhouette into Left / Right halves
        │
        ▼
5. Define Heel Region Thresholds at bottom of each half
        │
        ▼
6. Overlay grid of horizontal + vertical lines on image
        │
        ▼
7. Detect contour intersections with grid inside heel regions
   → Left heel intersections → marked BLUE
   → Right heel intersections → marked PINK
        │
        ▼
8. Select lowest intersection point per heel → Final heel coordinate
        │
        ▼
9. Draw colored dot at detected heel → Save annotated frame
        │
        ▼
Output: Heel coordinates (x, y) → Fed into Manhattan Distance formula
```

### Key problem it solved
When the Helios2+ camera captured regions where the person was far away, **depth values extended to infinity** — those regions appeared completely black with no point cloud data. Neither Open3D nor PCL could reconstruct these missing areas.

**Solution:** Color-based distance mapping. The Helios2+ color gradient (cyan → blue) was used to estimate distance:
- Each color step = **100 cm**
- Cyan to blue range = **0–200 cm**
- This allowed the system to estimate distances in infinity regions using color information alone, making stride length calculation possible even in problematic frames.

### What makes it unique
> Unlike standard pose estimation libraries, `helios_bot` was built ground-up for **3D depth camera data** — handling silhouette-based detection, color-mapped depth estimation, and infinity region recovery — none of which existing tools support out of the box.

---

## 🔧 How Helios Bot Was Built — Technical Deep Dive

Helios Bot was built in **two phases** across March and April 2025, each solving a different problem with Helios2+ depth camera data.

---

### Phase 1 — PLY File Processing Pipeline (March)

**Why it was needed:**
MediaPipe pose estimation works on RGB video but **completely fails on raw Helios2+ depth frames** (PLY point cloud files). MeshLab was tried first — it could open PLY files but crashed when applying denoising scripts and couldn't convert to PNG reliably. Open3D was found as the solution.

**Libraries Used:**
| Component | Tool |
|-----------|------|
| 3D point cloud processing | Open3D |
| Background noise removal | Black point filtering (hue threshold = 0.88) |
| Point cloud conversion | NumPy arrays (XYZ + RGB) |
| Image capture from 3D render | Open3D Visualizer |
| Video generation from frames | OpenCV (`cv2.VideoWriter`) |
| File handling | `os`, `glob` |

**Process Flowchart:**

![Helios Bot Process Flowchart](helios_bot_flowchart.png)

> The flowchart above shows the complete Helios Bot pipeline: from loading raw PLY files, validating color data, converting to NumPy arrays, removing black background points, rendering in Open3D, adjusting camera orientation, capturing screenshots, and finally generating an MP4 video from all processed frames.

---

### Phase 2 — Silhouette Heel Detection (April)

**Why it was needed:**
After converting PLY → PNG, MediaPipe still couldn't reliably detect heel keypoints on depth-colored silhouettes. A custom heel detection algorithm was built from scratch.

**Libraries Used:**
| Component | Tool |
|-----------|------|
| Image processing | OpenCV (`cv2`) |
| Noise reduction | Gaussian Blur |
| Silhouette extraction | Binary Thresholding |
| Body contour detection | `cv2.findContours` |
| Grid overlay | `cv2.line` |
| Heel marking | `cv2.circle` + `cv2.putText` |
| Coordinates | NumPy |

**Step-by-Step Build Logic:**
```
STEP 1 — Load & Validate
  → Load Helios2+ depth frame
  → Convert BGR → Grayscale → Gaussian Blur (remove sensor noise)
  → Apply Binary Threshold → highlight body silhouette edges

STEP 2 — Contour Detection
  → cv2.findContours() on thresholded image
  → Select LARGEST contour = person's body
  → Draw bounding rectangle around body

STEP 3 — Split Body Left / Right
  → Divide bounding box into LEFT half (left leg) and RIGHT half (right leg)
  → Define Heel Region Threshold:
     Bottom 15-20% of each half = heel zone only
     (ignores knees, hips, upper body completely)

STEP 4 — Grid Line Overlay
  → Draw horizontal + vertical grid lines across full image
  → Fixed pixel spacing using cv2.line()

STEP 5 — Intersection Detection (Core Logic)
  → For every grid intersection point (x, y):
     Check if point is INSIDE heel region boundary
     Check if point lies ON or NEAR body contour
  → If both true:
     Left heel region  → mark BLUE
     Right heel region → mark PINK

STEP 6 — Final Heel Point Selection
  → From all intersections per heel:
     Select LOWEST Y-coordinate = deepest point = actual heel
  → Draw colored dot at final position:
     Left heel  = BLUE dot
     Right heel = PINK dot

STEP 7 — Output
  → Save annotated frame to output folder
  → Print heel coordinates (x, y) to console
  → Pass coordinates → Manhattan Distance → step/stride length
```

### Infinity Region Recovery (Color-Based Distance Mapping)

When Helios2+ depth data extended to infinity, regions appeared completely black — Open3D and PCL both failed to reconstruct these areas.

```
Problem:
  Infinity regions = no depth data = reconstruction impossible

Solution — Color-based distance mapping:
  Helios2+ encodes depth as color gradient (cyan → blue)
  Each color band = 100 cm distance step
  Cyan  = 0–100 cm (close range)
  Blue  = 100–200 cm (far range)

Implementation:
  → Detect color range using HSV masking
  → Map detected color → estimated distance in cm
  → Use as substitute depth value
  → Feed into Manhattan Distance for stride calculation
```

### Key Engineering Decisions

| Problem | Failed Approach | Final Solution |
|---------|----------------|----------------|
| PLY → image conversion | MeshLab (crashed) | Open3D |
| Heel detection on depth frames | Standard MediaPipe | Custom grid-intersection algorithm |
| Unequal leg splitting | Simple midpoint division | Heel Region Threshold |
| Infinity depth regions | Open3D/PCL reconstruction | Color-based distance mapping |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Pose Estimation | MediaPipe Pose (33 landmarks) |
| Computer Vision | OpenCV (frame extraction, preprocessing, equalizeHist) |
| 3D Point Cloud | Helios2+ camera, PLY file processing |
| Numerical Computing | NumPy, Pandas |
| Algorithms | Cosine Similarity, Manhattan Distance |
| Camera Calibration | Custom scaling factors, coordinate normalization |
| Language | Python 3.8+ |

---

## 📁 Repository Structure

```
gait-analysis-parkinsons/
│
├── README.md                  ← This file
├── results/
│   ├── 2D_normal_analysis.csv         ← 30 subject results
│   ├── 3D_lateral_analysis.csv        ← Helios lateral view results
│   └── 3D_frontview_analysis.csv      ← Helios front view results (40 videos)
├── docs/
│   ├── methodology.md                 ← Detailed algorithm documentation
│   ├── algorithm_flowchart.png        ← System pipeline diagram
│   └── monthly_reports/               ← Research progress reports
└── samples/
    └── output_frames/                 ← Sample annotated output frames
```

> ⚠️ **Source code is not publicly available** to protect ongoing research. Code is available **upon request** for verified academic and research collaborations.

---

## 📬 Request Code Access

This project is part of active research at Central Health Innovation (CHI), Manav Rachna University. If you are a **researcher or professor** interested in collaboration or code access, please reach out:

**Ayon Dandapath**
- 📧 ayondandap15@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/ayon-dandapath)
- 🐙 [GitHub](https://github.com/AyonDandapath)

Please include your institution, research context, and intended use in your message.

---

## 📚 References

1. Human Pose Estimation for Clinical Analysis of Gait Pathologies
2. Knee Flexion/Extension Angle Measurement for Gait Analysis using MediaPipe
3. MediaPipe Pose Landmarker Documentation — https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
4. Assessment of Gait — https://musculoskeletalkey.com/assessment-of-gait/
5. Gait Disorders — https://my.clevelandclinic.org/health/diseases/21092-gait-disorders

---

## ⚖️ License

This project is associated with ongoing clinical research at Manav Rachna University. All rights reserved. Code and data are not for redistribution without explicit written permission.

---

<p align="center">
  Made with ❤️ for Parkinson's Disease Research<br>
  <i>Central Health Innovation (CHI) | Manav Rachna University | 2024–2025</i>
</p>
