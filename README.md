# 📸 Camera Distortion Calibrator

A **robust single-image camera calibration pipeline** that estimates radial distortion parameters from a planar rectangular grid pattern (e.g., tiled floor, checkerboard).  
Implements the **Brown–Conrady distortion model** with normalized coordinate optimization for numerical stability.

---

## 🎯 Features

- ✅ **Single Image Calibration** – Estimates distortion from just one image of a planar grid  
- ✅ **Robust Pipeline** – RANSAC-based outlier removal for noisy detections  
- ✅ **Normalized Optimization** – Prevents numeric explosion through coordinate normalization  
- ✅ **Brown–Conrady Model** – Industry-standard radial distortion model *(k₁, k₂, k₃)*  
- ✅ **Comprehensive Output** – Undistorted image, calibration parameters, and detailed visualizations  
- ✅ **Google Colab Ready** – Run directly in browser, no local setup required  

---

## 🧩 Problem Statement

Given a single photograph of a **planar rectangular grid** captured with unknown camera parameters and radial distortion (possibly with occlusions, oblique perspective, noise, or lighting variations), estimate:

- 📈 **Radial distortion coefficients:** k₁, k₂, k₃  
- 🎯 **Camera intrinsics:** focal length, principal point  
- 🖼️ **Undistorted image reconstruction**

---

## 🚀 Quick Start (Google Colab)

1. **Open Google Colab**  
   - Go to [colab.research.google.com](https://colab.research.google.com/)  
   - Create a new notebook: **File → New Notebook**

2. **Upload Your Image**  
   - Click the 📁 **Files** icon in the left sidebar  
   - Click the 📤 **Upload** button  
   - Upload your grid image and rename it to `whoami.png` or `whoami.jpg`  

3. **Upload the Calibration Script**  
   - Copy–paste the entire code from this repo into a code cell  

4. **Run the Code Cell**  
   - Execute to get distortion parameters and undistorted output  

---

💡 *Fully compatible with Google Colab — no local setup needed!*

📸 Sample Results
<img width="1236" height="987" alt="image" src="https://github.com/user-attachments/assets/b694b354-e6be-4313-b4b3-706ab064b60f" />

Json Output
<img width="1396" height="807" alt="image" src="https://github.com/user-attachments/assets/8c191258-fe7d-4c98-8319-397f06c312dd" />




