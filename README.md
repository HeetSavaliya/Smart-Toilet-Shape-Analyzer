# 🧠 AI Rim & Shape Analyzer — v1.0

**AI-powered toilet image segmentation and rim measurement tool**

---

## 🚀 Overview

This web-based app performs smart toilet rim analysis using AI segmentation models and ellipse fitting. It's designed to process 3 images of a toilet setup and calculate precise measurements and error analysis using a reference object.

---

## 📸 How It Works

1. **Correct Orientation Instructions**  
   When you open the app, you’ll first see guidance on how to orient your images properly before uploading.

2. **Upload Images**  
   You need to upload **three specific images**:
   - 📷 Open toilet **without seat**
   - 📷 Open toilet **with seat**
   - 📷 Closed toilet

3. **Run AI Segmentation & Analysis**  
   Click the **“Let AI Do The Work”** button:
   - The app runs segmentation models to generate accurate binary masks of the toilet components.
   - Reference object (e.g., coin or matchbox) is detected and used to calculate the **pixel-to-cm scale**.
   - Ellipse fitting is applied to extract rim boundaries.
   - Key measurements (a–h) are computed based on your analysis needs.

4. **Error Analysis**  
   - You’ll see 8 predicted values labeled **a to h**.
   - Enter the **actual ground truth** values for each.
   - Click the **“Start Error Analysis”** button.
   - The app displays:
     - 📊 Individual errors for each measurement on the left.
     - 📈 A corresponding error graph on the right.
     - 🧮 **Average error percentage** at the bottom.

5. **Download Full Report**  
   - Once analysis is complete, a link is generated to download a `.zip` archive.
   - This zip contains:
     - All processed images
     - Measurements and predicted vs actual values
     - Error analysis graphs and summary

---

## 🔖 Version

This is **Version 1.0** of the app.

---

## 🛠️ Tech Stack

- Python
- Gradio (for web UI)
- PyTorch + Segmentation Models
- OpenCV & NumPy
- Ellipse fitting algorithms

---

## 📁 Folder Structure

root/
│
├── app.py # Main app logic (Gradio)
├── static/ # All images are stored in this static folder
      │
      ├──  im1.jpg # first image that depicts wrong orientation
      ├──  im2.jpg # second image that depicts wrong orientation
      ├──  im3.jpg # third image that depicts wrong orientation
      ├──  im4.jpg # fourth image that depicts correct orientation
      ├──  reference.jpg # this image is a reference to see which letter corresponds to what dimensions

---

## 📬 Feedback

Feel free to contact me at savaliyaheet19@gmail.com to know more about this project.

---
