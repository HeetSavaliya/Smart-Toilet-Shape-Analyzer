# 🧠 AI Rim & Shape Analyzer — v1.1

- minor ui changes from v1.0 and added auth token so that only authorised users can see this as a website.

**AI-powered toilet image segmentation and rim measurement tool**

---

# Try using this app here [🧠 Smart Toilet Shape Analyzer](https://huggingface.co/spaces/HeetSavaliya/Final_App)

## 🚀 Overview

This web-based app performs smart toilet rim analysis using AI segmentation models and ellipse fitting. It's designed to process 3 images of a toilet setup and calculate precise measurements and error analysis using a reference object.

---

## 📸 How It Works

1. **Correct Orientation Instructions**  
   When you open the app, you’ll first see guidance on how to orient your images properly before uploading.
   <img width="1876" height="840" alt="image" src="https://github.com/user-attachments/assets/352c8886-7af1-4292-bccd-72dec7501373" />

3. **Upload Images**  
   You need to upload **three specific images**:
   - 📷 Open toilet **without seat**
   - 📷 Open toilet **with seat**
   - 📷 Closed toilet
<img width="1829" height="608" alt="image" src="https://github.com/user-attachments/assets/1ee5280b-7e89-4617-aa91-ab795a074baf" />

4. **Run AI Segmentation & Analysis**  
   Click the **“Let AI Do The Work”** button:
   - The app runs segmentation models to generate accurate binary masks of the toilet components.
   - Reference object (e.g., coin or matchbox) is detected and used to calculate the **pixel-to-cm scale**.
   - Ellipse fitting is applied to extract rim boundaries.
   - Key measurements (a–h) are computed based on your analysis needs.
<img width="265" height="90" alt="image" src="https://github.com/user-attachments/assets/69aa5fee-eba2-4b1d-8ad3-6919f797874b" />

5. **Error Analysis**  
   - You’ll see 8 predicted values labeled **a to h**.
   - Refer to the reference diagram to know what each letter represents.
   - Select the mode (in cm/ in inches) that you prefer.
   - Enter the **actual/true** values for each.
   - Click the **“Start Error Analysis”** button.
   - The app displays:
     - 📊 Individual errors for each measurement on the left.
     - 📈 A corresponding error graph on the right.
     - 🧮 **Average error percentage** at the bottom.
<img width="343" height="400" alt="image" src="https://github.com/user-attachments/assets/116e1f80-072d-4c98-9f28-f98f09c4d313" />
<img width="1845" height="680" alt="image" src="https://github.com/user-attachments/assets/9f2295f9-4a5e-4785-8587-5f30dfa66782" />
<img width="1838" height="606" alt="image" src="https://github.com/user-attachments/assets/88a3e4b8-1612-4642-be93-fea7ddd8218e" />
<img width="1836" height="859" alt="image" src="https://github.com/user-attachments/assets/1dfda791-f526-4126-b714-1e1fc69c7e9a" />

6. **Download Full Report**  
   - Once analysis is complete, a link is generated to download a `.zip` archive.
   - This zip contains:
     - All processed images
     - Measurements and predicted vs actual values
     - Error analysis graphs and summary
<img width="1837" height="666" alt="image" src="https://github.com/user-attachments/assets/18a2d4ac-c1e2-49c6-8188-1ca16bed783d" />

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
```
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
```
---

## 📬 Feedback

Feel free to contact me at savaliyaheet19@gmail.com to know more about this project.

      © 2025 HeetSavaliya

---
