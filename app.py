# ================================
# 🔁 PART 1: Silent Package Install and Library Imports
# ================================

import subprocess, sys

def install_packages_p1():
    packages_p1 = [
        "segmentation_models_pytorch", "opencv-python-headless", "matplotlib",
        "numpy", "torch", "torchvision", "albumentations", "gradio"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", *packages_p1],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

install_packages_p1()
print("✅ Downloaded all necessary libraries successfully (silent mode)")

import os, cv2, numpy as np, torch, urllib.request
import albumentations as A
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
from PIL import Image as PILImage_p3
from albumentations.pytorch import ToTensorV2
import gradio as gr
from torch import tensor

# ================================ 
# 🧠 PART 2: Download and Load All Segmentation Models
# ================================

# 👇 Define GitHub URLs for models
model_urls_p2 = {
    "toilet_holes": "https://github.com/HeetSavaliya/Toilet-Holes-Segmentation-Model/raw/main/toilet_holes_segmentation_model.pth",
    "toilet_rim": "https://github.com/HeetSavaliya/Toilet-Rim-Segmentation-Model/raw/main/toilet_rim_segmentation_model.pth",
    "coin_5": "https://github.com/HeetSavaliya/5-Rupee-Coin-Segmentation-Model/raw/main/5coin_segmentation_model.pth"
}

# 👇 Define filenames to save locally
model_paths_p2 = {
    "toilet_holes": "toilet_holes_segmentation_model.pth",
    "toilet_rim": "toilet_rim_segmentation_model.pth",
    "coin_5": "5coin_segmentation_model.pth"
}

# 👇 Download function
def download_model_if_needed_p2(url, filename):
    if not os.path.exists(filename):
        print(f"⬇️ Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded: {filename}")
    else:
        print(f"🟢 Found cached model: {filename}")

# 👇 Load all models and return them
def load_models_p2():
    for key in model_paths_p2:
        download_model_if_needed_p2(model_urls_p2[key], model_paths_p2[key])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_holes = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
    model_holes.load_state_dict(torch.load(model_paths_p2["toilet_holes"], map_location=device))
    model_holes.to(device).eval()
    print("✅ Loaded: Toilet Holes Identification")

    model_rim = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
    model_rim.load_state_dict(torch.load(model_paths_p2["toilet_rim"], map_location=device))
    model_rim.to(device).eval()
    print("✅ Loaded: Toilet Rim Identification")

    model_coinref = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
    model_coinref.load_state_dict(torch.load(model_paths_p2["coin_5"], map_location=device))
    model_coinref.to(device).eval()
    print("✅ Loaded: 5 Coin Identification")

    return {
        "device": device,
        "model_holes": model_holes,
        "model_rim": model_rim,
        "model_coinref": model_coinref
    }

# Load models once when the script starts
print("Loading models...")
global_models_and_device = load_models_p2()
print("Models loaded successfully.")

# Extract individual components for easier access
GLOBAL_DEVICE = global_models_and_device["device"]
GLOBAL_HOLES = global_models_and_device["model_holes"]
GLOBAL_RIM = global_models_and_device["model_rim"]
GLOBAL_COIN = global_models_and_device["model_coinref"]
models_dict = global_models_and_device

# ================================
# 📸 PART 3: Upload, Rotate & Store (Multi-user Safe)
# ================================

from PIL import Image as PILImage
import numpy as np

def rotate_image_p3(image_p3, angle_p3):
    if image_p3 is None:
        return None
    
    if isinstance(image_p3, np.ndarray):
        image_p3 = PILImage.fromarray(image_p3)
    
    elif hasattr(image_p3, 'read'):  # file-like object
        image_p3 = PILImage.open(image_p3)
    
    return image_p3.rotate(-angle_p3, expand=True)

def submit_and_store_all_images_p3(img1, angle1, img2, angle2, img3, angle3):
    image_rotated1 = rotate_image_p3(img1, angle1)
    image_rotated2 = rotate_image_p3(img2, angle2)
    image_rotated3 = rotate_image_p3(img3, angle3)
    return image_rotated1, image_rotated2, image_rotated3, angle1, angle2, angle3

import gradio as gr
import os
import uuid
from PIL import Image
from datetime import datetime
import numpy as np

BASE_DIR = "user_uploads"
os.makedirs(BASE_DIR, exist_ok=True)

def init_session():
    session_id = str(uuid.uuid4())[:8]  # short unique ID
    session_path = os.path.join(BASE_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    return session_id

# 💾 Save uploaded images to the session folder
def handle_upload(img1, img2, img3, session_id):
    saved_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if img1 is not None:
        if isinstance(img1, np.ndarray):
            img1 = PILImage.fromarray(img1)
        filename = f"{timestamp}_noseat.png"
        file_path = os.path.join(BASE_DIR, session_id, filename)
        img1.save(file_path)
        saved_files.append(filename)
    
    if img2 is not None:
        if isinstance(img2, np.ndarray):
            img2 = PILImage.fromarray(img2)
        filename = f"{timestamp}_seat.png"
        file_path = os.path.join(BASE_DIR, session_id, filename)
        img2.save(file_path)
        saved_files.append(filename)
    
    if img3 is not None:
        if isinstance(img3, np.ndarray):
            img3 = PILImage.fromarray(img3)
        filename = f"{timestamp}_closed.png"
        file_path = os.path.join(BASE_DIR, session_id, filename)
        img3.save(file_path)
        saved_files.append(filename)

    if saved_files:
        return f"✅ Saved: {', '.join(saved_files)}"
    else:
        return "⚠️ No images were uploaded."

# ================================
# 🎯 PART 4: Segmentation & Overlay (Multi-user Safe)
# ================================

from torchvision import transforms
from PIL import Image as PILImage_p4
from io import BytesIO

transform_p4 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_mask_p4(model, device, image_pil):
    if image_pil is None: raise ValueError("❌ Image missing!")
    image_np = np.array(image_pil)
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        raise ValueError("❌ Must be RGB image")
    
    device2 = 'cuda' if torch.cuda.is_available() else 'cpu'


    original_size = (image_np.shape[1], image_np.shape[0])
    resized = cv2.resize(image_np, (256, 256))
    tensor = transform_p4(resized).unsqueeze(0).to(device2)
    with torch.no_grad():
        out = model(tensor)
        pred = torch.sigmoid(out).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8)
        return cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

def create_overlay_p4(image_pil, mask, color=(255, 0, 0)):
    image_np = np.array(image_pil).copy()
    overlay = image_np.copy()
    overlay[mask == 1] = color
    return overlay

def segment_and_overlay_all_p4(img1, img2, img3, model_holes_p2, model_rim_p2, model_coinref_p2, device_p2):
    
    # Get models and device from state
    model_coin = model_coinref_p2
    model_holes = model_holes_p2
    model_rim = model_rim_p2
    device = device_p2

    image_dict = {
        'open_noseat': img1,
        'open_seat': img2,
        'closed': img3
    }

    binary_masks = {'ref': {}, 'holes': {}, 'rim': {}}
    grid = []

    for key in ['open_noseat', 'open_seat', 'closed']:
        img = image_dict[key]
        mask_ref = predict_mask_p4(model_coin, device, img)
        mask_holes = predict_mask_p4(model_holes, device, img)
        mask_rim = predict_mask_p4(model_rim, device, img)

        binary_masks['ref'][key] = mask_ref
        binary_masks['holes'][key] = mask_holes
        binary_masks['rim'][key] = mask_rim

        overlay_ref = create_overlay_p4(img, mask_ref, (0, 255, 0))
        overlay_holes = create_overlay_p4(img, mask_holes, (255, 0, 0))
        overlay_rim = create_overlay_p4(img, mask_rim, (0, 0, 255))

        grid.append([np.array(img), overlay_ref, overlay_holes, overlay_rim])

    # Create overlay image grid
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    titles = ["Original", "Ref Coin", "Holes", "Rim"]
    rows = ["No Seat", "With Seat", "Closed"]

    for i in range(3):
        for j in range(4):
            axes[i][j].imshow(grid[i][j])
            axes[i][j].axis('off')
            if i == 0:
                axes[i][j].set_title(titles[j])
        axes[i][0].text(-50, 128, rows[i], rotation=90, va='center')

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    overlay_grid_image = PILImage_p4.open(buf)

    return overlay_grid_image, binary_masks, image_dict

# ================================
# 💰 PART 5: ₹5 Coin Detection & px/cm Ratio (Multi-user Safe)
# ================================

from PIL import Image as PILImage_p5
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gradio as gr

reduce_radius_px_p5 = 1
real_diameter_cm_p5 = 2.3

def detect_and_plot_reference_p5(image_dict, mask_dict):
    ref_ratios = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, key in enumerate(['open_noseat', 'open_seat', 'closed']):
        image = np.array(image_dict[key]).copy()
        mask = mask_dict['ref'][key]

        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest)
            radius = max(radius - reduce_radius_px_p5, 0)
            cv2.circle(image, (int(x) - 2, int(y) - 1), int(radius), (0, 255, 0), 2)
            ref_ratios[key] = (2 * radius) / real_diameter_cm_p5
        else:
            ref_ratios[key] = None

        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f"{key}\n{ref_ratios[key]:.2f} px/cm" if ref_ratios[key] else f"{key}\nNot Detected")

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    result_img = PILImage_p5.open(buf)
    
    ref_str = (
        f"Open (No Seat):\n{ref_ratios['open_noseat']:4.2f}px/cm\n"
        f"\nOpen (With Seat):\n{ref_ratios['open_seat']:4.2f}px/cm\n"
        f"\nClosed:\n{ref_ratios['closed']:4.2f}px/cm"
    )

    return gr.update(value=result_img, visible=True), ref_ratios, gr.update(value=ref_str, visible=True)

# ================================
# 📐 PART 6: Rim Measurement + Visualization (Multi-user Safe)
# ================================

from PIL import Image as PILImage_p6
from io import BytesIO
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def analyze_rim_intersections_p6(image_dict, mask_dict, ref_ratios):
    image_key = 'open_seat'
    mask = mask_dict['rim'][image_key]
    mask_u8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("❌ Need both inner and outer contours.")
    
    outer, inner = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    M = cv2.moments(inner)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = np.array([cx, cy])

    # 🟢 Use outer ellipse for angle, and ensure major axis is down
    if len(outer) < 5:
        raise ValueError("❌ Outer contour too small for ellipse fitting.")
    ellipse = cv2.fitEllipse(outer)
    (xc, yc), (MA, ma), angle = ellipse  # MA = major axis length, ma = minor

    if MA < ma:
        angle += 90  # major axis must be the longer one, rotate if not

    angle_rad = np.deg2rad(angle)
    dir_down = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    if dir_down[1] < 0:
        dir_down *= -1  # ensure it points downward

    dir_right = np.array([-dir_down[1], dir_down[0]])
    if dir_right[0] < 0:
        dir_right *= -1

    def get_intersections(mask, center, direction, max_steps=1000):
        prev = mask[int(center[1]), int(center[0])]
        outer_pt, inner_pt = None, None
        for step in range(1, max_steps):
            pt = center + step * direction
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                break
            val = mask[y, x]
            if outer_pt is None and val == 1 and prev == 0:
                outer_pt = np.array([x, y])
            elif outer_pt is not None and val == 0 and prev == 1:
                inner_pt = np.array([x, y])
                break
            prev = val
        return inner_pt, outer_pt

    inner_down, outer_down = get_intersections(mask, center, dir_down)
    inner_right, outer_right = get_intersections(mask, center, dir_right)

    def dist(a, b): return np.linalg.norm(a - b) if a is not None and b is not None else None

    d_down_inner_px = dist(center, inner_down)
    d_down_outer_px = dist(center, outer_down)
    d_right_inner_px = dist(center, inner_right)
    d_right_outer_px = dist(center, outer_right)

    px_per_cm = ref_ratios[image_key]
    to_cm = lambda px: px / px_per_cm if px is not None else None
    to_in = lambda px: px / px_per_cm / 2.54 if px is not None else None
    fmt = lambda val: f"{val:.2f}" if val else "N/A"

    # Measurements
    rim_cm = {
        "down_inner": to_cm(d_down_inner_px),
        "down_outer": to_cm(d_down_outer_px),
        "right_inner": to_cm(d_right_inner_px),
        "right_outer": to_cm(d_right_outer_px),
    }
    rim_inch = {
        "down_inner": to_in(d_down_inner_px),
        "down_outer": to_in(d_down_outer_px),
        "right_inner": to_in(d_right_inner_px),
        "right_outer": to_in(d_right_outer_px),
    }
    rim_ui = {
        "Down Inner": f"{fmt(rim_cm['down_inner'])} cm | {fmt(rim_inch['down_inner'])} in",
        "Down Outer": f"{fmt(rim_cm['down_outer'])} cm | {fmt(rim_inch['down_outer'])} in",
        "Right Inner": f"{fmt(rim_cm['right_inner'])} cm | {fmt(rim_inch['right_inner'])} in",
        "Right Outer": f"{fmt(rim_cm['right_outer'])} cm | {fmt(rim_inch['right_outer'])} in",
    }

    # Visualization
    pil_img = image_dict[image_key]
    if pil_img is None:
        raise ValueError("❌ No image found for analysis!")
    image_vis = np.array(pil_img)

    cv2.circle(image_vis, center, 4, (255, 255, 0), 5)
    for pt, color in zip(
        [outer_down, inner_down, outer_right, inner_right],
        [(255, 0, 0), (0, 255, 0), (255, 0, 255), (255, 255, 0)]
    ):
        if pt is not None:
            cv2.line(image_vis, center, pt, color, 2)
            cv2.circle(image_vis, pt, 4, color, 5)
    cv2.arrowedLine(image_vis, center, (center + dir_down * 100).astype(int), (0, 255, 0), 2)
    cv2.arrowedLine(image_vis, center, (center + dir_right * 100).astype(int), (0, 0, 255), 2)

    # Labels with matplotlib
    def draw_label(pt, label, offset):
        if pt is not None:
            pos = pt + offset
            plt.text(pos[0], pos[1], label, fontsize=9, color='white',
                     ha='center', va='center',
                     bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))

    plt.figure(figsize=(6, 6))
    plt.imshow(image_vis)
    plt.title("Rim Widths from Inner Center")
    offset = np.array([0, 70])
    draw_label(inner_down, rim_ui["Down Inner"], -offset)
    draw_label(outer_down, rim_ui["Down Outer"], -offset)
    draw_label(inner_right, rim_ui["Right Inner"], -offset)
    draw_label(outer_right, rim_ui["Right Outer"], offset)
    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    vis_img = PILImage_p6.open(buf)
    
    seat_str = (
        f"Down Inner:\n{d_down_inner_px:4.1f}px | {rim_cm['down_inner']:4.2f}cm | {rim_inch['down_inner']:4.2f}in\n"
        f"\nDown Outer:\n{d_down_outer_px:4.1f}px | {rim_cm['down_outer']:4.2f}cm | {rim_inch['down_outer']:4.2f}in\n"
        f"\nRight Inner:\n{d_right_inner_px:4.1f}px | {rim_cm['right_inner']:4.2f}cm | {rim_inch['right_inner']:4.2f}in\n"
        f"\nRight Outer:\n{d_right_outer_px:4.1f}px | {rim_cm['right_outer']:4.2f}cm | {rim_inch['right_outer']:4.2f}in\n"
        f"\nDown Angle:\n{(angle - 90) % 360:4.1f}°\n"
        f"\nRight Angle:\n{angle:4.1f}°"
    )

    return gr.update(value=vis_img, visible=True), rim_ui, rim_cm, rim_inch, gr.update(value=seat_str, visible=True)

# ================================
# 📐 PART 7: Ellipse-Based Rim Width Measurement from Image (open_noseat)
# ================================

from PIL import Image as PILImage_p7
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def analyze_rim_ellipse_red_p7(image_dict, mask_dict, ref_ratios):
    image_key = 'open_noseat'
    mask = mask_dict['rim'][image_key]
    image = np.array(image_dict[image_key]).copy()
    mask_u8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("❌ Need both inner and outer contours in red-marked image.")

    outer_contour, inner_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    if len(inner_contour) < 5:
        raise ValueError("❌ Need at least 5 points to fit ellipse!")

    ellipse = cv2.fitEllipse(inner_contour)
    (center_x, center_y), (major_axis, minor_axis), angle_deg = ellipse
    angle_deg += 90
    center = np.array([int(center_x), int(center_y)])
    center_p7 = center

    # Compute direction vectors
    angle_rad = np.deg2rad(angle_deg)
    dir_down = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    if dir_down[1] < 0: dir_down *= -1
    dir_right = np.array([-dir_down[1], dir_down[0]])
    if dir_right[0] < 0: dir_right *= -1

    def get_intersections(mask, center, direction, max_steps=1000):
        outer_pt, inner_pt = None, None
        prev = mask[int(center[1]), int(center[0])]
        for step in range(1, max_steps):
            pt = center + step * direction
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                break
            val = mask[y, x]
            if outer_pt is None and val == 1 and prev == 0:
                outer_pt = np.array([x, y])
            elif outer_pt is not None and val == 0 and prev == 1:
                inner_pt = np.array([x, y])
                break
            prev = val
        return inner_pt, outer_pt

    inner_down, outer_down = get_intersections(mask, center, dir_down)
    inner_right, outer_right = get_intersections(mask, center, dir_right)

    def dist(a, b): return np.linalg.norm(a - b) if a is not None and b is not None else None
    d_down_inner = dist(center, inner_down)
    d_down_outer = dist(center, outer_down)
    d_right_inner = dist(center, inner_right)
    d_right_outer = dist(center, outer_right)

    px_per_cm = ref_ratios[image_key]
    to_cm = lambda px: px / px_per_cm if px else None
    to_in = lambda px: px / px_per_cm / 2.54 if px else None
    fmt = lambda val: f"{val:.2f}" if val else "N/A"

    rim_cm = {
        "down_inner": to_cm(d_down_inner),
        "down_outer": to_cm(d_down_outer),
        "right_inner": to_cm(d_right_inner),
        "right_outer": to_cm(d_right_outer),
    }
    rim_inch = {
        "down_inner": to_in(d_down_inner),
        "down_outer": to_in(d_down_outer),
        "right_inner": to_in(d_right_inner),
        "right_outer": to_in(d_right_outer),
    }
    rim_ui = {
        "Down Inner": f"{fmt(rim_cm['down_inner'])} cm | {fmt(rim_inch['down_inner'])} in",
        "Down Outer": f"{fmt(rim_cm['down_outer'])} cm | {fmt(rim_inch['down_outer'])} in",
        "Right Inner": f"{fmt(rim_cm['right_inner'])} cm | {fmt(rim_inch['right_inner'])} in",
        "Right Outer": f"{fmt(rim_cm['right_outer'])} cm | {fmt(rim_inch['right_outer'])} in",
    }

    # Visualization
    cv2.circle(image, center, 4, (255, 255, 0), 5)
    for pt, color in zip(
        [outer_down, inner_down, outer_right, inner_right],
        [(255, 0, 0), (0, 255, 0), (255, 0, 255), (255, 255, 0)]
    ):
        if pt is not None:
            cv2.line(image, center, pt, color, 2)
            cv2.circle(image, pt, 4, color, 5)
    cv2.arrowedLine(image, center, (center + dir_down * 100).astype(int), (0, 255, 0), 2)
    cv2.arrowedLine(image, center, (center + dir_right * 100).astype(int), (0, 0, 255), 2)

    def draw_label(pt, label, offset):
        if pt is not None:
            pos = pt + offset
            plt.text(pos[0], pos[1], label, fontsize=9, color='white',
                    ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title("Rim Intersections from Inner Center (Using Ellipse)")
    offset = np.array([0, 70])
    draw_label(inner_down, rim_ui["Down Inner"], -offset)
    draw_label(outer_down, rim_ui["Down Outer"], -offset)
    draw_label(inner_right, rim_ui["Right Inner"], -offset)
    draw_label(outer_right, rim_ui["Right Outer"], offset)
    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    vis_img = PILImage_p7.open(buf)
    
    rim_str = (
        f"Down Inner:\n{d_down_inner:4.1f}px | {rim_cm['down_inner']:4.2f}cm | {rim_inch['down_inner']:4.2f}in\n"
        f"\nDown Outer:\n{d_down_outer:4.1f}px | {rim_cm['down_outer']:4.2f}cm | {rim_inch['down_outer']:4.2f}in\n"
        f"\nRight Inner:\n{d_right_inner:4.1f}px | {rim_cm['right_inner']:4.2f}cm | {rim_inch['right_inner']:4.2f}in\n"
        f"\nRight Outer:\n{d_right_outer:4.1f}px | {rim_cm['right_outer']:4.2f}cm | {rim_inch['right_outer']:4.2f}in\n"
        f"\nDown Angle:\n{(angle_deg - 90) % 360:4.1f}°\n"
        f"\nRight Angle:\n{angle_deg:4.1f}°"
    )

    return gr.update(value=vis_img, visible=True), rim_ui, rim_cm, rim_inch, center.tolist(), dir_down.tolist(), dir_right.tolist(), gr.update(value=rim_str, visible=True)

# ================================
# 🧩 PART 8: Rim Height Analyzer (Stateless, Session-Safe)
# ================================

import gradio as gr
import numpy as np
import cv2
import math
from PIL import Image as PILImage_p8
from io import BytesIO
import matplotlib.pyplot as plt

def find_inner_top_p8(mask, center, direction, max_steps=1000):
    prev = mask[int(center[1]), int(center[0])]
    for step in range(1, max_steps):
        pt = center + step * direction
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            break
        val = mask[y, x]
        if prev == 1 and val == 0:
            return np.array([x, y])
        prev = val
    return None

def find_outer_bottom_p8(mask, center, direction, max_steps=1000):
    prev = mask[int(center[1]), int(center[0])]
    for step in range(1, max_steps):
        pt = center - step * direction
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            break
        val = mask[y, x]
        if prev == 0 and val == 1:
            return np.array([x, y])
        prev = val
    return None

def run_rim_height_analysis_p8(
    _trigger_button,
    image_dict_p4,
    binary_masks_p4,
    ref_ratios_p5,
    center_p7,
    dir_down_p7
):
    image = np.array(image_dict_p4["open_noseat"]).copy()
    mask = binary_masks_p4["rim"]["open_noseat"]
    px_per_cm = ref_ratios_p5["open_noseat"]
    center_pt = np.array(center_p7)
    dir_vec = np.array(dir_down_p7)

    raw_angle = np.rad2deg(math.atan2(dir_vec[1], dir_vec[0]))
    down_angle_deg = (450 - raw_angle) % 360
    perp_angle_deg = (down_angle_deg + 90) % 360

    inner_top = find_outer_bottom_p8(mask, center_pt, dir_vec)
    outer_bottom = find_inner_top_p8(mask, center_pt, dir_vec)

    if inner_top is None or outer_bottom is None:
        return "⚠️ Could not find both rim points", None, None, None, None, None

    rim_height_px = np.linalg.norm(outer_bottom - inner_top)
    rim_height_cm = rim_height_px / px_per_cm
    rim_height_in = rim_height_cm / 2.54

    # --- Visualization ---
    cv2.circle(image, tuple(inner_top), 5, (0, 255, 255), -1)
    cv2.circle(image, tuple(outer_bottom), 5, (255, 255, 0), -1)
    cv2.line(image, tuple(inner_top), tuple(outer_bottom), (0, 0, 255), 2)

    arrow_end = (center_pt + dir_vec * 100).astype(int)
    cv2.arrowedLine(image, center_pt.astype(int), arrow_end, (0, 255, 0), 2)

    label = f"{rim_height_px:.1f}px | {rim_height_cm:.2f}cm | {rim_height_in:.2f}in"
    mid = ((inner_top + outer_bottom) / 2).astype(int)
    text_pos = (mid[0] + 10, mid[1] - 10)
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 1, 2)
    rect_start = (text_pos[0] - 10, text_pos[1] - th - 10)
    rect_end = (text_pos[0] + tw + 10, text_pos[1] + 10)
    cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    cv2.putText(image, label, text_pos, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    buf = BytesIO()
    plt.imsave(buf, image)
    buf.seek(0)
    rim_result_image_p8 = PILImage_p8.open(buf)
    
    rim_str = (
        f"Rim Height:\n{rim_height_px:4.1f}px | {rim_height_cm:4.2f}cm | {rim_height_in:4.2f}in\n\nDown Angle:\n{down_angle_deg:4.1f}°"
        )

    return label, gr.update(value=rim_result_image_p8, visible=True), rim_height_px, rim_height_cm, rim_height_in, inner_top.tolist(), gr.update(value=rim_str, visible=True)

# ============================
# 🕳️ PART 9: Hole Width Measurement (Stateless)
# ============================

from PIL import Image as PILImage_p9
from io import BytesIO

def analyze_hole_width_perpendicular_p9(
    _trigger,
    binary_masks_p4,
    ref_ratios_p5,
    image_dict_p4,
    dir_right_p7,
    center_p7
):
    # --- Input setup ---
    mask = binary_masks_p4["holes"]["open_noseat"]
    mask_u8 = (mask * 255).astype(np.uint8)
    ys, xs = np.where(mask == 1)
    points = np.stack([xs, ys], axis=1).astype(float)

    dir_scan = np.array(dir_right_p7)
    dir_scan = dir_scan / np.linalg.norm(dir_scan)
    dir_perp = np.array([-dir_scan[1], dir_scan[0]])

    # --- Project points on perpendicular axis ---
    projs = points @ dir_perp
    proj_min, proj_max = np.min(projs), np.max(projs)

    best_dist = -1
    pt_min_p9 = None
    pt_max_p9 = None
    for offset in np.arange(proj_min, proj_max, 1.0):
        mask_line = np.abs(projs - offset) < 0.5
        line_points = points[mask_line]
        if len(line_points) >= 2:
            line_proj = line_points @ dir_scan
            i_min = np.argmin(line_proj)
            i_max = np.argmax(line_proj)
            d = np.linalg.norm(line_points[i_max] - line_points[i_min])
            if d > best_dist:
                best_dist = d
                pt_min_p9 = line_points[i_min]
                pt_max_p9 = line_points[i_max]

    if pt_min_p9 is None or pt_max_p9 is None:
        return None, None, None, None, None, None, None

    # --- Compute pixel width ---
    hole_width_px_p9 = best_dist

    # --- Orientation angle ---
    dir_line = pt_max_p9 - pt_min_p9
    dir_line = dir_line / np.linalg.norm(dir_line)
    angle_rad = np.arctan2(dir_line[1], dir_line[0])
    angle_deg_p9 = (450 - np.rad2deg(angle_rad)) % 360

    # --- Convert to real-world units ---
    px_per_cm = ref_ratios_p5["open_noseat"]
    cm_per_px = 1.0 / px_per_cm
    hole_width_cm_p9 = hole_width_px_p9 * cm_per_px
    hole_width_inch_p9 = hole_width_cm_p9 / 2.54

    # --- Visualization ---
    image = np.array(image_dict_p4["open_noseat"]).copy()
    cv2.circle(image, pt_min_p9.astype(int), 5, (0, 255, 0), -1)
    cv2.circle(image, pt_max_p9.astype(int), 5, (0, 0, 255), -1)
    cv2.line(image, pt_min_p9.astype(int), pt_max_p9.astype(int), (255, 255, 0), 2)

    label = f"{hole_width_px_p9:.1f}px | {hole_width_cm_p9:.2f}cm | {hole_width_inch_p9:.2f}in"
    mid = ((pt_min_p9 + pt_max_p9) / 2).astype(int)
    text_pos = (mid[0] - 30, mid[1] - 30)

    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 1, 2)
    rect_start = (text_pos[0] - 10, text_pos[1] - th - 10)
    rect_end = (text_pos[0] + tw + 10, text_pos[1] + 10)
    cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(image, label, text_pos, font, 1, (255, 255, 255), 2)

    # --- Return PIL image ---
    buf = BytesIO()
    plt.imsave(buf, image)
    buf.seek(0)
    hole_result_image_p9 = PILImage_p9.open(buf)
    
    hole_width_str = (
        f"Hole Width:\n{hole_width_px_p9:4.1f}px | {hole_width_cm_p9:4.2f}cm | {hole_width_inch_p9:4.2f}in\n"
        f"\nOrientation Angle:\n{angle_deg_p9:4.1f}°"
    )

    return gr.update(value=hole_result_image_p9, visible=True), hole_width_px_p9, hole_width_cm_p9, hole_width_inch_p9, angle_deg_p9, pt_min_p9.tolist(), pt_max_p9.tolist(), gr.update(value=hole_width_str, visible=True)

# =============================
# 🧩 PART 10: Rim-to-Hole Line Distance (Stateless)
# =============================

from PIL import Image as PILImage_p10
from io import BytesIO

def compute_top_to_hole_distance_p10(
    _trigger,
    inner_top_p8,
    pt_min_p9,
    pt_max_p9,
    dir_down_p7,
    ref_ratios_p5,
    image_dict_p4
):
    def line_intersection_p10(p1, d1, p2, d2):
        A = np.array([d1, -d2]).T
        b = p2 - p1
        if np.linalg.matrix_rank(A) < 2:
            return None
        t_s = np.linalg.lstsq(A, b, rcond=None)[0]
        return p1 + t_s[0] * d1

    pt_min = np.array(pt_min_p9)
    pt_max = np.array(pt_max_p9)
    inner_top = np.array(inner_top_p8)
    dir_down = np.array(dir_down_p7)

    dir_hole_line = pt_max - pt_min
    dir_hole_line = dir_hole_line / np.linalg.norm(dir_hole_line)
    intersection_point = line_intersection_p10(inner_top, dir_down, pt_min, dir_hole_line)

    if intersection_point is None:
        print("❌ Lines are parallel.")
        return None, None, None, None, None, None, None

    # --- Distance ---
    dist_px = np.linalg.norm(intersection_point - inner_top)
    px_per_cm = ref_ratios_p5["open_noseat"]
    cm_per_px = 1.0 / px_per_cm
    dist_cm = dist_px * cm_per_px
    dist_inch = dist_cm / 2.54

    # --- Angles ---
    angle_down_deg = (450 - np.rad2deg(np.arctan2(dir_down[1], dir_down[0]))) % 360
    vec = dir_hole_line
    angle_perp_deg = (450 - np.rad2deg(np.arctan2(vec[1], vec[0]))) % 360

    # --- Visualization ---
    image = np.array(image_dict_p4["open_noseat"]).copy()
    cv2.circle(image, inner_top.astype(int), 4, (0, 255, 0), 5)
    cv2.circle(image, intersection_point.astype(int), 4, (0, 0, 255), 5)
    cv2.line(image, inner_top.astype(int), intersection_point.astype(int), (255, 255, 0), 2)
    cv2.line(image, pt_min.astype(int), pt_max.astype(int), (255, 0, 255), 1)

    label = f"{dist_px:.1f}px | {dist_cm:.2f}cm | {dist_inch:.2f}in"
    mid = ((inner_top + intersection_point) / 2).astype(int)
    text_pos = (mid[0] + 10, mid[1] - 10)

    overlay = image.copy()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    rect_start = (text_pos[0] - 10, text_pos[1] - th - 10)
    rect_end = (text_pos[0] + tw + 10, text_pos[1] + 10)
    cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    buf = BytesIO()
    plt.imsave(buf, image)
    buf.seek(0)
    rim_result_image_p10 = PILImage_p10.open(buf)
    
    rim_to_hole_str = (
        f"Inner Rim to Hole Distance:\n{dist_px:4.1f}px | {dist_cm:4.2f}cm | {dist_inch:4.2f}in\n"
        f"\nRim Direction Angle (Down):\n{angle_down_deg:4.1f}°\n"
        f"\nHole Width Angle:\n{angle_perp_deg:4.1f}°"
    )

    return (
        gr.update(value=rim_result_image_p10, visible=True),
        dist_px,
        dist_cm,
        dist_inch,
        angle_down_deg,
        angle_perp_deg,
        intersection_point.tolist(),
        gr.update(value=rim_to_hole_str, visible=True)
    )

# ================================
# 🌀 PART 10: Ellipse-Based Rim Orientation (Closed Lid, Stateless)
# ================================

from PIL import Image as PILImage_p10
from io import BytesIO

def analyze_closed_rim_orientation_p10(_trigger, binary_masks_p4, image_dict_p4):
    def remove_top_based_on_angle(mask, center, angle_deg, threshold):
        mask = (mask > 0).astype(np.uint8)
        h, w = mask.shape
        cx, cy = center
        angle_rad = np.deg2rad(angle_deg)

        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        perp_dx = -dy
        perp_dy = dx

        for y in range(h):
            x_coords = np.where(mask[y] == 1)[0]
            if len(x_coords) == 0:
                continue

            distances = []
            for x in x_coords:
                px, py = x, y
                dxp = px - cx
                dyp = py - cy
                dist = abs(dxp * perp_dx + dyp * perp_dy)
                distances.append(dist)

            if max(distances) < threshold:
                mask[y, x_coords] = 0
            else:
                break

        return mask * 255

    # --- Cleaning ---
    mask = binary_masks_p4["rim"]["closed"]
    center_estimate = (150, 220)
    clean_mask = remove_top_based_on_angle(mask, center_estimate, 23, 30)

    # Update original mask
    binary_masks_p4["rim"]["closed"] = clean_mask

    # --- Ellipse Fitting ---
    mask_u8 = (clean_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert contours, "❌ No contours found in closed rim mask!"
    rim_contour = max(contours, key=cv2.contourArea)
    assert len(rim_contour) >= 5, "❌ Need at least 5 points to fit an ellipse!"
    ellipse = cv2.fitEllipse(rim_contour)

    (center_x, center_y), (major_axis, minor_axis), angle_deg = ellipse
    ellipse_center = np.array([int(center_x), int(center_y)])

    angle_deg += 90  # Align long axis as vertical
    angle_rad = math.radians(angle_deg)
    dir_down = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    dir_down /= np.linalg.norm(dir_down)
    # ✅ Ensure dir_down points downward (positive Y)
    if dir_down[1] < 0:
        dir_down *= -1

    dir_right = np.array([-dir_down[1], dir_down[0]])
    
    # ✅ Ensure dir_right points rightward (positive X)
    if dir_right[0] < 0:
        dir_right *= -1
    
    # Assuming dir_down is a 2D unit vector [x, y]
    angle_rad_back = math.atan2(dir_down[1], dir_down[0])
    angle_deg = math.degrees(angle_rad_back)

    ellipse_angle_deg = (450 - angle_deg) % 360

    # --- Visualization ---
    image = np.array(image_dict_p4["closed"]).copy()
    cv2.circle(image, ellipse_center, 4, (255, 255, 0), 5)
    pt_down = (ellipse_center + dir_down * 100).astype(int)
    pt_right = (ellipse_center + dir_right * 100).astype(int)
    cv2.arrowedLine(image, ellipse_center, pt_down, (0, 255, 0), 3)
    cv2.arrowedLine(image, ellipse_center, pt_right, (0, 255, 255), 3)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title("Toilet Rim Orientation using Ellipse Fitting")
    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    ellipse_viz_image = PILImage_p10.open(buf)
    
    closed_orientation_str = (
        f"Downward Direction Angle:\n{ellipse_angle_deg:4.1f}°\n"
        f"\nPerpendicular Direction Angle:\n{(ellipse_angle_deg + 90) % 360:4.1f}°\n"
        f"\nEllipse Center:\n({ellipse_center[0]}, {ellipse_center[1]})"
    )

    return (
        gr.update(value=ellipse_viz_image, visible=True),
        ellipse_angle_deg,
        ellipse_center.tolist(),
        dir_down.tolist(),
        dir_right.tolist(),
        binary_masks_p4,  # return updated mask
        gr.update(value=closed_orientation_str, visible=True)
    )

# ================================
# 🧮 PART 11: Rim Height Along Downward Direction (Closed Rim, Stateless)
# ================================

from PIL import Image as PILImage_p11
from io import BytesIO

def analyze_rim_height_on_closed_p11(
    _trigger,
    binary_masks_p4,
    image_closed_lid_rotated_p3,
    ellipse_angle_deg_p10,
    ref_ratios_p5,
    rim_height_cm_p8
):
    def find_extreme_points_along_line(mask, center, direction, max_steps=2000):
        H, W = mask.shape
        pt1 = pt2 = None
        for step in range(1, max_steps):
            pt = center + step * direction
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (0 <= x < W and 0 <= y < H): break
            if mask[y, x] > 0:
                pt2 = np.array([x, y])
            elif pt2 is not None:
                break
        for step in range(1, max_steps):
            pt = center - step * direction
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (0 <= x < W and 0 <= y < H): break
            if mask[y, x] > 0:
                pt1 = np.array([x, y])
            elif pt1 is not None:
                break
        return pt1, pt2

    def dist(a, b):
        return np.linalg.norm(a - b) if a is not None and b is not None else None

    # --- Mask & Image ---
    mask = binary_masks_p4["rim"]["closed"]
    image = np.array(image_closed_lid_rotated_p3)

    # --- Center ---
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert contours, "❌ No contours found in closed rim mask!"
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = np.array([cx, cy])

    # --- Direction Vector from Ellipse ---
    angle_rad = np.deg2rad((450 - ellipse_angle_deg_p10) % 360)
    dir_vec = np.array([math.cos(angle_rad), math.sin(angle_rad)])

    pt_start, pt_end = find_extreme_points_along_line(mask, center, dir_vec)
    rim_height_px = dist(pt_start, pt_end)
    px_per_cm = ref_ratios_p5['closed']
    full_rim_height_cm = rim_height_px / px_per_cm
    full_rim_height_in = full_rim_height_cm / 2.54

    rim_height_cm = rim_height_cm_p8
    rim_height_in = rim_height_cm / 2.54

    closed_remaining_cm = full_rim_height_cm - rim_height_cm
    closed_remaining_in = closed_remaining_cm / 2.54

    if pt_start is not None and pt_end is not None:
        vis = image.copy()
        cv2.circle(vis, tuple(center), 4, (255, 255, 0), -1)
        cv2.circle(vis, tuple(pt_start), 5, (0, 0, 255), -1)
        cv2.circle(vis, tuple(pt_end), 5, (0, 255, 0), -1)
        cv2.line(vis, tuple(pt_start), tuple(pt_end), (0, 255, 255), 2)

        label = f"{rim_height_px:.1f}px | {full_rim_height_cm:.2f}cm | {full_rim_height_in:.2f}in"
        mid_point = ((pt_start + pt_end) / 2).astype(int)
        text_pos = (mid_point[0] + 10, mid_point[1] - 10)
        overlay = vis.copy()
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        rect_start = (text_pos[0] - 10, text_pos[1] - text_h - 10)
        rect_end = (text_pos[0] + text_w + 10, text_pos[1] + 10)
        cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        plt.figure(figsize=(6, 6))
        plt.imshow(vis)
        plt.title("Rim Height on Closed Lid")
        plt.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        rim_height_vis = PILImage_p11.open(buf)
        
        pt_start_p11 = pt_start
        
        rim_closed_str = (
            f"Total Height:\n{rim_height_px:4.1f}px | {full_rim_height_cm:4.2f}cm | {full_rim_height_in:4.2f}in"
        )

        return (
            gr.update(value=rim_height_vis, visible=True),
            rim_height_cm,
            rim_height_in,
            full_rim_height_cm,
            full_rim_height_in,
            closed_remaining_cm,
            closed_remaining_in,
            pt_start_p11,
            gr.update(value=rim_closed_str, visible=True)
        )

    else:
        print("⚠️ Could not find valid intersection points.")
        return (None, None, None, None, None, None, None, None, None)

# ================================
# 🧱 PART 12: Draw Remaining Closed Lid Portion (Stateless)
# ================================

def draw_remaining_closed_portion_p12(
    _trigger,
    pt_start_p11,
    closed_remaining_cm_p11,
    ref_ratios_p5,
    ellipse_dir_down_p10,
    image_closed_lid_rotated_p3
):
    # --- Clone and compute points ---
    pt_start = np.array(pt_start_p11)
    dir_down = np.array(ellipse_dir_down_p10)
    remaining_cm = closed_remaining_cm_p11
    px_per_cm = ref_ratios_p5['closed']
    remaining_px = remaining_cm * px_per_cm
    pt_end = (pt_start + dir_down * remaining_px).astype(int)

    # --- Compute angle ---
    vec = pt_end - pt_start
    angle_rad = np.arctan2(vec[1], vec[0])
    remaining_angle_deg = (450 - np.rad2deg(angle_rad)) % 360

    # --- Draw on image ---
    image = np.array(image_closed_lid_rotated_p3).copy()
    cv2.line(image, tuple(pt_start), tuple(pt_end), (0, 0, 255), 3)  # red
    cv2.circle(image, tuple(pt_start), 5, (0, 255, 0), -1)  # green
    cv2.circle(image, tuple(pt_end), 5, (255, 0, 0), -1)    # blue

    # --- Add label ---
    remaining_in = remaining_cm / 2.54
    label = f"({remaining_cm:.2f} cm) | ({remaining_in:.2f} in)"
    text_pos = (pt_end[0] + 10, pt_end[1] - 10)
    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    
    closed_remaining_str = (
        f"Remaining Portion Length:\n{remaining_cm:4.2f}cm | {remaining_in:4.2f}in\n"
        f"\nDirection Angle:\n{remaining_angle_deg:4.1f}°"
    )

    return (
        gr.update(value=(PILImage_p11.fromarray(image)), visible=True),
        remaining_angle_deg,
        pt_start,
        pt_end,
        gr.update(value=closed_remaining_str, visible=True)
    )

# ================================
# 🧱 PART 13: Top Rim Width Measurement (Stateless)
# ================================

def analyze_top_rim_width_p13(
    _trigger,
    binary_masks_p4,
    ellipse_center_p10,
    ellipse_angle_deg_p10,
    ref_ratios_p5,
    image_closed_lid_rotated_p3
):
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage_p13

    mask = binary_masks_p4['rim']['closed']
    center = np.array(ellipse_center_p10)
    angle_deg = (450 - ellipse_angle_deg_p10) % 360
    px_per_cm = ref_ratios_p5['closed']
    image = np.array(image_closed_lid_rotated_p3).copy()

    def bresenham_line(x0, y0, x1, y1):
        points = []
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep: x0, y0, x1, y1 = y0, x0, y1, x1
        swapped = False
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            swapped = True
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx / 2
        ystep = 1 if y0 < y1 else -1
        y = y0
        for x in range(x0, x1 + 1):
            pt = (y, x) if steep else (x, y)
            points.append(pt)
            error -= dy
            if error < 0:
                y += ystep
                error += dx
        if swapped: points.reverse()
        return points

    def find_top_and_perpendicular_extremes(mask, center, angle_deg, perp_halfwidth=700, max_up_scan=1500):
        mask = (mask > 0).astype(np.uint8)
        h, w = mask.shape
        angle_rad = np.deg2rad(angle_deg)
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        up_dir = np.array([-dx, -dy])
        perp_dir = np.array([-dy, dx])

        pt_top = None
        for i in range(max_up_scan):
            pt = center + up_dir * i
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h and mask[y, x] == 1:
                pt_top = (x, y)
        if pt_top is None:
            raise ValueError("No mask pixel found when scanning upward from center.")

        pt_top = (pt_top[0], int(pt_top[1] + h * 0.02))
        left_pt = (int(round(pt_top[0] - perp_dir[0] * perp_halfwidth)),
                   int(round(pt_top[1] - perp_dir[1] * perp_halfwidth)))
        right_pt = (int(round(pt_top[0] + perp_dir[0] * perp_halfwidth)),
                    int(round(pt_top[1] + perp_dir[1] * perp_halfwidth)))

        line_pts = bresenham_line(left_pt[0], left_pt[1], right_pt[0], right_pt[1])
        valid_pts = [pt for pt in line_pts if 0 <= pt[0] < w and 0 <= pt[1] < h and mask[pt[1], pt[0]] == 1]
        if len(valid_pts) < 2:
            raise ValueError("Not enough mask pixels found along perpendicular line.")
        pt1, pt2 = valid_pts[0], valid_pts[-1]
        dist_px = np.linalg.norm(np.array(pt2) - np.array(pt1))
        return pt_top, pt1, pt2, dist_px

    pt_top, pt1, pt2, dist_px = find_top_and_perpendicular_extremes(mask, center, angle_deg)
    dist_cm = dist_px / px_per_cm
    dist_in = dist_cm / 2.54
    vec = np.array(pt2) - np.array(pt1)
    angle_rad_actual = np.arctan2(vec[1], vec[0])
    angle_deg_actual = (450 - np.rad2deg(angle_rad_actual)) % 360

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=1)
    ax.scatter(*pt1, color='lime', s=20)
    ax.scatter(*pt2, color='cyan', s=20)
    mid_x = (pt1[0] + pt2[0]) / 2
    mid_y = (pt1[1] + pt2[1]) / 2
    ax.text(mid_x, mid_y + 100, f"{dist_px:.1f}px | {dist_cm:.2f}cm | {dist_in:.2f}in",
            fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.6))
    ax.set_title("Top Rim Width Measurement")
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    vis_image = PILImage_p13.open(buf)
    
    top_rim_str = (
        f"Top Rim Width:\n{dist_px:4.1f}px | {dist_cm:4.2f}cm | {dist_in:4.2f}in\n"
        f"\nOrientation Angle:\n{angle_deg_actual % 180:4.1f}°"
    )

    return gr.update(value=vis_image, visible=True), pt_top, pt1, pt2, dist_px, dist_cm, dist_in, angle_deg_actual, gr.update(value=top_rim_str, visible=True)

# ================================
# 🧩 FINAL PART: Combined App Launcher
# ================================
import gradio as gr
from urllib.parse import parse_qs

SECRET_TOKEN = "3v80mdr2k3rjig98bv9mcotf89"
# SECRET_TOKEN = os.getenv("ACCESS_TOKEN", "NO_TOKEN_SET")
def check_token(request: gr.Request):
    try:
        query_str = request.request.url.query
        query = parse_qs(query_str)
        token = query.get("access", [""])[0]
        if token == SECRET_TOKEN:
            return gr.update(visible=True), gr.update(visible=False), gr.update(value="✅ Access Granted", visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True), gr.update(value="🚫 Unauthorized", visible=False)
    except Exception as e:
        print("Token check error:", e)
        return gr.update(visible=False), gr.update(visible=True), gr.update(value="❌ Invalid request", visible=False)

with gr.Blocks(title="🧪 Toilet Segmentation & Measurement App") as full_app_interface:
    status_box = gr.Textbox(label="Status", visible=False)
    protected_content = gr.Column(visible=False)
    unauthorized_message = gr.Markdown("🚫 **Unauthorized access**", visible=False)

    full_app_interface.load(
        fn=check_token,
        inputs=[],
        outputs=[protected_content, unauthorized_message, status_box]
    )

    with protected_content:
        gr.Markdown("# 🚽 Smart Toilet Image Checker", elem_id="centered-title")
        gr.Markdown("Just upload your toilet photos — then sit back and watch the AI work its magic!", elem_id="centered-title")
        gr.Markdown("📘 **New here?** If you're not sure how to use this app, please check out the [step-by-step instructions](https://drive.google.com/file/d/1FmdJBWi046iscXAXlAh321Iyc5w8RHcc/view?usp=sharing).")

        session_state = gr.State(init_session())
        
        with gr.Row():
            gr.Markdown("### 📷 How to Upload Images Correctly (Example)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("❌ Wrong")
                wrong1 = gr.Image(value="./static/im1.jpg", show_label=False, interactive=False, height=200)
            with gr.Column():
                gr.Markdown("❌ Wrong")
                wrong2 = gr.Image(value="./static/im2.jpg", show_label=False, interactive=False, height=200)
            with gr.Column():
                gr.Markdown("❌ Wrong")
                wrong3 = gr.Image(value="./static/im3.jpg", show_label=False, interactive=False, height=200)

        with gr.Row():
            with gr.Column():
                z="filler"
            with gr.Column():
                gr.Markdown("✅ Correct")
                correct = gr.Image(value="./static/im4.jpg", show_label=False, interactive=False, height=200)
            with gr.Column():
                z="filler"

        binary_masks_p4 = gr.State()
        image_dict_p4 = gr.State()
        out_ref_ratios_p5 = gr.State()
        out_rimellipse_ui_p7 = gr.State()
        out_rimellipse_cm_p7 = gr.State()
        out_rimellipse_inch_p7 = gr.State()
        inner_top_p7 = gr.State()
        dir_down_p7 = gr.State()
        dir_right_p7 = gr.State()
        btn_rimheight_p8 = gr.State()
        rimheight_text_p8 = gr.State()
        rim_height_px_p8 = gr.State()
        rim_height_cm_p8 = gr.State()
        rim_height_inch_p8 = gr.State()
        inner_top_p8 = gr.State()
        btn_measure_holewidth_p9 = gr.State()
        hole_width_px_p9 = gr.State()
        hole_width_cm_p9 = gr.State()
        hole_width_inch_p9 = gr.State()
        angle_deg_p9 = gr.State()
        pt_min_p9 = gr.State()
        pt_max_p9 = gr.State()
        btn_top_to_hole_p10 = gr.State()
        top_to_hole_line_px_p10 = gr.State()
        top_to_hole_line_cm_p10 = gr.State()
        top_to_hole_line_inch_p10 = gr.State()
        angle_down_deg_p10 = gr.State()
        angle_perp_deg_p10 = gr.State()
        intersection_point_p10 = gr.State()
        btn_ellipse_orient_p10 = gr.State()
        ellipse_angle_deg_p10 = gr.State()
        ellipse_center_p10 = gr.State()
        ellipse_dir_down_p10 = gr.State()
        ellipse_dir_right_p10 = gr.State()
        updated_binary_masks_p4 = gr.State()
        btn_rim_height_closed_p11 = gr.State()
        rim_height_cm_p11 = gr.State()
        rim_height_in_p11 = gr.State()
        full_rim_height_cm_p11 = gr.State()
        full_rim_height_in_p11 = gr.State()
        closed_remaining_cm_p11 = gr.State()
        closed_remaining_in_p11 = gr.State()
        pt_start_p11 = gr.State()
        btn_draw_remaining_p12 = gr.State()
        remaining_angle_deg_p12 = gr.State()
        pt_start_p12 = gr.State()
        pt_end_p12 = gr.State()
        btn_measure = gr.State()
        out_pt_top = gr.State()
        out_pt1 = gr.State()
        out_pt2 = gr.State()
        out_dist_px = gr.State()
        out_dist_cm = gr.State()
        out_dist_in = gr.State()
        out_angle_deg = gr.State()
        out_rim_measurements_p6 = gr.State()
        out_rim_measurements_cm_p6 = gr.State()
        out_rim_measurements_inch_p6 = gr.State()
        models_holes_p2 = gr.State(None)
        models_rim_p2 = gr.State(None)
        models_coinref_p2 = gr.State(None)
        device_p2 = gr.State(None)
        gallery_segmentation_p4 = gr.State()
        
        full_app_interface.load(
            fn=lambda: [GLOBAL_HOLES, GLOBAL_RIM, GLOBAL_COIN, device_p2],
            outputs=[models_holes_p2, models_rim_p2, models_coinref_p2, device_p2],
            queue=False  # This event usually doesn't need to be queued
        )
        
        gr.HTML("""<style>
                    #centered-title {
                        text-align: center;
                        width: 100%
                    }
                    #footer-text{
                        text-align: center;
                        font-size: 14px;
                        color: #666;
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        line-height: 1.6
                    }
                    #pretty-box textarea {
                        font-size: 16px;
                        font-family: monospace;
                        border: 2px solid #ccc;
                        border-radius: 12px;
                        padding: 10px 14px;
                        resize: none;
                        height: 50px;
                        line-height: 1.4;
                        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
                        min-width: 200px;
                        max-width: 100%;

                    }
                    
                    #pretty-box label {
                        text-align: center;
                        font-size: 18px;
                    }
                    
                    #fit-box {
                        min-width: 200px;
                        margin: auto;
                    }
                    .gr-block.gr-group {
                        background-color: inherit !important;
                        padding: 16px;
                    }
                </style>""")
        
        def rotate_image_live(img):
            if img is None:
                return None

            import numpy as np
            import cv2
            from PIL import Image

            img_np = np.array(img)
            h, w = img_np.shape[:2]

            # Transpose shape for 90-degree rotation (clockwise)
            rotated = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)

            return Image.fromarray(rotated)

        import gradio as gr
        from PIL import Image
        import os

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📸 Open Toilet (No Seat)")
                uploader1 = gr.UploadButton("Upload Image", file_types=["image"])
                input1 = gr.Image(height=500, width=800, label="Preview", interactive=False, visible=False)
                filename1 = gr.Markdown()
                rotte1 = gr.Button("Rotate", visible=False)

            with gr.Column():
                gr.Markdown("### 📸 Open Toilet (With Seat)")
                uploader2 = gr.UploadButton("Upload Image", file_types=["image"])
                input2 = gr.Image(height=500, width=800, label="Preview", interactive=False, visible=False)
                filename2 = gr.Markdown()
                rotte2 = gr.Button("Rotate", visible=False)

            with gr.Column():
                gr.Markdown("### 📸 Closed Lid Toilet")
                uploader3 = gr.UploadButton("Upload Image", file_types=["image"])
                input3 = gr.Image(height=500, width=800, label="Preview", interactive=False, visible=False)
                filename3 = gr.Markdown()
                rotte3 = gr.Button("Rotate", visible=False)
        
        def load_image_and_filename(file_obj):
            if file_obj is None:
                return None, "", gr.update(visible=False), gr.update(visible=False)
            filepath = file_obj.name
            img = Image.open(filepath)
            filename = os.path.basename(filepath)
            return img, f"📁 Filename: {filename}", gr.update(visible=True), gr.update(visible=True)

        uploader1.upload(load_image_and_filename, inputs=uploader1, outputs=[input1, filename1, input1, rotte1])
        uploader2.upload(load_image_and_filename, inputs=uploader2, outputs=[input2, filename2, input2, rotte2])
        uploader3.upload(load_image_and_filename, inputs=uploader3, outputs=[input3, filename3, input3, rotte3])
            
        rotte1.click(fn=rotate_image_live, inputs=[input1], outputs=input1)
        rotte2.click(fn=rotate_image_live, inputs=[input2], outputs=input2)
        rotte3.click(fn=rotate_image_live, inputs=[input3], outputs=input3)

        output = gr.State()

        # Step 2 onwards (Auto-triggered after Part 3)
        with gr.Row():
            run_pipeline_btn = gr.Button("🧠 'Let AI Do the Work'")
            run_pipeline_btn.click(
                fn=handle_upload,
                inputs=[input1, input2, input3, session_state],
                outputs=output
            )
        with gr.Row():
            process = gr.Markdown("")
        with gr.Column(visible=False) as group_to_show:

            with gr.Row():
                gr.Markdown("# Coin Reference Detection:", elem_id="centered-title")
            with gr.Row():
                out_ref_image_p5 = gr.Image(label="🪙 Coin Reference Detection", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    ref_ratios_str = gr.Textbox(label="🪙 Coin Reference Ratios", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Seat Dimensions:", elem_id="centered-title")
            with gr.Row():
                out_rim_image_p6 = gr.Image(label="📏 Rim Width", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    seat_measurement_str = gr.Textbox(label="📏 Seat Dimensions:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Rim Dimensions:", elem_id="centered-title")
            with gr.Row():
                out_rimellipse_image_p7 = gr.Image(label="📏 Rim Dimensions", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    rim_measurement_str = gr.Textbox(label="📏 Rim Dimensions:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Length of Rim (Inner Top to Outer Bottom):", elem_id="centered-title")
            with gr.Row():
                rimheight_image_p8 = gr.Image(label="📏 Red Line Rim", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    rim_height_str = gr.Textbox(label="📏 Rim Height:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Hole Width:", elem_id="centered-title")
            with gr.Row():
                holewidth_image_p9 = gr.Image(label="🕳️ Hole Width", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    hole_width_str = gr.Textbox(label="🕳️ Hole Width:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Distance from Holes to Top of Inner Rim:", elem_id="centered-title")
            with gr.Row():
                rim_to_hole_img_p10 = gr.Image(label="⬇️ Rim Line (Open)", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    hole_to_top_str = gr.Textbox(label="⬇️ Rim to Hole Dimensions:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Direction of Closed Lid Toilet:", elem_id="centered-title")
            with gr.Row():
                ellipse_viz_image_p10 = gr.Image(label="🔄 Ellipse Arrows", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    direction_str = gr.Textbox(label="🔄 Direction:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Total Height of Entire Toilet:", elem_id="centered-title")
            with gr.Row():
                rim_height_vis_p11 = gr.Image(label="📏 Rim Height on Closed", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    total_height_str = gr.Textbox(label="📏 Total Height:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Distance from Top Portion of Toilet to Holes:", elem_id="centered-title")
            with gr.Row():
                remaining_lid_img_p12 = gr.Image(label="📏 Remaining Lid Portion", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    remaining_str = gr.Textbox(label="📏 Remaining Portion Dimensions:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"

            with gr.Row():
                gr.Markdown("# Width of Top part of Toilet:", elem_id="centered-title")
            with gr.Row():
                out_image = gr.Image(label="📏 Top Rim Width", height=600, width=800, visible=False)
            with gr.Row():
                with gr.Column():
                    z="filler"
                with gr.Column(elem_id="fit-box"):
                    top_width_str = gr.Textbox(label="📏 Top Width:", interactive=False, elem_id='pretty-box', visible=False)
                with gr.Column():
                    z="filler"
            
            gr.Markdown("## 🎯 Prediction Accuracy Analysis", elem_id="centered-title")

            with gr.Row():
                gr.Markdown("### 🖼️ Reference Diagram", elem_id="centered-title")
            with gr.Row():
                ref_image = gr.Image(value="./static/reference.jpg", interactive=False, label="Measurement Guide", height=600)

            unit_dropdown = gr.Radio(choices=["in", "cm"], label="Select Unit", value="in")

            with gr.Row():
                gr.Markdown("### 🧾 Predicted vs Actual Values Table", elem_id='centered-title')

            with gr.Row():
                gr.Markdown("### 📊 Predicted vs Actual Comparison")

            with gr.Column():
                # First row: a, b, c, d
                with gr.Row():
                    with gr.Column():
                        a1 = gr.Number(label="a (Predicted)", interactive=False)
                        a2 = gr.Number(label="a (Actual)")
                    with gr.Column():
                        b1 = gr.Number(label="b (Predicted)", interactive=False)
                        b2 = gr.Number(label="b (Actual)")
                with gr.Row():
                    with gr.Column():
                        c1 = gr.Number(label="c (Predicted)", interactive=False)
                        c2 = gr.Number(label="c (Actual)")
                    with gr.Column():
                        d1 = gr.Number(label="d (Predicted)", interactive=False)
                        d2 = gr.Number(label="d (Actual)")
                
                # Second row: e, f, g, h
                with gr.Row():
                    with gr.Column():
                        e1 = gr.Number(label="e (Predicted)", interactive=False)
                        e2 = gr.Number(label="e (Actual)")
                    with gr.Column():
                        f1 = gr.Number(label="f (Predicted)", interactive=False)
                        f2 = gr.Number(label="f (Actual)")
                with gr.Row():
                    with gr.Column():
                        g1 = gr.Number(label="g (Predicted)", interactive=False)
                        g2 = gr.Number(label="g (Actual)")
                    with gr.Column():
                        h1 = gr.Number(label="h (Predicted)", interactive=False)
                        h2 = gr.Number(label="h (Actual)")
            
            submit_btn = gr.Button("🎯 Evaluate Accuracy")
            with gr.Row():
                progress = gr.Markdown("")
            
            with gr.Column(visible=False) as error:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("🧾 Individual Error %", elem_id='centered-title')
                        result_json = gr.JSON(label="🧾 Individual Error %")
                    with gr.Column():
                        gr.Markdown("📉 Error Plot", elem_id='centered-title')
                        error_plot = gr.Image(label="📉 Error Plot", height=400, width=600, interactive=False, visible=False)
                with gr.Row():
                    avg_error_text = gr.Textbox(label="🎯 Average Error %", interactive=False, elem_id='centered-title')
                
                with gr.Row():
                    gr.Markdown("📥 Download All Results", elem_id='centered-title')
                with gr.Row():
                    download_all_file = gr.File(label="📦 Your Download")
        
        run_pipeline_btn.click(fn=lambda: gr.update(visible=True), outputs=group_to_show)
        run_pipeline_btn.click(fn=lambda: gr.update(value="🔄 Processing... (Please Wait)"), outputs=process)
        submit_btn.click(fn=lambda: gr.update(value="🔄 Calculating Error Percentages... (Please Wait)"), outputs=progress)
            
        import gradio as gr
        
        def get_predictions_by_unit(unit, closed_remaining_in, top_to_hole_line_inch,
                                hole_width_inch, out_rimellipse_inch, rim_height_inch, out_dist_in):
            # Compute values in inches
            a1 = abs(closed_remaining_in - top_to_hole_line_inch)
            b1 = top_to_hole_line_inch
            c1 = hole_width_inch
            d1 = float(out_rimellipse_inch["right_inner"]) * 2
            e1 = float(out_rimellipse_inch["down_inner"]) * 2
            f1 = float(out_rimellipse_inch["right_outer"]) * 2
            g1 = rim_height_inch
            h1 = out_dist_in

            if unit == "in":
                return [round(a1, 2), round(b1, 2), round(c1, 2), round(d1, 2), round(e1, 2), round(f1, 2), round(g1, 2), round(h1, 2)] + [gr.update(value="✅ Process Complete.")]
            elif unit == "cm":
                return [round(x * 2.54, 2) for x in [a1, b1, c1, d1, e1, f1, g1, h1]] + [gr.update(value="✅ Process Complete.")]
            else:
                return [0.0] * 8 + [gr.update(value="✅ Process Complete.")]

        def compare_measurements(a1, b1, c1, d1, e1, f1, g1, h1, a2, b2, c2, d2, e2, f2, g2, h2):
            a2, b2, c2, d2, e2, f2, g2, h2 = float(a2), float(b2), float(c2), float(d2), float(e2), float(f2), float(g2), float(h2)
            a1, b1, c1, d1, e1, f1, g1, h1 = float(a1), float(b1), float(c1), float(d1), float(e1), float(f1), float(g1), float(h1)
            def error(gt, pred):
                if gt == 0:
                    return "N/A"
                return f"{abs(gt - pred) / gt * 100:.2f}%"

            errors = {
                "a": error(a2, a1),
                "b": error(b2, b1),
                "c": error(c2, c1),
                "d": error(d2, d1),
                "e": error(e2, e1),
                "f": error(f2, f1),
                "g": error(g2, g1),
                "h": error(h2, h1)
            }

            valid_errors = [
                abs(gt - pred) / gt * 100
                for gt, pred in [
                    (a2, a1), (b2, b1), (c2, c1), (d2, d1), (e2, e1), (f2, f1), (g2, g1), (h2, h1)
                ] if gt != 0
            ]

            avg_error = f"{sum(valid_errors)/len(valid_errors):.2f}%" if valid_errors else "N/A"

            return errors, f"✅ Average Error: {avg_error}"

        def update_preds(unit, closed_remaining_in_p11, top_to_hole_line_inch_p10, hole_width_inch_p9, out_rimellipse_inch_p7, rim_height_inch_p8, out_dist_in):
            return get_predictions_by_unit(unit, closed_remaining_in_p11, top_to_hole_line_inch_p10,hole_width_inch_p9, out_rimellipse_inch_p7,rim_height_inch_p8, out_dist_in)
        
        unit_dropdown.change(
            fn=update_preds,
            inputs=[unit_dropdown, closed_remaining_in_p11, top_to_hole_line_inch_p10, hole_width_inch_p9, out_rimellipse_inch_p7, rim_height_inch_p8, out_dist_in],
            outputs=[a1, b1, c1, d1, e1, f1, g1, h1, process]
        )
        
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        
        def plot_error_graph(a2, b2, c2, d2, e2, f2, g2, h2, a1, b1, c1, d1, e1, f1, g1, h1):
            labels = list("abcdefgh")
            pred_vals = [float(x) for x in [a1, b1, c1, d1, e1, f1, g1, h1]]
            true_vals = [float(x) for x in [a2, b2, c2, d2, e2, f2, g2, h2]]

            errors = []
            for pred, true in zip(pred_vals, true_vals):
                if true == 0:
                    errors.append(0)
                else:
                    err = abs(pred - true) / true * 100
                    errors.append(err)

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, errors, color="#f05a28")
            ax.set_ylim(0, max(errors) * 1.2 if errors else 1)
            ax.set_ylabel("Error (%)")
            ax.set_xlabel("Parameter")
            ax.set_title("Individual Error per Parameter")

            # Convert plot to PIL Image
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)

            return gr.update(value=img, visible=True)
        
        submit_btn.click(
            fn=lambda *args: (
                compare_measurements(*args)[0],
                compare_measurements(*args)[1],
                (plot_error_graph(*args))
            ),
            inputs=[a1, b1, c1, d1, e1, f1, g1, h1, a2, b2, c2, d2, e2, f2, g2, h2],
            outputs=[result_json, avg_error_text, error_plot]
        )
        
        def download_all_results_combined(
            input1, input2, input3, gallery_segmentation_p4, out_ref_image_p5, ref_ratios_str, out_rim_image_p6, seat_measurement_str,
            out_rimellipse_image_p7, rim_measurement_str, rimheight_image_p8, rim_height_str, holewidth_image_p9, hole_width_str,
            rim_to_hole_img_p10, hole_to_top_str, ellipse_viz_image_p10, direction_str, rim_height_vis_p11, total_height_str,
            remaining_lid_img_p12, remaining_str, out_image, top_width_str, a1, b1, c1, d1, e1, f1, g1, h1, a2, b2, c2, d2, e2, f2, g2,
            h2, result_json, error_plot, avg_error_text
        ):
            import os, zipfile, tempfile
            from PIL import Image

            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(tempfile.gettempdir(), "all_results.zip")

            with zipfile.ZipFile(zip_path, "w") as zipf:

                # Save images if present
                image_dict = {
                    "input1": input1,
                    "input2": input2,
                    "input3": input3,
                    "gallery_segmentation_p4": gallery_segmentation_p4,
                    "out_ref_image_p5": out_ref_image_p5,
                    "out_rim_image_p6": out_rim_image_p6,
                    "out_rimellipse_image_p7": out_rimellipse_image_p7,
                    "rimheight_image_p8": rimheight_image_p8,
                    "holewidth_image_p9": holewidth_image_p9,
                    "rim_to_hole_img_p10": rim_to_hole_img_p10,
                    "ellipse_viz_image_p10": ellipse_viz_image_p10,
                    "rim_height_vis_p11": rim_height_vis_p11,
                    "remaining_lid_img_p12": remaining_lid_img_p12,
                    "out_image": out_image,
                    "error_plot": error_plot
                }

                for name, img in image_dict.items():
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    if isinstance(img, Image.Image):
                        img_path = os.path.join(temp_dir, f"{name}.png")
                        img.save(img_path)
                        zipf.write(img_path, arcname=f"{name}.png")

                # Save a combined text summary
                text_lines = [
                    "📋 Measurement Summary",
                    "----------------------",
                    f"ref_ratios_str:\n{ref_ratios_str}",
                    f"seat_measurement_str:\n{seat_measurement_str}",
                    f"rim_measurement_str:\n{rim_measurement_str}",
                    f"rim_height_str:\n{rim_height_str}",
                    f"hole_width_str:\n{hole_width_str}",
                    f"hole_to_top_str:\n{hole_to_top_str}",
                    f"direction_str:\n{direction_str}",
                    f"total_height_str:\n{total_height_str}",
                    f"remaining_str:\n{remaining_str}",
                    f"top_width_str:\n{top_width_str}",
                    "",
                    "📏 Predicted Values:",
                    f"a: {a1}",
                    f"b: {b1}",
                    f"c: {c1}",
                    f"d: {d1}",
                    f"e: {e1}",
                    f"f: {f1}",
                    f"g: {g1}",
                    f"h: {h1}",
                    "",
                    "📏 Actual Values:",
                    f"a: {a2}",
                    f"b: {b2}",
                    f"c: {c2}",
                    f"d: {d2}",
                    f"e: {e2}",
                    f"f: {f2}",
                    f"g: {g2}",
                    f"h: {h2}",
                    "",
                    "🎯 Accuracy",
                    f"result_json:\n{result_json}",
                    f"avg_error_text:\n{avg_error_text}"
                ]
                summary_path = os.path.join(temp_dir, "summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(text_lines))
                zipf.write(summary_path, arcname="summary.txt")

            return zip_path, gr.update(visible=True), gr.update(value="")
        
        submit_btn.click(
            fn=download_all_results_combined,
            inputs=[
                input1, input2, input3, gallery_segmentation_p4, out_ref_image_p5, ref_ratios_str, out_rim_image_p6, seat_measurement_str,
                out_rimellipse_image_p7, rim_measurement_str, rimheight_image_p8, rim_height_str, holewidth_image_p9, hole_width_str,
                rim_to_hole_img_p10, hole_to_top_str, ellipse_viz_image_p10, direction_str, rim_height_vis_p11, total_height_str,
                remaining_lid_img_p12, remaining_str, out_image, top_width_str, a1, b1, c1, d1, e1, f1, g1, h1, a2, b2, c2, d2, e2, f2, g2,
                h2, result_json, error_plot, avg_error_text
            ],
            outputs=[download_all_file, error, progress]
        )

        # 🔗 Pipeline chaining starting from Part 4
        run_pipeline_btn.click(fn=segment_and_overlay_all_p4,
            inputs=[
                input1,
                input2,
                input3,
                models_holes_p2,
                models_rim_p2,
                models_coinref_p2,
                device_p2
            ],
            outputs=[
                gallery_segmentation_p4,
                binary_masks_p4,
                image_dict_p4,
            ]).\
            then(fn=detect_and_plot_reference_p5,
            inputs=[
                image_dict_p4,
                binary_masks_p4
            ],
            outputs=[
                out_ref_image_p5,     # overlay image
                out_ref_ratios_p5,    # ref_ratios to pass to next part
                ref_ratios_str
            ]).\
            then(fn=analyze_rim_intersections_p6,
            inputs=[
                image_dict_p4,
                binary_masks_p4,
                out_ref_ratios_p5
            ],
            outputs=[
                out_rim_image_p6,
                out_rim_measurements_p6,
                out_rim_measurements_cm_p6,
                out_rim_measurements_inch_p6,
                seat_measurement_str
            ]).\
            then(fn=analyze_rim_ellipse_red_p7,
            inputs=[
                image_dict_p4,
                binary_masks_p4,
                out_ref_ratios_p5
            ],
            outputs=[
                out_rimellipse_image_p7,
                out_rimellipse_ui_p7,
                out_rimellipse_cm_p7,
                out_rimellipse_inch_p7,
                inner_top_p7,
                dir_down_p7,
                dir_right_p7,
                rim_measurement_str
            ]).\
            then(fn=run_rim_height_analysis_p8,
            inputs=[
                btn_rimheight_p8,    # Trigger button as dummy input
                image_dict_p4,
                binary_masks_p4,
                out_ref_ratios_p5,
                inner_top_p7,
                dir_down_p7
            ],
            outputs=[
                rimheight_text_p8,
                rimheight_image_p8,
                rim_height_px_p8,
                rim_height_cm_p8,
                rim_height_inch_p8,
                inner_top_p8,
                rim_height_str
            ]).\
            then(fn=analyze_hole_width_perpendicular_p9,
            inputs=[
                btn_measure_holewidth_p9,  # trigger button dummy input
                binary_masks_p4,
                out_ref_ratios_p5,
                image_dict_p4,
                dir_right_p7,
                inner_top_p7
            ],
            outputs=[
                holewidth_image_p9,
                hole_width_px_p9,
                hole_width_cm_p9,
                hole_width_inch_p9,
                angle_deg_p9,
                pt_min_p9,
                pt_max_p9,
                hole_width_str
            ]).\
            then(fn=compute_top_to_hole_distance_p10,
            inputs=[
                btn_top_to_hole_p10,    # trigger button
                inner_top_p8,
                pt_min_p9,
                pt_max_p9,
                dir_down_p7,
                out_ref_ratios_p5,
                image_dict_p4
            ],
            outputs=[
                rim_to_hole_img_p10,
                top_to_hole_line_px_p10,
                top_to_hole_line_cm_p10,
                top_to_hole_line_inch_p10,
                angle_down_deg_p10,
                angle_perp_deg_p10,
                intersection_point_p10,
                hole_to_top_str
            ]).\
            then(fn=analyze_closed_rim_orientation_p10,
            inputs=[
                btn_ellipse_orient_p10,   # trigger
                binary_masks_p4,
                image_dict_p4
            ],
            outputs=[
                ellipse_viz_image_p10,
                ellipse_angle_deg_p10,
                ellipse_center_p10,
                ellipse_dir_down_p10,
                ellipse_dir_right_p10,
                updated_binary_masks_p4,
                direction_str
            ]).\
            then(fn=analyze_rim_height_on_closed_p11,
            inputs=[
                btn_rim_height_closed_p11,
                binary_masks_p4,
                input3,
                ellipse_angle_deg_p10,
                out_ref_ratios_p5,
                rim_height_cm_p8
            ],
            outputs=[
                rim_height_vis_p11,
                rim_height_cm_p11,
                rim_height_in_p11,
                full_rim_height_cm_p11,
                full_rim_height_in_p11,
                closed_remaining_cm_p11,
                closed_remaining_in_p11,
                pt_start_p11,
                total_height_str
            ]).\
            then(fn=draw_remaining_closed_portion_p12,
            inputs=[
                btn_draw_remaining_p12,
                pt_start_p11,
                closed_remaining_cm_p11,
                out_ref_ratios_p5,
                ellipse_dir_down_p10,
                input3
            ],
            outputs=[
                remaining_lid_img_p12,
                remaining_angle_deg_p12,
                pt_start_p12,
                pt_end_p12,
                remaining_str
            ]).\
            then(fn=analyze_top_rim_width_p13,
            inputs=[
                btn_measure,
                binary_masks_p4,
                ellipse_center_p10,
                ellipse_angle_deg_p10,
                out_ref_ratios_p5,
                input3,
            ],
            outputs=[
                out_image,
                out_pt_top,
                out_pt1,
                out_pt2,
                out_dist_px,
                out_dist_cm,
                out_dist_in,
                out_angle_deg,
                top_width_str
            ]).\
            then(fn=update_preds,
            inputs=[unit_dropdown, closed_remaining_in_p11, top_to_hole_line_inch_p10, hole_width_inch_p9, out_rimellipse_inch_p7, rim_height_inch_p8, out_dist_in],
            outputs=[a1, b1, c1, d1, e1, f1, g1, h1, process])
        
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("\nFor any queries, Feel free to contact 📧 core.atsc@gmail.com", elem_id='centered-title')
        
        with gr.Row():
            gr.Markdown(
                """
                ---
                #### 👨‍💻 Created by Heet Savaliya  
                📧 Email: savaliyaheet19@gmail.com  
                🔗  [LinkedIn](https://www.linkedin.com/in/heet-savaliya-03b863252/)
                🔗  [GitHub](https://github.com/heetsavaliya)  
                © 2025 Heet Savaliya
                """,
                elem_id="footer-text",
            )
import os

port = int(os.environ.get("PORT", 7860))

# 🚀 Launch App
# if __name__ == "__main__":
full_app_interface.launch(server_name="0.0.0.0", server_port=port,)
