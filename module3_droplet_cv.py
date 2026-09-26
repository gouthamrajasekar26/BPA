"""
MODULE 3: OpenCV Droplet Extraction & Impact Angle Engine
=========================================================
Applies computer vision segmentation to extract blood droplets from 2D wall images,
implements the 15% tail-clipping algorithm, and calculates exact 3D trigonometric
impact angles theta = arcsin(w / h_clipped).
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from PIL import Image


def process_bloodstain_image(
    image_input: Any,
    min_area: int = 15,
    max_area: int = 50000,
    tail_clip_factor: float = 0.85,
    adaptive_block_size: int = 15
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Ingests a bloodstain image (file path, PIL Image, or numpy ndarray), performs OpenCV
    segmentation, fits ellipses, executes the tail-clipping algorithm (15% reduction), 
    computes impact angles, and outputs extracted droplet spatial metadata alongside an 
    annotated visualization array.
    """
    # 1. Standardize Input Image to BGR Numpy Array
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Could not load image at path: {image_input}")
    elif isinstance(image_input, Image.Image):
        img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    annotated_img = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Image Preprocessing & Adaptive Thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Block size must be odd and > 1
    block_size = max(3, adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1)
    
    # Combined Otsu and Adaptive Thresholding for blood stain isolation
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 3
    )
    binary = cv2.bitwise_or(thresh_otsu, thresh_adapt)

    # Clean up morphological noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 3. Contour Extraction
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    droplets_data = []

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Fit ellipse requires at least 5 points
        if len(cnt) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (axis1, axis2), orientation_deg = ellipse

            # Identify raw major axis (Length 'h') and minor axis (Width 'w')
            w = float(min(axis1, axis2))
            h_raw = float(max(axis1, axis2))

            if h_raw <= 0 or w <= 0:
                continue

            # =========================================================================
            # CRITICAL TAIL-CLIPPING ALGORITHM
            # Shortens major axis by (1 - tail_clip_factor) to strip directional tail noise
            # Default factor: 0.85 (15% reduction)
            # =========================================================================
            h_clipped = max(w, h_raw * tail_clip_factor)

            # Compute Impact Angle: theta = arcsin(w / h_clipped) in degrees
            ratio = min(1.0, max(0.01, w / h_clipped))
            impact_angle_rad = np.arcsin(ratio)
            impact_angle_deg = float(np.degrees(impact_angle_rad))

            # Store droplet data
            drop_meta = {
                "id": idx + 1,
                "centroid": (float(cx), float(cy)),
                "width_mm": float(w),
                "length_raw_mm": float(h_raw),
                "length_clipped_mm": float(h_clipped),
                "impact_angle_deg": impact_angle_deg,
                "orientation_deg": float(orientation_deg),
                "area_px": float(area),
                "aspect_ratio": float(w / h_clipped),
                "tail_clip_applied": float(h_raw - h_clipped)
            }
            droplets_data.append(drop_meta)

            # Draw green bounding ellipse on visualization overlay
            cv2.ellipse(annotated_img, ellipse, (0, 255, 0), 2)
            cv2.circle(annotated_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)

            # Draw directional vector line indicating angle of flight
            vec_length = 25
            rad = np.radians(orientation_deg - 90) # Adjust relative to vertical
            p2_x = int(cx + vec_length * np.cos(rad))
            p2_y = int(cy + vec_length * np.sin(rad))
            cv2.arrowedLine(annotated_img, (int(cx), int(cy)), (p2_x, p2_y), (255, 0, 0), 1, tipLength=0.3)

        except Exception:
            continue

    # Convert annotated image back to RGB for downstream displays
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    print(f"[Module 3] OpenCV Droplet Engine extracted {len(droplets_data)} valid blood droplets (Tail Clip Factor: {tail_clip_factor}).")
    return droplets_data, annotated_rgb


def generate_synthetic_droplet_samples(count: int = 10, tail_clip_factor: float = 0.85) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Generates synthetic droplet metadata and test canvas for standalone testing."""
    height, width = 600, 800
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Target 2D origin point
    target_origin = (400, 500)
    
    for i in range(count):
        cx = np.random.uniform(100, 700)
        cy = np.random.uniform(50, 400)
        
        # Calculate angle towards origin point
        dx = target_origin[0] - cx
        dy = target_origin[1] - cy
        dist = np.sqrt(dx**2 + dy**2)
        orientation = np.degrees(np.arctan2(dy, dx)) + 90
        
        impact_angle = np.degrees(np.arctan2(150.0, dist)) # Simulated height 150cm
        w = float(np.random.uniform(3, 8))
        h_clipped = float(w / max(0.1, np.sin(np.radians(impact_angle))))
        h_raw = h_clipped / tail_clip_factor # Reconstruct raw before tail clipping
        
        # Render sample droplet ellipse onto canvas
        cv2.ellipse(canvas, ((cx, cy), (w*2, h_raw*2), orientation), (120, 10, 20), -1)
        
    droplets, annotated_rgb = process_bloodstain_image(canvas, tail_clip_factor=tail_clip_factor)
    return droplets, annotated_rgb


if __name__ == "__main__":
    print("=== MODULE 3: OpenCV Droplet Extraction & Impact Angle Engine ===")
    drops, visual_img = generate_synthetic_droplet_samples(count=8)
    print(f"[Module 3] Sample Droplet #1 Analysis:")
    if drops:
        d = drops[0]
        print(f"   - Centroid: (X={d['centroid'][0]:.1f}, Y={d['centroid'][1]:.1f})")
        print(f"   - Raw Length (h_raw): {d['length_raw_mm']:.2f} mm")
        print(f"   - Tail-Clipped Length (h_clipped): {d['length_clipped_mm']:.2f} mm")
        print(f"   - Width (w): {d['width_mm']:.2f} mm")
        print(f"   - Calculated Impact Angle theta: {d['impact_angle_deg']:.2f} deg\n")
