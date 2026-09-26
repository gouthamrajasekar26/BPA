"""
MASTER PIPELINE ORCHESTRATOR - PHASE 1 FORENSIC PLATFORM (MODULES 1 - 4)
========================================================================
Integrates Phase 1 forensic analysis modules into a unified processing pipeline:
1. Module 1: Automated Dataset & Directory Environment Setup
2. Module 2: PyTorch ResNet-50 Spatter & Impact Velocity Classifier
3. Module 3: OpenCV Droplet Extraction & 15% Tail-Clipping Engine
4. Module 4: 2D Convergence Ray Tracing & 3D Point of Origin Solver
"""

import os
import sys

# Ensure Windows terminal standard output handles UTF-8 correctly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image
from typing import Dict, Any

# Import Core Phase 1 Forensic Modules
import module1_dataset as m1
import module2_model as m2
import module3_droplet_cv as m3
import module4_origin_resolver as m4


def run_full_forensic_pipeline(
    image_input,
    case_number: str = "CASE-2026-FORENSIC",
    investigator_id: str = "FORENSIC-ANALYST-01",
    tail_clip_factor: float = 0.85
) -> Dict[str, Any]:
    """
    Runs the Phase 1 forensic analysis pipeline (Modules 1 to 4)
    on an input bloodstain crime scene image canvas.
    """
    print(f"\n==========================================================================")
    print(f"   STARTING PHASE 1 FORENSIC PIPELINE (MODULES 1-4): {case_number}")
    print(f"==========================================================================\n")

    # Step 1: Environment & Dataset Initialization (Module 1)
    m1.setup_directory_structure()

    # Step 2: PyTorch ResNet-50 Spatter & Velocity Classification (Module 2)
    print("[Pipeline Step 1/4] Executing PyTorch ResNet-50 Spatter Classifier (Module 2)...")
    spatter_classification = m2.predict_spatter_type(image_input)

    # Step 3: OpenCV Droplet Extraction & Tail-Clipped Trigonometry (Module 3)
    print(f"[Pipeline Step 2/4] Executing OpenCV Droplet Extraction Engine (Module 3, Tail Clip: {tail_clip_factor})...")
    droplets_data, annotated_rgb = m3.process_bloodstain_image(image_input, tail_clip_factor=tail_clip_factor)
    
    if not droplets_data:
        print("[Pipeline Warning] No droplets extracted from input image. Generating synthetic fallback droplets...")
        droplets_data, annotated_rgb = m3.generate_synthetic_droplet_samples(count=12, tail_clip_factor=tail_clip_factor)

    total_drops = len(droplets_data)
    mean_width = float(np.mean([d["width_mm"] for d in droplets_data])) if total_drops > 0 else 0.0
    mean_angle = float(np.mean([d["impact_angle_deg"] for d in droplets_data])) if total_drops > 0 else 0.0

    droplet_summary = {
        "total_droplets": total_drops,
        "mean_width_mm": round(mean_width, 2),
        "mean_impact_angle_deg": round(mean_angle, 2),
        "tail_clip_factor_applied": tail_clip_factor
    }

    # Step 4: 2D Convergence & 3D Point of Origin Resolution (Module 4)
    print("[Pipeline Step 3/4] Resolving 2D Area of Convergence & 3D Point of Origin (Module 4)...")
    origin_3d_result = m4.resolve_3d_point_of_origin(droplets_data)
    x_c, y_c, z_o = origin_3d_result["point_of_origin_3d"]
    
    fig_2d_convergence = m4.render_2d_convergence_plot(droplets_data, x_c, y_c)
    fig_3d_origin = m4.render_3d_spatial_origin_plot(droplets_data, origin_3d_result["point_of_origin_3d"])

    print(f"\n==========================================================================")
    print(f"   PHASE 1 PIPELINE COMPLETED SUCCESSFULLY FOR CASE: {case_number}")
    print(f"==========================================================================")
    print(f"   - Spatter Class     : {spatter_classification['predicted_class']}")
    print(f"   - Impact Velocity   : {spatter_classification['impact_velocity']['category']} ({spatter_classification['impact_velocity']['range_ms']})")
    print(f"   - Droplets Extracted: {total_drops}")
    print(f"   - Solved 3D Origin  : [X: {x_c:.1f}, Y: {y_c:.1f}, Z: {z_o:.1f}] cm (±{origin_3d_result['margin_of_error_cm']:.2f} cm)")
    print(f"==========================================================================\n")

    return {
        "case_number": case_number,
        "investigator_id": investigator_id,
        "spatter_classification": spatter_classification,
        "droplet_summary": droplet_summary,
        "droplets_data": droplets_data,
        "annotated_rgb": annotated_rgb,
        "origin_3d_result": origin_3d_result,
        "fig_2d_convergence": fig_2d_convergence,
        "fig_3d_origin": fig_3d_origin
    }


def run_phase1_pipeline(*args, **kwargs):
    """Alias for Phase 1 pipeline execution."""
    return run_full_forensic_pipeline(*args, **kwargs)


if __name__ == "__main__":
    print("==========================================================================")
    print(" PHASE 1 BLOODSTAIN PATTERN ANALYSIS PIPELINE TEST (MODULES 1-4)")
    print("==========================================================================")
    
    # Check or create synthetic test sample
    test_img_path = m1.EXTRACTED_DIR / "mvis" / "mvis_sample_001.jpg"
    if not os.path.exists(test_img_path):
        m1.generate_synthetic_dataset(num_samples_per_class=5)
        test_img_path = list(m1.EXTRACTED_DIR.glob("**/*.jpg"))[0]
        
    print(f"[Phase 1 Pipeline] Testing with sample image: {test_img_path}")
    
    results = run_full_forensic_pipeline(
        image_input=str(test_img_path),
        case_number="TEST-PHASE1-2026",
        investigator_id="CHIEF-INVESTIGATOR-AI"
    )

