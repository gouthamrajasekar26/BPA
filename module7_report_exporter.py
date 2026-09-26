"""
MODULE 7: Automated Forensic Report & Export Engine
===================================================
Compiles analysis metrics from all upstream modules, generates formatted terminal 
forensic summary reports, and exports standardized JSON schemas (`analysis_output.json`).
"""

import json
import os
from datetime import datetime
from typing import Dict, Any


def compile_forensic_report(
    case_number: str,
    investigator_id: str,
    spatter_classification: Dict[str, Any],
    droplet_summary: Dict[str, Any],
    origin_3d_result: Dict[str, Any],
    weapon_result: Dict[str, Any],
    export_json_path: str = "analysis_output.json"
) -> Dict[str, Any]:
    """
    Compiles data metrics into standard forensic report schema, exports `analysis_output.json`,
    and outputs a formatted textual summary to the system log console.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    report_payload = {
        "metadata": {
            "system_name": "Bloodstain Pattern AI & Spatial Analysis System",
            "case_number": case_number,
            "investigator_id": investigator_id,
            "timestamp": timestamp_str,
            "schema_version": "1.0.0"
        },
        "deep_learning_spatter_analysis": {
            "predicted_spatter_type": spatter_classification.get("predicted_class", "N/A"),
            "confidence_score": spatter_classification.get("confidence", 0.0),
            "categorical_probabilities": spatter_classification.get("class_probabilities", {})
        },
        "computer_vision_droplet_metrics": {
            "total_droplets_processed": droplet_summary.get("total_droplets", 0),
            "mean_droplet_width_mm": droplet_summary.get("mean_width_mm", 0.0),
            "mean_impact_angle_deg": droplet_summary.get("mean_impact_angle_deg", 0.0),
            "tail_clipping_factor_applied": 0.85
        },
        "spatial_3d_point_of_origin": {
            "coordinates_xyz_cm": origin_3d_result.get("point_of_origin_3d", [0.0, 0.0, 0.0]),
            "x_convergence_cm": origin_3d_result.get("x_origin", 0.0),
            "y_convergence_cm": origin_3d_result.get("y_origin", 0.0),
            "z_outward_depth_cm": origin_3d_result.get("z_origin_outward", 0.0),
            "margin_of_error_cm": origin_3d_result.get("margin_of_error_cm", 0.0)
        },
        "weapon_classification": {
            "weapon_class_code": weapon_result.get("weapon_class_code", "N/A"),
            "weapon_description": weapon_result.get("weapon_classification", "N/A"),
            "rule_justification": weapon_result.get("rule_justification", "N/A"),
            "inference_confidence": weapon_result.get("inference_confidence", 0.0)
        }
    }

    # Save to JSON file
    with open(export_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)
        
    print(f"[Module 7] Saved structured forensic report to: {os.path.abspath(export_json_path)}")

    # Print Formatted Console Summary
    print_formatted_terminal_report(report_payload)

    return report_payload


def print_formatted_terminal_report(report: Dict[str, Any]):
    """Outputs high-impact formatted ASCII forensic summary report to console."""
    meta = report["metadata"]
    dl = report["deep_learning_spatter_analysis"]
    cv = report["computer_vision_droplet_metrics"]
    po = report["spatial_3d_point_of_origin"]
    wp = report["weapon_classification"]

    banner = "=" * 78
    sub_banner = "-" * 78

    summary = f"""
{banner}
   FORENSIC ANALYSIS SUMMARY REPORT: BLOODSTAIN PATTERN AI & SPATIAL SYSTEM
{banner}
  CASE NUMBER       : {meta['case_number']}
  INVESTIGATOR ID   : {meta['investigator_id']}
  TIMESTAMP         : {meta['timestamp']}
{sub_banner}
 1. SPATTER CLASSIFICATION (PyTorch ResNet-50)
    - Primary Class    : {dl['predicted_spatter_type']}
    - Confidence Score : {dl['confidence_score']*100:.2f}%
{sub_banner}
 2. DROPLET METRICS & TAIL-CLIPPED TRIGONOMETRY (OpenCV Engine)
    - Total Droplets   : {cv['total_droplets_processed']}
    - Mean Droplet W   : {cv['mean_droplet_width_mm']:.2f} mm
    - Mean Impact Angle: {cv['mean_impact_angle_deg']:.2f} deg (Formula: theta = arcsin(w / h*0.85))
{sub_banner}
 3. 3D POINT OF ORIGIN RECONSTRUCTION (Tangent Dynamics d = h * tan(theta))
    - 3D Coordinates   : [X: {po['coordinates_xyz_cm'][0]:.1f} cm, Y: {po['coordinates_xyz_cm'][1]:.1f} cm, Z: {po['coordinates_xyz_cm'][2]:.1f} cm]
    - Margin of Error  : ±{po['margin_of_error_cm']:.2f} cm
{sub_banner}
 4. WEAPON INFERENCE ENGINE (Heuristic & Decision Tree)
    - Weapon Category  : {wp['weapon_description']}
    - Confidence       : {wp['inference_confidence']*100:.1f}%
    - Justification    : {wp['rule_justification']}
{banner}
"""
    print(summary)


if __name__ == "__main__":
    print("=== MODULE 7: Automated Forensic Report & Export Engine ===")
    mock_dl = {"predicted_class": "Medium-Velocity Impact Spatter (MVIS)", "confidence": 0.94, "class_probabilities": {"MVIS": 0.94}}
    mock_cv = {"total_droplets": 12, "mean_width_mm": 2.4, "mean_impact_angle_deg": 33.2}
    mock_po = {"point_of_origin_3d": [310.5, 420.1, 142.8], "x_origin": 310.5, "y_origin": 420.1, "z_origin_outward": 142.8, "margin_of_error_cm": 2.1}
    mock_wp = {"weapon_class_code": "Class II", "weapon_classification": "Class II: Blunt Force Object (Bat/Hammer)", "rule_justification": "MVIS droplet size profile", "inference_confidence": 0.91}
    
    compile_forensic_report("CASE-2026-991", "DET-4082", mock_dl, mock_cv, mock_po, mock_wp)
    print("[Module 7] Execution complete!\n")
