"""
MODULE 5: Weapon Classification Heuristic Engine
================================================
Combines rule-based forensic heuristics with a Scikit-Learn Decision Tree classifier
to perform statistical weapon category inference from spatter metrics.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.tree import DecisionTreeClassifier


WEAPON_CLASSES = {
    "Class I": "Class I: Firearm Discharge (High Velocity / Extreme Pressure)",
    "Class II": "Class II: Blunt Force / Sharp Force Object (Bat, Hammer, Knife)",
    "Class III": "Class III: Swung / Cast-Off Weapon Mechanics (Linear Rhythmic Arcs)",
    "Class IV": "Class IV: Gravitational Passive Drip / Low Velocity (Static Pool/Drip)"
}


class WeaponHeuristicEngine:
    """
    Forensic Inference Engine evaluating spatter classification, droplet diameter profiles,
    impact velocities, and linear spatial distributions to output definitive weapon class.
    """
    def __init__(self):
        self.clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        self._train_internal_tree_model()

    def _train_internal_tree_model(self):
        """Trains internal Scikit-Learn decision tree on synthetic forensic feature vectors."""
        # Features: [spatter_class_code, mean_droplet_w_mm, droplet_count, linearity_score]
        # Spatter codes: 0: Passive, 1: Transfer, 2: MVIS, 3: HVIS
        X_train = np.array([
            # Class I: Firearm Discharge (HVIS, tiny droplets <1mm, high drop count)
            [3, 0.4, 150, 0.2],
            [3, 0.7, 120, 0.3],
            [3, 0.9, 200, 0.1],
            # Class II: Blunt / Sharp Force (MVIS, 1-4mm droplets)
            [2, 1.8, 35, 0.4],
            [2, 2.5, 45, 0.3],
            [2, 3.8, 25, 0.5],
            # Class III: Swung / Cast-Off (Linear arcs, uniform sizes)
            [2, 2.2, 30, 0.88],
            [1, 3.0, 20, 0.92],
            [0, 2.8, 25, 0.85],
            # Class IV: Passive Drip (Large drops >4mm, low drop count)
            [0, 5.5, 8, 0.1],
            [0, 6.2, 5, 0.15],
            [1, 4.8, 12, 0.2]
        ])
        y_train = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]) # 1: Class I, 2: Class II, 3: Class III, 4: Class IV
        self.clf.fit(X_train, y_train)

    def classify_weapon(
        self,
        spatter_type: str,
        droplets: List[Dict[str, Any]],
        spatter_confidence: float = 0.90
    ) -> Dict[str, Any]:
        """
        Executes hybrid rule-based and Decision Tree statistical inference on extracted metrics.
        """
        num_drops = len(droplets)
        
        if num_drops > 0:
            mean_w = float(np.mean([d["width_mm"] for d in droplets]))
            std_w = float(np.std([d["width_mm"] for d in droplets]))
            
            # Estimate spatial linearity (cast-off arc signature check)
            centroids = np.array([d["centroid"] for d in droplets])
            if len(centroids) >= 3:
                # Principal component analysis for linearity variance ratio
                cov = np.cov(centroids.T)
                evals = np.real(np.linalg.eigvals(cov))
                linearity_score = float(max(evals) / (sum(evals) + 1e-5))
            else:
                linearity_score = 0.1
        else:
            mean_w = 2.0
            std_w = 0.5
            linearity_score = 0.1

        # Map spatter string to numerical code
        spatter_lower = spatter_type.lower()
        if "hvis" in spatter_lower or "high" in spatter_lower:
            code = 3
        elif "mvis" in spatter_lower or "medium" in spatter_lower:
            code = 2
        elif "transfer" in spatter_lower:
            code = 1
        else:
            code = 0

        # Rule 1: High Velocity Impact Spatter + Mist Droplets (< 1.0mm) -> Firearm
        if (code == 3 or spatter_confidence > 0.85 and "hvis" in spatter_lower) and mean_w < 1.2:
            weapon_code = "Class I"
            rule_justification = "High-Velocity Impact Spatter (HVIS) detected with mist-like droplet diameters (<1.0mm), indicating extreme energy transfer (e.g. firearm discharge)."
            confidence = 0.96

        # Rule 2: Swung / Cast-Off Weapon (High spatial linearity / rhythmic arc)
        elif linearity_score > 0.78 and num_drops >= 5:
            weapon_code = "Class III"
            rule_justification = "Linear rhythmic droplet spatial alignment (Linearity Score > 0.78) indicates cast-off force vector from a swung weapon."
            confidence = 0.91

        # Rule 3: Medium Velocity Impact Spatter + Droplets 1.0mm - 4.0mm -> Blunt/Sharp Force
        elif code == 2 or (1.0 <= mean_w <= 4.2):
            weapon_code = "Class II"
            rule_justification = "Medium-Velocity Impact Spatter (MVIS) with droplet size profile (1.0mm - 4.0mm) matches impact mechanics of a blunt force object (bat, hammer) or sharp force instrument."
            confidence = 0.89

        # Rule 4: Passive Drip / Static Blood Accumulation (> 4.0mm)
        else:
            weapon_code = "Class IV"
            rule_justification = "Large droplet diameter profile (>4.0mm) under low velocity gravitational drip mechanics."
            confidence = 0.85

        # Cross-validate with Decision Tree model prediction
        ml_input = np.array([[code, mean_w, num_drops, linearity_score]])
        tree_pred_class_num = int(self.clf.predict(ml_input)[0])
        tree_pred_map = {1: "Class I", 2: "Class II", 3: "Class III", 4: "Class IV"}
        tree_pred_code = tree_pred_map.get(tree_pred_class_num, weapon_code)

        final_description = WEAPON_CLASSES[weapon_code]

        return {
            "weapon_class_code": weapon_code,
            "weapon_classification": final_description,
            "rule_justification": rule_justification,
            "inference_confidence": confidence,
            "decision_tree_consensus": tree_pred_code == weapon_code,
            "metrics_analyzed": {
                "spatter_type_input": spatter_type,
                "mean_droplet_width_mm": round(mean_w, 2),
                "droplet_count": num_drops,
                "spatial_linearity_score": round(linearity_score, 3)
            }
        }


if __name__ == "__main__":
    print("=== MODULE 5: Weapon Classification Heuristic Engine ===")
    engine = WeaponHeuristicEngine()
    
    # Test sample firearm spatter
    test_hvis = [{"centroid": (i*10, i*10), "width_mm": 0.5} for i in range(20)]
    res = engine.classify_weapon("High-Velocity Impact Spatter (HVIS)", test_hvis)
    
    print("[Module 5] Weapon Category Inference Result:")
    print(f"   - Identified Class: {res['weapon_classification']}")
    print(f"   - Confidence: {res['inference_confidence']*100:.1f}%")
    print(f"   - Justification: {res['rule_justification']}\n")
