from typing import Dict, List, Tuple
import numpy as np


class PPEComplianceChecker:
    
    def __init__(self):
        self.class_names = {
            0: 'helmet',
            1: 'gloves',
            2: 'vest',
            3: 'boots',
            4: 'goggles',
            5: 'person'
        }
        
        self.mandatory_ppe = {
            'high_risk_area': ['helmet', 'vest', 'gloves', 'boots'],
            'standard_area': ['helmet', 'vest'],
            'office_area': ['helmet']
        }
        
        self.risk_penalties = {
            'helmet': 50,
            'vest': 30,
            'gloves': 10,
            'boots': 20,
            'goggles': 5
        }
    
    def check_compliance(self, detections: Dict[str, List], 
                        area_type: str = 'standard_area') -> Dict:
        
        required_ppe = self.mandatory_ppe.get(area_type, ['helmet', 'vest'])
        
        detected_ppe = set(detections.keys())
        detected_ppe.discard('person')
        
        missing_ppe = set(required_ppe) - detected_ppe
        
        is_compliant = len(missing_ppe) == 0
        
        return {
            'is_compliant': is_compliant,
            'detected_ppe': list(detected_ppe),
            'missing_ppe': list(missing_ppe),
            'required_ppe': required_ppe
        }
    
    def calculate_risk_score(self, detections: Dict[str, List],
                            area_type: str = 'standard_area',
                            confidence_threshold: float = 0.5) -> Dict:
        
        required_ppe = self.mandatory_ppe.get(area_type, ['helmet', 'vest'])
        
        base_risk = 0
        penalty_breakdown = {}
        
        for ppe_item in required_ppe:
            if ppe_item not in detections:
                penalty = self.risk_penalties.get(ppe_item, 10)
                base_risk += penalty
                penalty_breakdown[ppe_item] = penalty
        
        confidence_factor = 1.0
        min_confidence = 1.0
        for ppe_name, boxes in detections.items():
            if boxes:
                avg_conf = np.mean([box['confidence'] for box in boxes])
                min_confidence = min(min_confidence, avg_conf)
        
        if min_confidence < confidence_threshold:
            confidence_factor = 0.8
        
        final_risk = min(int(base_risk * confidence_factor), 100)
        
        return {
            'risk_score': final_risk,
            'base_risk': base_risk,
            'confidence_factor': confidence_factor,
            'penalty_breakdown': penalty_breakdown,
            'severity': self._get_severity_level(final_risk)
        }
    
    def _get_severity_level(self, risk_score: int) -> str:
        if risk_score >= 80:
            return 'CRITICAL'
        elif risk_score >= 50:
            return 'HIGH'
        elif risk_score >= 25:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def associate_ppe_to_person(self, person_box: List[float],
                               ppe_boxes: Dict[str, List]) -> Dict[str, bool]:
        
        association = {ppe_name: False for ppe_name in ppe_boxes.keys()}
        
        person_center_x = (person_box[0] + person_box[2]) / 2
        person_center_y = (person_box[1] + person_box[3]) / 2
        person_height = person_box[3] - person_box[1]
        
        for ppe_name, boxes in ppe_boxes.items():
            for ppe_box in boxes:
                iou = self.calculate_iou(person_box, ppe_box)
                
                ppe_center_x = (ppe_box[0] + ppe_box[2]) / 2
                ppe_center_y = (ppe_box[1] + ppe_box[3]) / 2
                
                distance = np.sqrt(
                    (ppe_center_x - person_center_x) ** 2 + 
                    (ppe_center_y - person_center_y) ** 2
                )
                
                max_distance = person_height * 0.5
                
                if iou > 0.1 or distance < max_distance:
                    association[ppe_name] = True
                    break
        
        return association
