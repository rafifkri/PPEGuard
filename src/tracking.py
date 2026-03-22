from typing import Dict, List, Tuple
import numpy as np


class SimpleTracker:
    
    def __init__(self, max_age: int = 30):
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0
    
    def update(self, detections: Dict[str, List]) -> Dict[int, Dict]:
        
        person_boxes = detections.get('person', [])
        
        updated_tracks = {}
        for track_id, track in self.tracks.items():
            track['age'] += 1
            if track['age'] < self.max_age:
                updated_tracks[track_id] = track
        
        self.tracks = updated_tracks
        
        for person_box in person_boxes:
            assigned = False
            
            for track_id, track in self.tracks.items():
                last_box = track['last_box']
                iou = self._calculate_iou(person_box['box'], last_box)
                
                if iou > 0.3:
                    self.tracks[track_id]['last_box'] = person_box['box']
                    self.tracks[track_id]['age'] = 0
                    self.tracks[track_id]['detection_history'].append(detections)
                    assigned = True
                    break
            
            if not assigned:
                self.next_id += 1
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'last_box': person_box['box'],
                    'age': 0,
                    'detection_history': [detections],
                    'created_at': 0
                }
        
        return self.tracks
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
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
    
    def get_active_tracks(self) -> Dict[int, Dict]:
        return {k: v for k, v in self.tracks.items() if v['age'] < self.max_age}
    
    def reset(self):
        self.tracks = {}
        self.next_id = 0
