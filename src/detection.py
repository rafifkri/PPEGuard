from ultralytics import YOLO
from typing import Dict, List, Tuple
import os


class PPEDetector:
    
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.class_names = {
            0: 'helmet',
            1: 'gloves',
            2: 'vest',
            3: 'boots',
            4: 'goggles',
            5: 'person'
        }
    
    def detect(self, image_path: str, 
              conf_threshold: float = 0.5,
              iou_threshold: float = 0.45) -> Dict[str, List]:
        
        try:
            results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
            detections = self._parse_results(results)
            return detections
        except Exception as e:
            print(f"Error during detection: {e}")
            return {}
    
    def detect_batch(self, image_paths: List[str],
                    conf_threshold: float = 0.5,
                    iou_threshold: float = 0.45) -> Dict[str, Dict]:
        
        results_dict = {}
        for image_path in image_paths:
            detections = self.detect(image_path, conf_threshold, iou_threshold)
            results_dict[image_path] = detections
        
        return results_dict
    
    def _parse_results(self, results) -> Dict[str, List]:
        detections = {}
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f'unknown_{class_id}')
                confidence = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0]
                
                if class_name not in detections:
                    detections[class_name] = []
                
                detections[class_name].append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [(float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2],
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                })
        
        return detections
    
    def get_class_names(self) -> Dict[int, str]:
        return self.class_names
