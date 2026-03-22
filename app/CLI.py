import os
import sys
from pathlib import Path

from src.detection import PPEDetector
from src.compliance import PPEComplianceChecker
from src.image_utils import load_image, resize_image, draw_bounding_box, save_image


def main():
    model_path = "models/best_model.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using notebook 03")
        return
    
    detector = PPEDetector(model_path)
    checker = PPEComplianceChecker()
    
    test_image_dir = "data/construction-ppe/images/test"
    
    if not os.path.exists(test_image_dir):
        print(f"Test images directory not found: {test_image_dir}")
        return
    
    image_files = [
        f for f in os.listdir(test_image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if not image_files:
        print("No test images found")
        return
    
    print(f"Found {len(image_files)} test images")
    print()
    
    for idx, image_file in enumerate(image_files[:5], 1):
        print(f"Processing {idx}/{min(5, len(image_files))}: {image_file}")
        
        image_path = os.path.join(test_image_dir, image_file)
        
        try:
            detections = detector.detect(image_path, conf_threshold=0.5)
            
            for area_type in ['high_risk_area', 'standard_area']:
                compliance = checker.check_compliance(detections, area_type=area_type)
                risk_score = checker.calculate_risk_score(detections, area_type=area_type)
                
                print(f"  Area Type: {area_type}")
                print(f"    Compliant: {compliance['is_compliant']}")
                print(f"    Detected PPE: {compliance['detected_ppe']}")
                print(f"    Missing PPE: {compliance['missing_ppe']}")
                print(f"    Risk Score: {risk_score['risk_score']}/100")
                print(f"    Severity: {risk_score['severity']}")
                print()
        
        except Exception as e:
            print(f"  Error: {e}")
        
        print()


if __name__ == "__main__":
    main()
