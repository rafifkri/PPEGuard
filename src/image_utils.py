import cv2
import numpy as np
from typing import Dict, List, Tuple, Union
import io


def load_image(image_path: Union[str, io.IOBase]) -> np.ndarray:
    """Load image from file path or file-like object (e.g., Streamlit UploadedFile)."""
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
    else:
        # Handle file-like objects (Streamlit uploaded files, BytesIO, etc.)
        image_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not load image from file object")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, max_size: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    return image


def draw_bounding_box(image: np.ndarray, 
                     x1: float, y1: float, 
                     x2: float, y2: float,
                     label: str = "", 
                     confidence: float = 0.0,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        label_text = f"{label}"
        if confidence > 0:
            label_text = f"{label} {confidence:.2f}"
        
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(image, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)


def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    return image.shape[1], image.shape[0]
