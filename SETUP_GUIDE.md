SETUP AND EXECUTION GUIDE

PREREQUISITE
- Python 3.10+
- pip package manager
- CUDA 11.8+ (optional, for GPU acceleration)

INSTALLATION STEPS

1. Create Virtual Environment (Recommended)
   
   # Using venv
   python -m venv venv
   venv\Scripts\activate
   
   # OR using conda
   conda create -n ppe_env python=3.10
   conda activate ppe_env

2. Install Dependencies
   
   pip install -r requirements.txt

3. Verify Installation
   
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   python -c "from ultralytics import YOLO; print('YOLOv8 OK')"

WORKFLOW EXECUTION

Step 1: Exploratory Data Analysis
   
   Purpose: Understand dataset structure and class distribution
   
   Open Jupyter and run: notebooks/01_eda_analysis.ipynb
   Or: jupyter notebook notebooks/01_eda_analysis.ipynb
   
   Expected Output:
   - Class distribution plots
   - Image dimension analysis
   - Sample visualizations

Step 2: Data Preprocessing
   
   Purpose: Clean dataset and prepare for training
   
   Run: notebooks/02_data_preprocessing.ipynb
   
   What it does:
   - Removes irrelevant classes
   - Normalizes labels
   - Creates processed dataset in data/construction-ppe-processed/
   
   Expected Output:
   - Cleaned images and labels
   - data.yaml configuration

Step 3: Model Training
   
   Purpose: Train YOLOv8 on cleaned dataset
   
   Run: notebooks/03_model_training.ipynb
   
   Note:
   - First training run downloads YOLOv8m (400MB)
   - Training takes 30-60 minutes on GPU, much longer on CPU
   - Best model saved as: models/yolov8_ppe_detection/weights/best.pt
   
   Expected Output:
   - Trained model weights
   - Training metrics (mAP, precision, recall)
   - Training logs and visualizations

Step 4: Inference Testing
   
   Purpose: Test detection on sample images
   
   Run: notebooks/04_inference_detection.ipynb
   
   Expected Output:
   - Detection results on test images
   - PPE statistics
   - Confidence scores

Step 5: Copy Best Model
   
   After training, copy the best model to models/ directory:
   
   cp models/yolov8_ppe_detection/weights/best.pt models/best_model.pt

USING THE DASHBOARD

Launch Streamlit App
   
   streamlit run app.py
   
   This opens a web interface with:
   - Image detection tab
   - Analytics dashboard
   - Compliance rules viewer
   - Detection logs

Features:
   - Upload images for real-time detection
   - View compliance status
   - Track risk scores
   - Download detection logs
   - Configure area types

QUICK TESTING (Without Dashboard)

Run Demo Inference Script
   
   python inference_demo.py
   
   This:
   - Loads trained model
   - Processes first 5 test images
   - Prints detection results
   - Shows compliance checkups for different areas

TROUBLESHOOTING

Issue: Model not found
   Solution: Train model first using notebook 03, then copy to models/best_model.pt

Issue: CUDA out of memory
   Solution: Reduce batch size in training notebook (line: batch=8 instead of 16)

Issue: Installation fails
   Solution: Use specific Python version: python -m pip install --upgrade pip
           Then retry: pip install -r requirements.txt

Issue: No detections on images
   Solution: Check confidence threshold (lower it to 0.3 or 0.4)
           Verify model was trained properly

PERFORMANCE TIPS

GPU Usage:
   - Nvidia GPU: torch automatically uses CUDA if available
   - Check: python -c "import torch; print(torch.cuda.is_available())"

Batch Processing:
   - Edit inference_demo.py to process more images
   - Increase batch size in training for faster training

Model Optimization:
   - Use YOLOv8n (nano) for faster inference
   - Export to ONNX/TensorRT for deployment

DIRECTORY STRUCTURE REFERENCE

After Complete Setup:
   
   PPE_Risk/
   ├── data/
   │   ├── construction-ppe/                 Raw dataset
   │   ├── construction-ppe-processed/       Processed data
   │   ├── data.yaml                         YOLO config
   │   └── *.png                             Visualizations
   │
   ├── models/
   │   ├── best_model.pt                     Best trained model
   │   └── yolov8_ppe_detection/             Training outputs
   │
   ├── notebooks/
   │   ├── 01_eda_analysis.ipynb
   │   ├── 02_data_preprocessing.ipynb
   │   ├── 03_model_training.ipynb
   │   └── 04_inference_detection.ipynb
   │
   ├── src/
   │   ├── detection.py                      Detection module
   │   ├── compliance.py                     Compliance engine
   │   ├── tracking.py                       Tracking module
   │   ├── image_utils.py                    Image utilities
   │   └── __init__.py
   │
   ├── app.py                                Streamlit dashboard
   ├── inference_demo.py                     Quick test script
   ├── requirements.txt                      Dependencies
   ├── README.md                             Documentation
   └── .gitignore                            Git exclusions

NEXT STEPS AFTER SETUP

1. Run complete workflow (Steps 1-5 above)
2. Test dashboard with your own images
3. Adjust compliance rules as needed
4. Export model for deployment
5. Set up monitoring/logging system

ESTIMATED TIME

Complete workflow (first run):
- Setup and installation: 10-15 minutes
- EDA: 5 minutes
- Preprocessing: 10 minutes
- Training (GPU): 30-60 minutes
- Inference and testing: 5 minutes
- Dashboard testing: 10 minutes
- Total: 70-100 minutes

Total with CPU training: 3-4 hours

For questions or issues, refer to README.md or notebook comments.
