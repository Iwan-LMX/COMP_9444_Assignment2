# COMP9444 Assignment 2 - Mosquitos on Human Skin Recognition

## Project Overview
This repository contains the implementation for COMP9444 Assignment 2, focusing on **automated detection and recognition of mosquitos on human skin** using advanced computer vision and deep learning techniques.

### Problem Statement
Mosquito-borne diseases pose significant health risks worldwide, affecting millions of people annually. Early detection and monitoring of mosquito presence on human skin is crucial for:
- Disease prevention and control
- Public health surveillance
- Automated monitoring systems
- Real-time alert systems for high-risk areas

### Project Objectives
1. **Detection**: Identify the presence of mosquitos in images of human skin
2. **Localization**: Precisely locate mosquitos within the image using bounding boxes or segmentation
3. **Classification**: Distinguish between different mosquito species (if applicable)
4. **Real-time Processing**: Develop efficient models suitable for real-time applications

### Technical Approach
- **Computer Vision**: Advanced image processing and feature extraction
- **Deep Learning**: Convolutional Neural Networks (CNNs) for object detection
- **Transfer Learning**: Leverage pre-trained models (YOLO, R-CNN, EfficientNet)
- **Data Augmentation**: Enhance model robustness with diverse training scenarios
- **Model Optimization**: Balance accuracy and inference speed for practical deployment

### Expected Deliverables
- Trained deep learning model for mosquito detection
- Comprehensive evaluation metrics and performance analysis
- Visualization of detection results with confidence scores
- Documentation of methodology and experimental results

## Environment Setup

### Prerequisites
- Python 3.12+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd COMP_9444_Assignment2
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```

## Project Structure
```
COMP_9444_Assignment2/
├── main.ipynb              # Main project notebook
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── models/               # Saved models
├── outputs/              # Output files and results
└── logs/                 # Training logs
```

## Key Libraries Used
- **Computer Vision**: OpenCV, PIL, scikit-image
- **Deep Learning**: PyTorch, TensorFlow
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Augmentation**: Albumentations
- **Pre-trained Models**: timm, transformers

## Methodology

### Dataset Requirements
- **Image Types**: High-resolution images of human skin with mosquitos
- **Annotations**: Bounding box coordinates or segmentation masks for mosquito locations
- **Data Diversity**: Various skin tones, lighting conditions, and mosquito poses
- **Split Ratio**: 70% training, 15% validation, 15% testing

### Model Architecture Options
1. **Object Detection Models**:
   - **YOLO (You Only Look Once)**: Real-time detection with good speed-accuracy balance
   - **Faster R-CNN**: High accuracy for precise localization
   - **SSD (Single Shot Detector)**: Efficient multi-scale detection

2. **Segmentation Models**:
   - **U-Net**: Precise pixel-level segmentation
   - **Mask R-CNN**: Instance segmentation with bounding boxes
   - **DeepLab**: Semantic segmentation for detailed boundaries

3. **Classification Models** (if applicable):
   - **EfficientNet**: Optimal efficiency and accuracy
   - **ResNet**: Proven architecture for image classification
   - **Vision Transformer (ViT)**: State-of-the-art attention-based model

### Data Preprocessing Pipeline
1. **Image Preprocessing**:
   - Resize images to standard dimensions (224x224 or 416x416)
   - Normalize pixel values to [0, 1] range
   - Convert color spaces if needed (RGB, HSV, Lab)

2. **Data Augmentation**:
   - Random rotation, scaling, and translation
   - Color jittering and brightness adjustment
   - Gaussian noise and blur for robustness
   - Mosaic and mixup augmentations

3. **Annotation Processing**:
   - Convert bounding boxes to model-specific formats
   - Generate anchor boxes for detection models
   - Create segmentation masks if using segmentation approach

### Training Strategy
- **Transfer Learning**: Initialize with ImageNet pre-trained weights
- **Progressive Training**: Start with frozen backbone, then fine-tune
- **Loss Functions**: Focal loss for detection, Dice loss for segmentation
- **Optimization**: Adam optimizer with learning rate scheduling
- **Early Stopping**: Monitor validation loss to prevent overfitting

### Evaluation Metrics
- **Detection Metrics**:
  - Mean Average Precision (mAP) at IoU thresholds 0.5 and 0.75
  - Precision, Recall, and F1-score
  - Average Precision (AP) for different object sizes

- **Segmentation Metrics**:
  - Intersection over Union (IoU)
  - Dice Coefficient
  - Pixel Accuracy

- **Speed Metrics**:
  - Frames Per Second (FPS)
  - Inference time per image

## GPU Support
The project supports CUDA for GPU acceleration. Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### CUDA Installation (CUDA 12.9 Compatible)
Install PyTorch with CUDA 12.4 support (compatible with CUDA 12.9):
```bash
# For pip installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For conda installation
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

**GPU Type Compatibility:**
- **NVIDIA GPU**: Use `cuda` backend (recommended for this project)
- **Apple Silicon**: Use `mps` backend
- **AMD GPU**: Use `mps` backend (on macOS) or CPU fallback 

## Data Organization
- Place raw data in `data/raw/`
- Processed data will be saved to `data/processed/`
- Models will be saved to `models/`
- Results and visualizations will be saved to `outputs/`

## Notes
- All random seeds are set for reproducibility
- The project uses both PyTorch and TensorFlow (choose based on requirements)
- Data augmentation is configured using Albumentations
- Utility functions are provided for common tasks

## Author
Iwan Li
