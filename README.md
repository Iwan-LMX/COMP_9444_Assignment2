# COMP9444 Assignment 2 - Computer Vision Project

## Project Overview
This repository contains the implementation for COMP9444 Assignment 2, focusing on computer vision techniques and deep learning.

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

## Notebook Sections
1. **Environment Setup**: Import libraries and configure project
2. **Data Loading & Preprocessing**: Load and prepare datasets
3. **Exploratory Data Analysis**: Visualize and analyze data
4. **Model Development**: Build and configure models
5. **Training & Evaluation**: Train models and evaluate performance
6. **Results & Visualization**: Present final results

## GPU Support
The project supports CUDA for GPU acceleration. Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```
You may need to check by your self to install specific torch based on your GPU type.

For Nvidia GPU it is `cuda`.

For Apple silicon or AMD GPUs using `mps`. 

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
