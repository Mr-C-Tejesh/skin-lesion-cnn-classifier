# Skin Lesion CNN Classifier

CNN-based skin lesion classifier for early cancer detection.

## Features
- 7-class classification using EfficientNet-B0 (transfer learning)
- Grad-CAM heatmap visualizations
- SHAP pixel-level explainability
- PDF diagnostic report generation
- Patient history tracking (SQLite)

## Dataset
ISIC 2019 / HAM10000 — Download from https://www.isic-archive.com

## Setup
pip install -r requirements.txt
streamlit run app/main.py

## Tech Stack
PyTorch · Streamlit · EfficientNet · SHAP · OpenCV · ReportLab · SQLite
