# Skin Lesion CNN Classifier

CNN-based skin lesion classifier for early cancer detection.

## Features
- 7-class classification using EfficientNet-B0 (transfer learning)
- Grad-CAM heatmap visualizations
- PDF diagnostic report generation
- Patient history tracking (SQLite)

## Dataset & Demo Samples
ISIC 2019 / HAM10000 — Download the full dataset from [ISIC Archive](https://www.isic-archive.com).
We've also included a few testing images in the `samples/` directory so you can demo the application immediately without downloading the massive 2.5GB dataset!

## Model Checkpoint
To run the Streamlit Web App, you will need the trained weights (`best_model.pth`). 
Due to GitHub file size limits, the checkpoint is hosted externally:
1. Download the `best_model.pth` checkpoint from [Google Drive (Link Here)](#) or [Hugging Face (Link Here)](#).
2. Place the downloaded file tightly inside the `checkpoints/` directory of this project:
   `checkpoints/best_model.pth`

## Setup
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Tech Stack
PyTorch · Streamlit · EfficientNet · OpenCV · ReportLab · SQLite
