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
The application is configured to automatically securely download the trained `best_model.pth` weights (19MB) directly from our [Hugging Face Model Hub](https://huggingface.co/tejesh-c/skin-lesion-efficientnet) on its first run.
No manual downloading is required!

## Setup
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Tech Stack
PyTorch · Streamlit · EfficientNet · OpenCV · ReportLab · SQLite
