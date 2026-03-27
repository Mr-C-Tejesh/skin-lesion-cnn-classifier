"""
app/main.py — Streamlit web application for Skin Lesion Classification.

Three-page layout via sidebar navigation:
    1. 🔬 Analyze   — Upload image, enter patient info, run prediction,
                       view Grad-CAM + SHAP, download PDF report.
    2. 📋 History   — Table view of all past patient records with search.
    3. ℹ️  About     — Project description and dataset information.

Usage:
    streamlit run app/main.py
"""

import os
import sys
import tempfile
from datetime import datetime

import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SkinLesionModel, CLASS_NAMES, NUM_CLASSES
from app.predict import load_model, predict, preprocess_image
from app.gradcam import generate_gradcam_overlay
from app.shap_explain import generate_shap_plot
from app.pdf_report import generate_pdf_report
from app.history import init_db, insert_record, get_all_records, get_by_name, get_record_count


# ─── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Skin Lesion Classifier — AI Diagnostic Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0e0 !important;
    }

    /* Prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .prediction-card h2 {
        margin: 0;
        font-size: 1.5em;
    }
    .prediction-card p {
        margin: 5px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }

    /* Info cards */
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        border-radius: 0 8px 8px 0;
        padding: 16px;
        margin: 10px 0;
    }

    /* Disclaimer styling */
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 16px;
        margin: 20px 0;
        font-size: 0.85em;
    }

    /* Metric styling */
    .stMetric {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ────────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "model": None,
        "device": None,
        "predictions": None,
        "gradcam_image": None,
        "shap_path": None,
        "uploaded_image": None,
        "uploaded_image_path": None,
        "analysis_complete": False,
        "pdf_report_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_model_cached():
    """Load the model once and cache it in session state."""
    if st.session_state.model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.session_state.device = device
        st.session_state.model = load_model(device=device)


# ─── Initialize ──────────────────────────────────────────────────────────────

init_session_state()
init_db()


# ─── Sidebar Navigation ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🔬 Skin Lesion Classifier")
    st.markdown("---")

    page = st.radio(
        "Navigate to:",
        ["🔬 Analyze", "📋 History", "ℹ️ About"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"**Records:** {get_record_count()} patients analyzed"
    )
    st.markdown(
        f"**Device:** `{'CUDA' if torch.cuda.is_available() else 'CPU'}`"
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; opacity: 0.6; font-size: 0.8em;'>"
        "Powered by EfficientNet-B0<br>ISIC 2019 Dataset</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🔬 Analyze":
    st.markdown("# 🔬 Skin Lesion Analysis")
    st.markdown(
        "Upload a dermoscopic image of a skin lesion to get an AI-powered classification "
        "with visual explanations."
    )
    st.markdown("---")

    # ── Patient Information & Image Upload ─────────────────────────────────
    col_input, col_preview = st.columns([1, 1])

    with col_input:
        st.markdown("### 👤 Patient Information")

        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter full name",
            key="patient_name_input",
        )
        patient_age = st.number_input(
            "Patient Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1,
            key="patient_age_input",
        )

        st.markdown("### 📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Upload a dermoscopic image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="file_uploader",
            help="Supported formats: JPG, JPEG, PNG, BMP",
        )

    with col_preview:
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.markdown("### 🖼️ Uploaded Image Preview")
            st.image(image, caption="Uploaded Skin Lesion", use_container_width=True)

            # Save image to temp file for later use
            temp_img_path = os.path.join(
                tempfile.gettempdir(),
                f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            image.save(temp_img_path)
            st.session_state.uploaded_image = image
            st.session_state.uploaded_image_path = temp_img_path
        else:
            st.markdown("### 🖼️ Image Preview")
            st.info("👆 Upload an image to see the preview here.")

    st.markdown("---")

    # ── Run Analysis Button ────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])

    with col_btn:
        analyze_button = st.button(
            "🚀 Run Analysis",
            use_container_width=True,
            type="primary",
            disabled=(uploaded_file is None or patient_name.strip() == ""),
        )

    if uploaded_file is None or patient_name.strip() == "":
        st.warning("⚠️ Please enter the patient name and upload an image to proceed.")

    # ── Analysis Pipeline ──────────────────────────────────────────────────
    if analyze_button and uploaded_file is not None and patient_name.strip():
        load_model_cached()
        model = st.session_state.model
        device = st.session_state.device
        image = st.session_state.uploaded_image

        with st.spinner("🧠 Running AI analysis... This may take a moment."):

            # Step 1: Prediction
            st.markdown("#### ⏳ Step 1/4: Classifying lesion...")
            predictions = predict(model, image, device, top_k=3)
            st.session_state.predictions = predictions

            # Step 2: Grad-CAM
            st.markdown("#### ⏳ Step 2/4: Generating Grad-CAM heatmap...")
            gradcam_overlay = generate_gradcam_overlay(
                model, image, target_class=None, alpha=0.5, device=device
            )
            st.session_state.gradcam_image = gradcam_overlay

            # Save Grad-CAM image for PDF
            gradcam_path = os.path.join(tempfile.gettempdir(), "gradcam_overlay.png")
            gradcam_overlay.save(gradcam_path)

            # Step 3: SHAP
            st.markdown("#### ⏳ Step 3/4: Computing SHAP explanations...")
            shap_path = generate_shap_plot(
                model, image, target_class=None, device=device, n_background=25
            )
            st.session_state.shap_path = shap_path

            # Step 4: Generate PDF Report
            st.markdown("#### ⏳ Step 4/4: Generating PDF report...")
            pdf_path = generate_pdf_report(
                patient_name=patient_name.strip(),
                patient_age=patient_age,
                image_path=st.session_state.uploaded_image_path,
                predictions=predictions,
                gradcam_path=gradcam_path,
                shap_path=shap_path,
            )
            st.session_state.pdf_report_path = pdf_path

            # Save to history database
            top_class, top_conf = predictions[0]
            insert_record(
                name=patient_name.strip(),
                age=patient_age,
                predicted_class=top_class,
                confidence=top_conf,
                image_path=st.session_state.uploaded_image_path,
            )

            st.session_state.analysis_complete = True

        st.success("✅ Analysis complete!")

    # ── Display Results ────────────────────────────────────────────────────
    if st.session_state.analysis_complete and st.session_state.predictions:
        st.markdown("---")
        st.markdown("## 📊 Results")

        predictions = st.session_state.predictions

        # Top prediction card
        top_class, top_conf = predictions[0]
        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>🎯 {top_class}</h2>
                <p>Confidence: {top_conf:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top-3 predictions in columns
        st.markdown("### Top-3 Predictions")
        pred_cols = st.columns(3)
        for i, (cls_name, conf) in enumerate(predictions):
            with pred_cols[i]:
                emoji = ["🥇", "🥈", "🥉"][i]
                st.metric(
                    label=f"{emoji} Rank {i+1}",
                    value=f"{conf:.2f}%",
                    delta=cls_name,
                )

        st.markdown("---")

        # Visual explanations
        st.markdown("## 🔍 Visual Explanations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("### 🌡️ Grad-CAM Heatmap")
            if st.session_state.gradcam_image:
                st.image(
                    st.session_state.gradcam_image,
                    caption="Grad-CAM: Regions influencing the prediction",
                    use_container_width=True,
                )
            st.markdown(
                '<div class="info-card">'
                "<b>What is Grad-CAM?</b><br>"
                "Gradient-weighted Class Activation Mapping highlights which "
                "regions of the image most influenced the model's decision. "
                "Red/yellow areas indicate high importance."
                "</div>",
                unsafe_allow_html=True,
            )

        with viz_col2:
            st.markdown("### 📊 SHAP Explanation")
            if st.session_state.shap_path and os.path.exists(st.session_state.shap_path):
                st.image(
                    st.session_state.shap_path,
                    caption="SHAP: Pixel-level contribution to prediction",
                    use_container_width=True,
                )
            st.markdown(
                '<div class="info-card">'
                "<b>What is SHAP?</b><br>"
                "SHapley Additive exPlanations show each pixel's contribution "
                "to the final prediction, providing a detailed view of the "
                "model's reasoning process."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # PDF Download
        if st.session_state.pdf_report_path and os.path.exists(st.session_state.pdf_report_path):
            st.markdown("### 📄 Download Report")
            with open(st.session_state.pdf_report_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name=f"skin_lesion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )

        # Disclaimer
        st.markdown(
            '<div class="disclaimer">'
            "⚠️ <b>Disclaimer:</b> This analysis is generated by an AI system "
            "and is for informational/educational purposes only. It is <b>NOT a "
            "substitute for clinical diagnosis</b>. Always consult a qualified "
            "dermatologist for proper diagnosis and treatment."
            "</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📋 History":
    st.markdown("# 📋 Patient History")
    st.markdown("View and search past patient analysis records.")
    st.markdown("---")

    # Search functionality
    col_search, col_stats = st.columns([2, 1])

    with col_search:
        search_query = st.text_input(
            "🔍 Search by patient name",
            placeholder="Enter name to search...",
            key="history_search",
        )

    with col_stats:
        total_records = get_record_count()
        st.metric("Total Records", total_records)

    st.markdown("---")

    # Retrieve records
    if search_query.strip():
        records = get_by_name(search_query.strip())
        st.info(f"Found {len(records)} record(s) matching '{search_query}'")
    else:
        records = get_all_records()

    # Display records
    if records:
        # Convert to DataFrame for display
        df = pd.DataFrame(records)

        # Rename columns for display
        df = df.rename(columns={
            "id": "ID",
            "name": "Patient Name",
            "age": "Age",
            "timestamp": "Date & Time",
            "predicted_class": "Diagnosis",
            "confidence": "Confidence (%)",
            "image_path": "Image Path",
        })

        # Display the table
        st.dataframe(
            df[["ID", "Patient Name", "Age", "Date & Time", "Diagnosis", "Confidence (%)"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Patient Name": st.column_config.TextColumn("Patient Name", width="medium"),
                "Age": st.column_config.NumberColumn("Age", width="small"),
                "Date & Time": st.column_config.TextColumn("Date & Time", width="medium"),
                "Diagnosis": st.column_config.TextColumn("Diagnosis", width="medium"),
                "Confidence (%)": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
            },
        )

        # Summary statistics
        st.markdown("---")
        st.markdown("### 📊 Summary Statistics")

        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Total Analyses", len(records))
        with stat_cols[1]:
            avg_conf = sum(r["confidence"] for r in records) / len(records)
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        with stat_cols[2]:
            avg_age = sum(r["age"] for r in records) / len(records)
            st.metric("Avg Patient Age", f"{avg_age:.0f}")
        with stat_cols[3]:
            unique_diagnoses = len(set(r["predicted_class"] for r in records))
            st.metric("Unique Diagnoses", unique_diagnoses)

        # Diagnosis distribution
        st.markdown("### 📈 Diagnosis Distribution")
        diagnosis_counts = pd.Series(
            [r["predicted_class"] for r in records]
        ).value_counts()
        st.bar_chart(diagnosis_counts)

    else:
        st.info("📭 No records found. Start by analyzing a skin lesion image!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
    ## 🔬 Skin Lesion Cancer Detection using CNN

    This application uses a **fine-tuned EfficientNet-B0** convolutional neural network
    to classify dermoscopic images of skin lesions into **7 diagnostic categories**.
    The tool provides AI-powered analysis along with visual explanations to help
    understand the model's decision-making process.

    ### 🎯 Key Features

    | Feature | Description |
    |---------|-------------|
    | **AI Classification** | EfficientNet-B0 fine-tuned on ISIC 2019 dataset |
    | **Grad-CAM** | Visual heatmap showing regions influencing the prediction |
    | **SHAP Analysis** | Pixel-level contribution analysis using Shapley values |
    | **PDF Reports** | Downloadable diagnostic reports with all analysis results |
    | **Patient History** | SQLite-backed record keeping with search functionality |

    ---

    ### 📊 Dataset — ISIC 2019 (HAM10000)

    The model is trained on the **ISIC 2019 Skin Lesion Dataset (HAM10000)**, one of
    the largest publicly available collections of dermoscopic images. The dataset
    contains images across **7 diagnostic categories**:

    | # | Class | Description |
    |---|-------|-------------|
    | 1 | **Melanoma** | Malignant skin tumor from melanocytes |
    | 2 | **Melanocytic nevus** | Benign skin growth (common mole) |
    | 3 | **Basal cell carcinoma** | Most common type of skin cancer |
    | 4 | **Actinic keratosis** | Pre-cancerous rough, scaly patch |
    | 5 | **Benign keratosis** | Non-cancerous skin growth |
    | 6 | **Dermatofibroma** | Benign fibrous skin nodule |
    | 7 | **Vascular lesion** | Abnormality of blood vessels in skin |

    ---

    ### 🏗️ Technical Architecture

    ```
    Input Image (224×224×3)
            │
            ▼
    ┌──────────────────────┐
    │   EfficientNet-B0    │  ← Pretrained on ImageNet
    │   (Feature Extractor)│  ← Last 2 blocks fine-tuned
    └──────────────────────┘
            │
            ▼
    ┌──────────────────────┐
    │  Custom Classifier   │
    │  Dropout → FC(512)   │
    │  BatchNorm → ReLU    │
    │  Dropout → FC(7)     │
    └──────────────────────┘
            │
            ▼
    7-Class Softmax Prediction
    ```

    ---

    ### 🛠️ Technology Stack

    - **Deep Learning**: PyTorch, EfficientNet-B0
    - **Explainability**: Grad-CAM, SHAP (GradientExplainer)
    - **Frontend**: Streamlit
    - **PDF Generation**: ReportLab
    - **Database**: SQLite
    - **Image Processing**: OpenCV, Pillow

    ---

    ### ⚠️ Disclaimer

    > **This application is for educational and research purposes only.**
    > It is NOT intended for clinical use and should NOT be used as a substitute
    > for professional medical advice, diagnosis, or treatment.
    > Always seek the advice of a qualified dermatologist with any questions
    > regarding a medical condition.

    ---

    ### 📚 References

    1. Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large
       collection of multi-source dermatoscopic images of common pigmented skin lesions.*
       Sci Data 5, 180161 (2018).
    2. Tan, M. & Le, Q. V. *EfficientNet: Rethinking Model Scaling for Convolutional
       Neural Networks.* ICML 2019.
    3. Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks
       via Gradient-based Localization.* ICCV 2017.
    4. Lundberg, S. M. & Lee, S.-I. *A Unified Approach to Interpreting Model
       Predictions.* NeurIPS 2017.
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; opacity: 0.6;'>"
        "Built with ❤️ using Streamlit & PyTorch"
        "</div>",
        unsafe_allow_html=True,
    )
