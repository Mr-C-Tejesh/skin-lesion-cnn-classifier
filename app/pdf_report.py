"""
app/pdf_report.py — PDF report generation using ReportLab.

Generates a professional diagnostic report containing:
    - Patient information (name, age)
    - Uploaded skin lesion image
    - Top prediction with confidence score
    - Grad-CAM heatmap visualization
    - Medical disclaimer
"""

import os
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)


# ─── Color Palette ────────────────────────────────────────────────────────────

PRIMARY_COLOR = HexColor("#1a73e8")
DARK_COLOR = HexColor("#202124")
GRAY_COLOR = HexColor("#5f6368")
LIGHT_GRAY = HexColor("#f1f3f4")
RED_COLOR = HexColor("#d93025")
WHITE_COLOR = HexColor("#ffffff")


# ─── Custom Styles ────────────────────────────────────────────────────────────

def get_custom_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=PRIMARY_COLOR,
        spaceAfter=6,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=GRAY_COLOR,
        alignment=TA_CENTER,
        spaceAfter=20,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=PRIMARY_COLOR,
        spaceBefore=16,
        spaceAfter=8,
        borderWidth=1,
        borderColor=PRIMARY_COLOR,
        borderPadding=4,
    ))

    styles.add(ParagraphStyle(
        name="InfoLabel",
        parent=styles["Normal"],
        fontSize=10,
        textColor=GRAY_COLOR,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="InfoValue",
        parent=styles["Normal"],
        fontSize=11,
        textColor=DARK_COLOR,
    ))

    styles.add(ParagraphStyle(
        name="Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=RED_COLOR,
        alignment=TA_JUSTIFY,
        spaceBefore=20,
        spaceAfter=10,
        borderWidth=1,
        borderColor=RED_COLOR,
        borderPadding=8,
    ))

    styles.add(ParagraphStyle(
        name="PredictionMain",
        parent=styles["Normal"],
        fontSize=16,
        textColor=DARK_COLOR,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name="PredictionConfidence",
        parent=styles["Normal"],
        fontSize=12,
        textColor=PRIMARY_COLOR,
        alignment=TA_CENTER,
        spaceAfter=12,
    ))

    return styles


# ─── Report Generator ─────────────────────────────────────────────────────────

def generate_pdf_report(
    patient_name: str,
    patient_age: int,
    image_path: str,
    predictions: List[Tuple[str, float]],
    gradcam_path: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive PDF diagnostic report.

    Args:
        patient_name: Patient's full name.
        patient_age: Patient's age in years.
        image_path: Path to the uploaded skin lesion image.
        predictions: List of (class_name, confidence%) tuples (top-3).
        gradcam_path: Path to the Grad-CAM overlay image.
        save_path: Output path for the PDF. If None, saves to temp dir.
    Returns:
        Path to the generated PDF file.
    """
    if save_path is None:
        save_path = os.path.join(
            tempfile.gettempdir(),
            f"skin_lesion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        )

    styles = get_custom_styles()

    # Create the PDF document
    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    elements = []

    # ── Title ──────────────────────────────────────────────────────────────
    elements.append(Paragraph("🔬 Skin Lesion Analysis Report", styles["ReportTitle"]))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles["ReportSubtitle"],
    ))
    elements.append(HRFlowable(
        width="100%", thickness=2, color=PRIMARY_COLOR, spaceAfter=16
    ))

    # ── Patient Information ────────────────────────────────────────────────
    elements.append(Paragraph("Patient Information", styles["SectionHeader"]))

    patient_data = [
        ["Name:", patient_name, "Age:", f"{patient_age} years"],
        ["Date:", datetime.now().strftime("%Y-%m-%d"), "Time:", datetime.now().strftime("%H:%M:%S")],
    ]

    patient_table = Table(patient_data, colWidths=[2 * cm, 6 * cm, 2 * cm, 6 * cm])
    patient_table.setStyle(TableStyle([
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
        ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 10),
        ("FONT", (1, 0), (1, -1), "Helvetica", 11),
        ("FONT", (3, 0), (3, -1), "Helvetica", 11),
        ("TEXTCOLOR", (0, 0), (0, -1), GRAY_COLOR),
        ("TEXTCOLOR", (2, 0), (2, -1), GRAY_COLOR),
        ("TEXTCOLOR", (1, 0), (1, -1), DARK_COLOR),
        ("TEXTCOLOR", (3, 0), (3, -1), DARK_COLOR),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), 1, PRIMARY_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, WHITE_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 12))

    # ── Uploaded Image ─────────────────────────────────────────────────────
    elements.append(Paragraph("Uploaded Skin Lesion Image", styles["SectionHeader"]))

    if os.path.exists(image_path):
        img = RLImage(image_path, width=3.5 * inch, height=3.5 * inch)
        img.hAlign = "CENTER"
        elements.append(img)
    else:
        elements.append(Paragraph(
            "<i>Image not available</i>", styles["InfoValue"]
        ))
    elements.append(Spacer(1, 12))

    # ── Prediction Results ─────────────────────────────────────────────────
    elements.append(Paragraph("Classification Results", styles["SectionHeader"]))

    if predictions:
        # Primary prediction (top-1)
        top_class, top_conf = predictions[0]
        elements.append(Paragraph(
            f"Primary Diagnosis: {top_class}",
            styles["PredictionMain"],
        ))
        elements.append(Paragraph(
            f"Confidence: {top_conf:.2f}%",
            styles["PredictionConfidence"],
        ))

        # Top-3 predictions table
        table_data = [["Rank", "Diagnosis", "Confidence"]]
        for rank, (cls_name, conf) in enumerate(predictions, 1):
            table_data.append([str(rank), cls_name, f"{conf:.2f}%"])

        pred_table = Table(table_data, colWidths=[2 * cm, 10 * cm, 4 * cm])
        pred_table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE_COLOR),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 11),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            # Data rows
            ("FONT", (0, 1), (-1, -1), "Helvetica", 10),
            ("ALIGN", (0, 1), (0, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "CENTER"),
            # Borders and padding
            ("BOX", (0, 0), (-1, -1), 1, PRIMARY_COLOR),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            # Alternating row colors
            ("BACKGROUND", (0, 1), (-1, 1), LIGHT_GRAY),
            ("BACKGROUND", (0, 3), (-1, 3), LIGHT_GRAY),
        ]))
        elements.append(pred_table)

    elements.append(Spacer(1, 12))

    # ── Grad-CAM Visualization ─────────────────────────────────────────────
    elements.append(Paragraph("Grad-CAM Heatmap Visualization", styles["SectionHeader"]))
    elements.append(Paragraph(
        "The heatmap below highlights the image regions that most influenced "
        "the model's prediction. Warmer colors (red/yellow) indicate higher importance.",
        styles["InfoValue"],
    ))
    elements.append(Spacer(1, 6))

    if gradcam_path and os.path.exists(gradcam_path):
        gc_img = RLImage(gradcam_path, width=3.5 * inch, height=3.5 * inch)
        gc_img.hAlign = "CENTER"
        elements.append(gc_img)
    else:
        elements.append(Paragraph(
            "<i>Grad-CAM visualization not available</i>", styles["InfoValue"]
        ))

    elements.append(Spacer(1, 12))

    # ── Disclaimer ─────────────────────────────────────────────────────────
    elements.append(Spacer(1, 24))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=RED_COLOR, spaceAfter=8
    ))
    elements.append(Paragraph(
        "⚠️ <b>IMPORTANT DISCLAIMER:</b> This report is generated by an AI-based "
        "classification system and is intended for informational and educational purposes only. "
        "It is <b>NOT a substitute for professional clinical diagnosis</b>. The predictions "
        "made by this system should not be used as the sole basis for any medical decisions. "
        "Always consult a qualified dermatologist or healthcare professional for proper "
        "diagnosis and treatment. The developers of this system are not responsible for "
        "any clinical decisions made based on this report.",
        styles["Disclaimer"],
    ))

    # ── Build PDF ──────────────────────────────────────────────────────────
    doc.build(elements)
    return save_path


if __name__ == "__main__":
    # Quick test with dummy data
    test_predictions = [
        ("Melanoma", 78.5),
        ("Melanocytic nevus", 12.3),
        ("Basal cell carcinoma", 5.1),
    ]

    pdf_path = generate_pdf_report(
        patient_name="John Doe",
        patient_age=45,
        image_path="/tmp/test_image.png",
        predictions=test_predictions,
        gradcam_path=None,
    )
    print(f"✅ PDF report generated: {pdf_path}")
