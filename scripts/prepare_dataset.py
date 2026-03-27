import argparse
import os
import shutil
import pandas as pd

# Must stay in sync with model/model.py
CLASS_NAMES = [
    "Melanoma",
    "Melanocytic nevus",
    "Basal cell carcinoma",
    "Actinic keratosis",
    "Benign keratosis",
    "Dermatofibroma",
    "Vascular lesion",
]

# Robust mapping from ISIC diagnosis strings (diagnosis_2/3) to fixed project classes
DIAG_MAPPING = {
    # From diagnosis_3 (Specific)
    "Nevus": "Melanocytic nevus",
    "Melanoma, NOS": "Melanoma",
    "Basal cell carcinoma": "Basal cell carcinoma",
    "Solar or actinic keratosis": "Actinic keratosis",
    "Pigmented benign keratosis": "Benign keratosis",
    "Dermatofibroma": "Dermatofibroma",
    "Squamous cell carcinoma, NOS": "Actinic keratosis", # SCC handled as AK/Bowen's proxy
    
    # From diagnosis_2 (Broader/Fallback)
    "Benign soft tissue proliferations - Vascular": "Vascular lesion",
    "Indeterminate epidermal proliferations": "Actinic keratosis",
    "Benign melanocytic proliferations": "Melanocytic nevus",
    "Benign epidermal proliferations": "Benign keratosis",
    "Malignant melanocytic proliferations (Melanoma)": "Melanoma",
    "Malignant adnexal epithelial proliferations - Follicular": "Basal cell carcinoma",
    "Malignant epidermal proliferations": "Actinic keratosis",
    "Benign soft tissue proliferations - Fibro-histiocytic": "Dermatofibroma",
}


def main(images_dir: str, metadata_path: str, output_dir: str, copy: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV metadata
    meta = pd.read_csv(metadata_path)
    
    # Handle id column name mismatch
    id_col = "isic_id" if "isic_id" in meta.columns else "image_id"
    if id_col not in meta.columns:
        raise ValueError(f"Metadata CSV must contain 'image_id' or 'isic_id'. Found: {meta.columns}")

    # Create class sub‑folders
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    processed = 0
    skipped = 0
    for _, row in meta.iterrows():
        img_id = str(row[id_col])
        
        # Try finding a mapping from diagnosis_3 then diagnosis_2
        diag_3 = str(row.get("diagnosis_3", ""))
        diag_2 = str(row.get("diagnosis_2", ""))
        diag_code = str(row.get("dx", "")) # Original HAM10000 column
        
        class_name = None
        for val in [diag_3, diag_2, diag_code]:
            if val in DIAG_MAPPING:
                class_name = DIAG_MAPPING[val]
                break
        
        if class_name is None:
            # Maybe it's already one of our class names?
            if diag_3 in CLASS_NAMES: class_name = diag_3
            elif diag_2 in CLASS_NAMES: class_name = diag_2
            
        if class_name is None:
            print(f"⚠️  Unknown diagnosis for {img_id}: '{diag_3}' / '{diag_2}'; skipping")
            skipped += 1
            continue

        # Look for images in the images directory
        src_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.isfile(src_path):
            # Try without .jpg suffix or different cases if needed, but usually isic_id.jpg
            print(f"⚠️  Image file not found: {src_path}; skipping")
            skipped += 1
            continue

        dst_path = os.path.join(output_dir, class_name, f"{img_id}.jpg")
        if copy:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)
        processed += 1

    print(f"✅ Dataset prepared at: {output_dir}")
    print(f"   Processed {processed} images, skipped {skipped} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAM10000/ISIC for ImageFolder training")
    parser.add_argument("--images_dir", required=True, help="Folder containing raw .jpg files")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--output_dir", required=True, help="Target ImageFolder directory")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    args = parser.parse_args()
    main(args.images_dir, args.metadata, args.output_dir, args.copy)
