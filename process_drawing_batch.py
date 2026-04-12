"""
Batch drawing processing utility for building a training dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from corrections_logger import log_correction
from drawing_processor import image_to_base64, pdf_to_images
from vision_extractor import extract_from_image, merge_extractions


def process_drawing(
    pdf_path: str,
    output_dir: str = "data/processed_drawings",
) -> dict[str, Any]:
    """
    Full extraction pipeline for one structural drawing PDF.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = pdf_to_images(pdf_path, dpi=200)
    extractions = []
    for image in images:
        b64 = image_to_base64(image)
        result = extract_from_image(b64)
        if result:
            extractions.append(result)

    merged = merge_extractions(extractions)
    drawing_id = Path(pdf_path).stem
    raw_path = output_path / f"{drawing_id}_raw.json"
    raw_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print(f"Processed: {drawing_id}")
    print(f"  Beams:   {len(merged.get('beams', []))}")
    print(f"  Columns: {len(merged.get('columns', []))}")
    print(f"  Girders: {len(merged.get('girders', []))}")
    return merged


def save_corrected_drawing(
    original: dict[str, Any],
    corrected: dict[str, Any],
    drawing_name: str,
    output_dir: str = "data/processed_drawings",
) -> str:
    """
    Store a corrected extraction pair and log it for future ML training.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    corrected_path = output_path / f"{drawing_name}_corrected.json"
    corrected_path.write_text(json.dumps(corrected, indent=2), encoding="utf-8")
    log_correction(original, corrected, drawing_name=drawing_name)
    return str(corrected_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process one structural drawing PDF into structured ML-ready data.")
    parser.add_argument("pdf_path", help="Path to the PDF drawing set")
    parser.add_argument("--output-dir", default="data/processed_drawings", help="Output directory")
    args = parser.parse_args()
    process_drawing(args.pdf_path, output_dir=args.output_dir)
