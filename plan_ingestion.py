"""
Floor-plan ingestion and preprocessing helpers.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from drawing_processor import pdf_to_images

try:
    import fitz
except Exception:  # pragma: no cover - optional dependency path
    fitz = None


def _load_pdf_with_pymupdf(pdf_bytes: bytes, *, dpi: int = 200) -> tuple[list[Image.Image], list[str]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF unavailable")
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale = dpi / 72.0
    images: list[Image.Image] = []
    texts: list[str] = []
    for page in document:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        images.append(image)
        texts.append(page.get_text("text"))
    return images, texts


def load_plan_document(uploaded_file) -> dict[str, Any]:
    """
    Accept a Streamlit uploaded file and return normalized PIL images.
    """
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    os.makedirs("data/uploads", exist_ok=True)
    text_pages: list[str] = []
    parser = "pil_image"

    if suffix == ".pdf":
        pdf_bytes = uploaded_file.getvalue()
        if fitz is not None:
            try:
                pages, text_pages = _load_pdf_with_pymupdf(pdf_bytes)
                parser = "pymupdf"
            except Exception:
                temp_path = Path("data/uploads") / filename
                temp_path.write_bytes(pdf_bytes)
                pages = pdf_to_images(str(temp_path))
                text_pages = [""] * len(pages)
                parser = "pdf2image_fallback"
        else:
            temp_path = Path("data/uploads") / filename
            temp_path.write_bytes(pdf_bytes)
            pages = pdf_to_images(str(temp_path))
            text_pages = [""] * len(pages)
            parser = "pdf2image_fallback"
    else:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        pages = [ImageOps.exif_transpose(image).convert("RGB")]
        text_pages = [""]

    processed_pages = [normalize_orientation(page) for page in pages]
    return {
        "filename": filename,
        "pages": processed_pages,
        "page_text": text_pages,
        "page_count": len(processed_pages),
        "source_type": "pdf" if suffix == ".pdf" else "image",
        "ingestion_engine": parser,
    }


def normalize_orientation(image: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(image).convert("RGB")
    if img.width < img.height and (img.height / max(img.width, 1)) > 1.35:
        img = img.rotate(90, expand=True)
    return img


def preprocess_plan_image(image: Image.Image) -> Image.Image:
    """
    Clean a plan image for downstream geometry extraction.
    """
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def prepare_plan_for_pipeline(uploaded_file) -> dict[str, Any]:
    document = load_plan_document(uploaded_file)
    base_page = document["pages"][0]
    preprocessed = preprocess_plan_image(base_page)
    document["selected_page"] = base_page
    document["preprocessed_page"] = preprocessed
    document["selected_page_text"] = document.get("page_text", [""])[0] if document.get("page_text") else ""
    return document
