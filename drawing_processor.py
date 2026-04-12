"""
Utilities for converting uploaded structural drawings into images for review.
"""

from __future__ import annotations

import base64
import io

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover - surfaced in the Streamlit UI
    convert_from_path = None
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """
    Convert a PDF into one PIL image per page.

    A 200 DPI rasterization is usually enough detail for schedule extraction
    without making each OpenAI image request unnecessarily large.
    """
    if convert_from_path is None:
        raise RuntimeError("pdf2image is not installed. Install it with: pip install pdf2image")
    return convert_from_path(pdf_path, dpi=dpi)


def image_to_base64(pil_image: Image.Image) -> str:
    """
    Convert a PIL image to a base64-encoded PNG string for the OpenAI API.
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def get_page_thumbnails(
    pil_images: list[Image.Image],
    max_width: int = 300,
) -> list[Image.Image]:
    """
    Create display thumbnails for Streamlit page selection.
    """
    thumbnails: list[Image.Image] = []
    for image in pil_images:
        thumb = image.copy()
        ratio = max_width / float(thumb.width)
        max_height = max(1, int(thumb.height * ratio))
        thumb.thumbnail((max_width, max_height))
        thumbnails.append(thumb)
    return thumbnails
