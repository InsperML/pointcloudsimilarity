from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def pdf_first_page_to_image_array(pdf_path: Path, dpi: int = 180) -> np.ndarray:
    """Render the first page of a PDF into an image array using pypdfium2."""
    try:
        pdfium = importlib.import_module("pypdfium2")
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: pypdfium2. Install it with: pip install pypdfium2"
        ) from error

    scale = max(float(dpi) / 72.0, 0.1)
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[0]
        try:
            pil_image = page.render(scale=scale).to_pil()
            return np.asarray(pil_image)
        finally:
            page.close()
    finally:
        doc.close()
