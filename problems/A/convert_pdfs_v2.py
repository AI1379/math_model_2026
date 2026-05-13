"""Re-convert two-column PDFs using pymupdf4llm."""

from pathlib import Path
import pymupdf4llm

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "data_md"

# The two problematic two-column papers
BAD_PAPERS = [
    "Ma et al. - 2025 - Phys-Liquid A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deforma.pdf",
    "Xie et al. - 2021 - Segmenting Transparent Object in the Wild with Transformer.pdf",
]

for fname in BAD_PAPERS:
    pdf_path = DATA_DIR / fname
    out_path = OUT_DIR / f"{pdf_path.stem}.md"

    print(f"Converting: {pdf_path.name} ...")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    out_path.write_text(md_text, encoding="utf-8")
    print(f"  -> {out_path.name}  ({len(md_text)} chars)")

print(f"\nDone. Files saved to {OUT_DIR}")
