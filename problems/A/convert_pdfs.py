"""Convert PDF papers in data/ to markdown text files using markitdown."""

from pathlib import Path
from markitdown import MarkItDown

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "data_md"
OUT_DIR.mkdir(exist_ok=True)

md = MarkItDown()

pdf_files = sorted(DATA_DIR.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF(s) in {DATA_DIR}")

for pdf_path in pdf_files:
    out_path = OUT_DIR / f"{pdf_path.stem}.md"
    print(f"Converting: {pdf_path.name} ...")
    try:
        result = md.convert(str(pdf_path))
        out_path.write_text(result.text_content, encoding="utf-8")
        print(f"  -> {out_path.name}  ({len(result.text_content)} chars)")
    except Exception as e:
        print(f"  -> FAILED: {e}")

print(f"\nDone. Markdown files saved to {OUT_DIR}")
