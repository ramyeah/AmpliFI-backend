import os
import sys
from app.ingest import ingest_pdf

docs_folder = "docs"
failed = []

pdfs = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]
print(f"Found {len(pdfs)} PDFs to ingest")

for i, filename in enumerate(pdfs):
    pdf_path = os.path.join(docs_folder, filename)
    source_name = filename.replace(".pdf", "").replace(" ", "-").lower()
    print(f"\n[{i+1}/{len(pdfs)}] Processing: {filename}")
    try:
        ingest_pdf(pdf_path, source_name)
    except Exception as e:
        print(f"  FAILED: {e}")
        failed.append(filename)

print(f"\n✅ Done! Ingested {len(pdfs) - len(failed)}/{len(pdfs)} files")
if failed:
    print(f"❌ Failed files: {failed}")