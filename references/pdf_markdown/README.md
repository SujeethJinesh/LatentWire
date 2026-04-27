# PDF Markdown Extracts

This directory contains text-first markdown extracts for the local PDF reference
corpus so subagents can read papers without repeatedly parsing binary PDFs.

Generation command:

```bash
./venv_arm64/bin/python -m pip install pypdf
./venv_arm64/bin/python scripts/convert_reference_pdfs_to_markdown.py \
  --references-dir references \
  --output-dir references/pdf_markdown
```

The conversion used `pypdf==6.10.2` inside `./venv_arm64`. The scratch helper
was promoted to `scripts/convert_reference_pdfs_to_markdown.py` after the first
conversion so the corpus can be regenerated.

Summary:

- PDFs scanned: 172
- Markdown files written: 172
- Normal extractions: 172
- UTF-8 surrogate replacement extractions: 0
- Failures after repair: 0

`conversion_manifest.json` records the source PDF path, source PDF SHA256,
page count, extracted character count, extraction status, and output markdown
path for each converted file.
