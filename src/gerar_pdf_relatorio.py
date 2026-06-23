"""
gerar_pdf_relatorio.py — Converte reports/relatorio_final.md em um PDF
pronto para submissão (A4, ~2 páginas), via Markdown → HTML + Chrome headless.

Uso: .venv/bin/python src/gerar_pdf_relatorio.py
"""

import subprocess
from pathlib import Path
import markdown

MD_PATH   = Path("reports/relatorio_final.md")
HTML_PATH = Path("reports/relatorio_final.html")
PDF_PATH  = Path("reports/relatorio_final.pdf")

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 1.05cm 1.3cm; }
* { box-sizing: border-box; }
body {
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 8.8pt; line-height: 1.19; color: #1a1a1a; margin: 0;
}
h1 { font-size: 12.5pt; margin: 0 0 1pt 0; line-height: 1.12; color: #11304e; }
h4 { font-size: 9.5pt; margin: 0 0 3pt 0; color: #44617e; font-weight: 600; }
h2 { font-size: 10pt; margin: 5pt 0 1.5pt 0; color: #11304e;
     border-bottom: 1px solid #c9d4df; padding-bottom: 1pt; }
p  { margin: 1.9pt 0; text-align: justify; }
ul { margin: 2.4pt 0; padding-left: 15pt; }
li { margin: 1pt 0; }
strong { color: #11304e; }
em { color: #444; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0; font-size: 8pt; break-inside: avoid; }
th, td { border: 1px solid #b9c4d0; padding: 1.8pt 4.5pt; }
th { background: #11304e; color: #fff; text-align: center; font-weight: 600; }
th strong { color: #fff; }
td:first-child { text-align: left; }
td { text-align: right; }
tr:nth-child(even) td { background: #f2f6fa; }
hr { border: none; border-top: 1px solid #ccc; margin: 6pt 0; }
code { background: #eef2f6; padding: 0 2px; border-radius: 2px; font-size: 8.6pt; }
img { display: block; margin: 2pt auto; max-width: 100%; }
figure { margin: 3pt 0; text-align: center; break-inside: avoid; }
figure.w55 { width: 40%; margin: 3pt auto; }
figcaption { font-size: 7.3pt; color: #555; margin-top: 0.5pt; font-style: italic;
             line-height: 1.2; }
.figrow { display: flex; gap: 10pt; align-items: flex-start; justify-content: center;
          break-inside: avoid; }
.figrow figure { flex: 1; margin: 3pt 0; }
.figrow img { max-height: 3.7cm; width: auto; max-width: 100%; }
"""

def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    html = f"""<!doctype html><html lang="pt-BR"><head>
<meta charset="utf-8"><style>{CSS}</style></head>
<body>{body}</body></html>"""
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"[html] {HTML_PATH}")

    subprocess.run([
        CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH.resolve()}",
        HTML_PATH.resolve().as_uri(),
    ], check=True, capture_output=True)
    print(f"[pdf]  {PDF_PATH}")


if __name__ == "__main__":
    main()
