"""
gerar_apresentacao.py — Gera o slide deck (16:9, PDF) a partir de
reports/apresentacao.md, reaproveitando as figuras do relatório.

Os slides são separados por uma linha contendo apenas '---'. Uma diretiva
opcional '<!-- class: capa -->' na primeira linha de um slide define a
classe CSS daquele slide.

Uso: .venv/bin/python src/gerar_apresentacao.py
"""

import re
import subprocess
from pathlib import Path
import markdown

MD_PATH   = Path("reports/apresentacao.md")
HTML_PATH = Path("reports/apresentacao.html")
PDF_PATH  = Path("reports/apresentacao.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: 338mm 190mm; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Helvetica Neue", Arial, sans-serif; color: #1b2735; }
.slide { width: 338mm; height: 190mm; padding: 15mm 20mm; page-break-after: always;
         position: relative; display: flex; flex-direction: column; overflow: hidden; }
.slide:last-child { page-break-after: auto; }
.slide .content { flex: 1; display: flex; flex-direction: column; justify-content: flex-start;
                  padding-top: 6mm; }
h2 { font-size: 29pt; color: #11304e; line-height: 1.08; }
h2::after { content: ""; display: block; width: 48mm; height: 3.5px;
            background: #C44E52; margin: 4mm 0 7mm 0; }
.kicker { font-size: 12pt; letter-spacing: 1.5px; text-transform: uppercase;
          color: #C44E52; font-weight: 700; margin-bottom: 2mm; }
ul { margin-left: 7mm; }
li { font-size: 16.5pt; line-height: 1.4; margin-bottom: 3.4mm; }
li ul { margin-top: 2mm; }
li ul li { font-size: 13.5pt; line-height: 1.3; margin-bottom: 1.4mm; color: #3a4a5a; }
strong { color: #11304e; }
em { color: #555; }
p { font-size: 16pt; line-height: 1.4; margin: 2mm 0; }
.pagenum { position: absolute; bottom: 8mm; right: 16mm; font-size: 10.5pt; color: #9aa6b2; }
.foot { position: absolute; bottom: 8mm; left: 20mm; font-size: 9.5pt; color: #9aa6b2; }

/* slide de capa */
.capa { background: #11304e; color: #fff; }
.capa .content { justify-content: center; }
.capa h1 { font-size: 38pt; line-height: 1.12; margin-bottom: 5mm; }
.capa .sub { font-size: 18pt; color: #aebfd2; margin-bottom: 11mm; font-weight: 400; }
.capa .meta { font-size: 13.5pt; color: #e8eef4; line-height: 1.75; }
.capa .meta strong { color: #fff; }
.capa h2 { color: #fff; }

/* figuras */
figure { margin: 3mm auto; text-align: center; }
figure img { max-height: 90mm; max-width: 100%; }
figcaption { font-size: 10.5pt; color: #667; font-style: italic; margin-top: 2mm; }
.figrow { display: flex; gap: 14mm; justify-content: center; align-items: flex-start; }
.figrow figure { flex: 1; }
.figrow img { max-height: 70mm; }

/* layout texto + figura lado a lado */
.split { display: flex; gap: 14mm; align-items: center; }
.split .txt { flex: 1.05; }
.split .fig { flex: 1; text-align: center; }
.split .fig img { max-height: 115mm; max-width: 100%; }
.split li { font-size: 15pt; margin-bottom: 3mm; }

/* destaque numérico */
.big { color: #C44E52; font-weight: 700; }

table { border-collapse: collapse; font-size: 12.5pt; margin: 3mm 0; }
th, td { border: 1px solid #b9c4d0; padding: 1.6mm 5mm; }
th { background: #11304e; color: #fff; }
td { text-align: right; } td:first-child { text-align: left; }
"""

CLASS_RE = re.compile(r"<!--\s*class:\s*([\w\- ]+?)\s*-->")


def main():
    raw = MD_PATH.read_text(encoding="utf-8")
    chunks = re.split(r"(?m)^-{3,}\s*$", raw)
    chunks = [c.strip() for c in chunks if c.strip()]

    slides_html = []
    page = 0
    for chunk in chunks:
        cls = ""
        m = CLASS_RE.search(chunk)
        if m:
            cls = m.group(1).strip()
            chunk = CLASS_RE.sub("", chunk, count=1).strip()
        body = markdown.markdown(chunk, extensions=["tables", "sane_lists", "attr_list"])
        is_capa = "capa" in cls
        if not is_capa:
            page += 1
            num = f'<span class="pagenum">{page}</span>'
            foot = '<span class="foot">KDD — Patologia cardíaca pediátrica (RHP)</span>'
        else:
            num = foot = ""
        slides_html.append(
            f'<section class="slide {cls}"><div class="content">{body}</div>{foot}{num}</section>'
        )

    html = (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            f'<style>{CSS}</style></head><body>{"".join(slides_html)}</body></html>')
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"[html] {HTML_PATH}  ({len(slides_html)} slides)")

    subprocess.run([
        CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH.resolve()}", HTML_PATH.resolve().as_uri(),
    ], check=True, capture_output=True)
    print(f"[pdf]  {PDF_PATH}")


if __name__ == "__main__":
    main()
