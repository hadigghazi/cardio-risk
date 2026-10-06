"""Turn report.md into the report notebook and execute it.

    python build_report.py                 -> outputs/report.ipynb (executed)
    python build_report.py --no-execute    -> outputs/report.ipynb (cells only)

report.md is the source of the report: plain Markdown in which every ```python fence becomes a code
cell and everything between fences a Markdown cell. It diffs and reviews like text; the notebook is
built from it, never edited by hand.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent
FENCE = re.compile(r"^```python\n(.*?)^```\s*$", re.M | re.S)


def to_notebook(source: str) -> nbformat.NotebookNode:
    source = re.sub(r"\A---\n.*?\n---\n", "", source, flags=re.S)  # front matter, if any
    cells, pos = [], 0
    for m in FENCE.finditer(source):
        text = source[pos:m.start()].strip()
        if text:
            cells.append(nbformat.v4.new_markdown_cell(text))
        cells.append(nbformat.v4.new_code_cell(m.group(1).rstrip()))
        pos = m.end()
    if source[pos:].strip():
        cells.append(nbformat.v4.new_markdown_cell(source[pos:].strip()))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=ROOT / "report.md", type=Path)
    parser.add_argument("--out", default=ROOT / "outputs" / "report.ipynb", type=Path)
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per cell")
    args = parser.parse_args()

    nb = to_notebook(args.source.read_text(encoding="utf-8"))
    if not args.no_execute:
        t0 = time.time()
        ExecutePreprocessor(timeout=args.timeout, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(ROOT)}})
        print(f"executed {len(nb.cells)} cells in {time.time() - t0:.0f} s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.out)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
