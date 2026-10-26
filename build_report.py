"""Turn report.md into the report notebook and execute it.

    python build_report.py                 -> outputs/report.ipynb (executed)
    python build_report.py --no-execute    -> outputs/report.ipynb (cells only)

report.md is the source of the report: plain Markdown in which every ```python fence becomes a code
cell and everything between fences a Markdown cell. It diffs and reviews like text; the notebook is
built from it, never edited by hand.

The notebook is self-contained: it does not import cardio.py. A line `<!-- code: <section> -->` in
report.md becomes a code cell holding that section of cardio.py (the sections are the banner comments
`# ---- name` in the file; `setup` is its imports and constants). cardio.py stays the single source
of the code, used by the tests and the presentation build, and the notebook shows every line of it
where it is first needed. Only the command-line section is left out.
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
INCLUDE = re.compile(r"^<!-- code: ([a-z ]+) -->\s*$", re.M)
BANNER = re.compile(r"^# -{10,} (.+)$", re.M)
# The notebook has no __file__: it finds the project folder by the data it needs
NOTEBOOK_ROOT = ('ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] '
                 'if (p / "data" / "cardio.csv").exists())')


def code_sections(path: Path = ROOT / "cardio.py") -> dict[str, str]:
    """cardio.py cut at its banner comments; 'setup' is everything before the first banner, without the
    module docstring, the command-line import and the future import, and with ROOT found from the
    notebook's folder."""
    source = path.read_text(encoding="utf-8")
    marks = list(BANNER.finditer(source))
    head = source[:marks[0].start()]
    head = re.sub(r'\A""".*?"""\s*', "", head, flags=re.S)
    head = re.sub(r"^from __future__ import annotations\n+", "", head, flags=re.M)
    head = re.sub(r"^import argparse\n", "", head, flags=re.M)
    head, n = re.subn(r"^ROOT = Path\(__file__\)\.resolve\(\)\.parent$", NOTEBOOK_ROOT, head, flags=re.M)
    assert n == 1, "cardio.py no longer defines ROOT from __file__"
    sections = {"setup": head.strip()}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(source)
        sections[m.group(1).strip()] = source[m.end():end].strip()
    return sections


def expand_includes(source: str, sections: dict[str, str]) -> str:
    def code(m):
        name = m.group(1).strip()
        if name not in sections or name == "CLI":
            raise SystemExit(f"report.md asks for an unknown cardio.py section: {name!r}")
        return "```python\n" + sections[name] + "\n```"
    return INCLUDE.sub(code, source)


def to_notebook(source: str, sections: dict[str, str] | None = None) -> nbformat.NotebookNode:
    source = re.sub(r"\A---\n.*?\n---\n", "", source, flags=re.S)  # front matter, if any
    source = expand_includes(source, sections if sections is not None else code_sections())
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


MARKER = re.compile(r"@@[^@\n]*@@")   # a number in the text still to be filled in from the outputs


def reuse_outputs(nb: nbformat.NotebookNode, executed: Path) -> nbformat.NotebookNode:
    """Copy the outputs of an executed notebook into `nb`, cell by cell. Only the text may differ: every
    code cell must be character-for-character the code that produced the outputs, or nothing is copied."""
    old = nbformat.read(executed, as_version=4)
    new_code = [c for c in nb.cells if c.cell_type == "code"]
    old_code = [c for c in old.cells if c.cell_type == "code"]
    if len(new_code) != len(old_code):
        raise SystemExit(f"{len(new_code)} code cells now, {len(old_code)} in {executed}: execute again")
    for i, (n, o) in enumerate(zip(new_code, old_code), 1):
        if n.source != o.source:
            raise SystemExit(f"code cell {i} differs from the executed one: execute again")
        n.outputs, n.execution_count = o.outputs, o.execution_count
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=ROOT / "report.md", type=Path)
    parser.add_argument("--out", default=ROOT / "outputs" / "report.ipynb", type=Path)
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--reuse-outputs", type=Path, metavar="EXECUTED.ipynb",
                        help="take the outputs from this executed notebook instead of executing (text changes only)")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per cell")
    args = parser.parse_args()

    nb = to_notebook(args.source.read_text(encoding="utf-8"))
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    if re.search(r"^\s*(import cardio|from cardio import)", code, re.M):
        raise SystemExit("the notebook must not import cardio.py: include its sections instead")
    if args.reuse_outputs:
        left = MARKER.findall("\n".join(c.source for c in nb.cells if c.cell_type == "markdown"))
        if left:
            raise SystemExit(f"text still has {len(left)} unfilled markers, e.g. {left[0]}")
        nb = reuse_outputs(nb, args.reuse_outputs)
    elif not args.no_execute:
        t0 = time.time()
        # executed in a folder without cardio.py, so a hidden dependency on it would fail here
        run_dir = args.out.parent / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        ExecutePreprocessor(timeout=args.timeout, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(run_dir)}})
        print(f"executed {len(nb.cells)} cells in {time.time() - t0:.0f} s")
    if args.reuse_outputs or not args.no_execute:
        figures = sum("image/png" in o.get("data", {}) for c in nb.cells if c.cell_type == "code" for o in c.outputs)
        shows = len(re.findall(r"plt\.show\(\)", code))
        print(f"{figures} figures embedded ({shows} plt.show() calls)")
        if figures < shows:
            raise SystemExit("figures missing from the notebook: is MPLBACKEND overriding the inline backend?")
        errors = [o.get("ename") for c in nb.cells if c.cell_type == "code" for o in c.outputs
                  if o.get("output_type") == "error"]
        if errors:
            raise SystemExit(f"the notebook has errors: {errors}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.out)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
