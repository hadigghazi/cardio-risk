import re

import build_report


def test_notebook_holds_every_section_of_the_code_and_does_not_import_it():
    sections = build_report.code_sections()
    source = (build_report.ROOT / "report.md").read_text(encoding="utf-8")
    included = set(build_report.INCLUDE.findall(source))
    assert included == set(sections) - {"CLI"}            # every section except the command line, once each
    assert len(build_report.INCLUDE.findall(source)) == len(included)
    nb = build_report.to_notebook(source, sections)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert not re.search(r"^\s*(import cardio|from cardio import)", code, re.M)
    assert not re.search(r"\bcardio\.(?!csv)", code)       # no leftover module prefix
    assert "__file__" not in code and "argparse" not in code


def test_setup_section_finds_the_project_by_its_data():
    setup = build_report.code_sections()["setup"]
    assert setup.startswith("import math")
    assert build_report.NOTEBOOK_ROOT in setup


def test_unknown_section_is_refused():
    import pytest
    with pytest.raises(SystemExit):
        build_report.expand_includes("<!-- code: nonsense -->", {"setup": "x = 1"})
