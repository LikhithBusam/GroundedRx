#!/usr/bin/env python3
"""Verify every code cell in GroundedRx_Colab.ipynb parses as valid Python.

Cheap, CPU-only, stdlib-only guard against a notebook edit that silently
breaks the file -- the same ast.parse() check used by hand throughout this
project's development, now wired into CI so it runs on every push instead
of relying on someone remembering to do it.

Lines starting with `!` (IPython shell magic, e.g. `!pip install`) are
stripped before parsing -- they aren't valid Python and were never meant to
be.
"""

import ast
import json
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "GroundedRx_Colab.ipynb"


def check_notebook(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        kept = []
        in_shell_continuation = False
        for line in source.splitlines():
            stripped = line.strip()
            if in_shell_continuation:
                in_shell_continuation = stripped.endswith("\\")
                continue
            if stripped.startswith("!"):
                in_shell_continuation = stripped.endswith("\\")
                continue
            kept.append(line)
        py_only = "\n".join(kept)
        try:
            ast.parse(py_only)
        except SyntaxError as e:
            errors.append(f"cell {i}: {e}")
    return errors


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"SKIP: {NOTEBOOK} not found")
        return 0

    errors = check_notebook(NOTEBOOK)
    if errors:
        print(f"FAILED: {len(errors)} code cell(s) don't parse as valid Python:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"OK: all code cells in {NOTEBOOK.name} parse as valid Python")
    return 0


if __name__ == "__main__":
    sys.exit(main())
