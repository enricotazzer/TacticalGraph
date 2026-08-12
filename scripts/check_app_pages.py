#!/usr/bin/env python
"""Render every Streamlit page headlessly and fail on any exception.

    python scripts/check_app_pages.py
    DATA_ROOT=/Volumes/nonexistent python scripts/check_app_pages.py   # portability check

Two implementation details are load-bearing, both learned the hard way:

1. **One subprocess per page.** Instantiating more than one `AppTest` in a single process
   aborts the interpreter outright (no traceback, no exit code you can catch), so pages cannot
   simply be looped over in-process.
2. **The unmounted-DATA_ROOT run is the one that matters.** The app is supposed to read the
   committed `demo_data/` bundle and never touch the external drive. Pointing DATA_ROOT at a
   volume that does not exist is what proves it -- and it is the check that caught real
   breakage when a page reached past the bundle into the live store.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"

# Rendered in one subprocess, so `sys.argv[1]` is the page path.
_RUNNER = """
import sys
from pathlib import Path

# Streamlit puts the *entrypoint's* directory on sys.path for a multipage app, which is how
# `from _shared import ...` resolves in normal use. Running a page file directly skips that, so
# it has to be added here or every page fails on an import that works in production.
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[1]))
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(sys.argv[1], default_timeout=180)
app.run()
if app.exception:
    for exception in app.exception:
        print("EXCEPTION:", exception.value, file=sys.stderr)
    raise SystemExit(1)
print(f"OK markdown={len(app.markdown)} dataframe={len(app.dataframe)} error={len(app.error)}")
for error in app.error:
    print("ST_ERROR:", error.value, file=sys.stderr)
raise SystemExit(2 if app.error else 0)
"""


def pages() -> list[Path]:
    found = [APP_DIR / "Home.py"]
    found.extend(sorted((APP_DIR / "pages").glob("*.py")))
    return [p for p in found if p.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-st-error",
        action="store_true",
        help="treat st.error() calls as warnings rather than failures",
    )
    args = parser.parse_args()

    runner = REPO_ROOT / ".app_page_runner.py"
    runner.write_text(_RUNNER)

    results: list[tuple[str, str, str]] = []
    try:
        for page in pages():
            proc = subprocess.run(
                [sys.executable, str(runner), str(page)],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            if proc.returncode == 0:
                status = "PASS"
            elif proc.returncode == 2:
                status = "PASS(st.error)" if args.allow_st_error else "ST_ERROR"
            else:
                status = "FAIL"
            detail = (proc.stdout.strip().splitlines() or [""])[-1]
            if status in {"FAIL", "ST_ERROR"}:
                detail = (proc.stderr.strip().splitlines() or [detail])[-1][:300]
            results.append((page.name, status, detail))
            print(f"{status:14s} {page.name:34s} {detail}")
    finally:
        runner.unlink(missing_ok=True)

    failures = [r for r in results if r[1] in {"FAIL", "ST_ERROR"}]
    print()
    print(f"{len(results) - len(failures)}/{len(results)} pages rendered cleanly")
    if failures:
        print(json.dumps([{"page": p, "status": s, "detail": d} for p, s, d in failures], indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
