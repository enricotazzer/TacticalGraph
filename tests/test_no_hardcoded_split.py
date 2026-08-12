"""Guard against gating behaviour on a split *name*.

This bug class has bitten three times while adding a second corpus, always the same way:
code written when `cross_season` was the only headline split compares `args.split` to that
literal, and on a single-season corpus the comparison is simply false. Nothing raises. The
script prints a full report and silently skips writing an artifact:

  * `train_outcome.py` did not write `module3_test_predictions.parquet`, so the demo's
    per-match probability timeline was blank.
  * `train_patterns.py` did not write `module4_chains.parquet`, so the pattern-browsing page
    had nothing to browse and `review_patterns.py` could not run at all.

The fix in both cases is `CORPORA[args.corpus].split_kinds[0]` -- the corpus's own primary
split. This test keeps it that way.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tacticalgraph.config import CORPORA

REPO_ROOT = Path(__file__).resolve().parents[1]

# `eval/splits.py` is the one place that *should* dispatch on split kind -- it is what turns a
# kind into folds, and it validates the kind against the corpus before doing so.
ALLOWED = {REPO_ROOT / "src" / "tacticalgraph" / "eval" / "splits.py"}

# Comparing a split name to decide which *sentence* to render is fine; comparing it to decide
# whether to write a file is not. Presentation code lives under app/ and is checked separately
# by the page-render harness.
CHECKED_DIRECTORIES = ("scripts", "src")

PATTERN = re.compile(r'==\s*["\']cross_season["\']|["\']cross_season["\']\s*==')


def _python_files() -> list[Path]:
    found: list[Path] = []
    for directory in CHECKED_DIRECTORIES:
        found.extend(sorted((REPO_ROOT / directory).rglob("*.py")))
    return [f for f in found if f not in ALLOWED and "__pycache__" not in f.parts]


def test_no_script_gates_behaviour_on_the_literal_cross_season():
    offenders = []
    for file in _python_files():
        for number, line in enumerate(file.read_text().splitlines(), start=1):
            if PATTERN.search(line):
                offenders.append(f"{file.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        "Behaviour must not be gated on the split *name* -- 'cross_season' does not exist on a "
        "single-season corpus, so the branch is silently skipped rather than failing. Use "
        "CORPORA[corpus].split_kinds[0] instead.\n\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("slug", sorted(CORPORA))
def test_every_corpus_has_a_primary_split(slug: str):
    """The replacement expression must resolve for every corpus, or the fix moves the bug."""
    kinds = CORPORA[slug].split_kinds
    assert kinds, f"corpus {slug!r} declares no split kinds"
    assert isinstance(kinds[0], str) and kinds[0]


def test_artifact_writes_use_the_primary_split():
    """The two writes this test exists for must reference the corpus registry."""
    for name, artifact in (
        ("scripts/train_outcome.py", "module3_test_predictions.parquet"),
        ("scripts/train_patterns.py", "module4_chains.parquet"),
    ):
        text = (REPO_ROOT / name).read_text()
        assert artifact in text, f"{name} no longer writes {artifact}"
        assert "split_kinds[0]" in text, (
            f"{name} must gate its artifact write on CORPORA[...].split_kinds[0]"
        )
