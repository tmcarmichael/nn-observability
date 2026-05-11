"""Tests for reports/paper_values.json and reports/scopes.json.

Provenance integrity: every source_files entry exists; every key_paths
resolves; every scope name in paper_values.json appears in scopes.json.

Direct-read auto-verification: for every macro with formula starting
with "direct read", load the JSON cell and confirm the macro value
matches at the precision the macro was emitted.

Idempotency: build_export() called twice yields identical content
(modulo timestamp).

Cross-repo coherence: paper_values.json and scopes.json refer to
results files that exist; scopes.json membership matches the live
SCOPES dict in analysis.load_results.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
RESULTS = REPO / "results"
PAPER_VALUES = REPORTS / "paper_values.json"
SCOPES_JSON = REPORTS / "scopes.json"
FIGURE_SOURCES = REPORTS / "figure_sources.json"


def _all_strings(obj) -> Iterator[str]:
    """Yield every string value reachable in a JSON-shaped object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v)


@pytest.fixture(scope="module")
def paper_values() -> dict:
    if not PAPER_VALUES.is_file():
        pytest.skip(f"{PAPER_VALUES} not generated; run paper-adot 'just paper-values'")
    return json.loads(PAPER_VALUES.read_text())


@pytest.fixture(scope="module")
def scopes_export() -> dict:
    if not SCOPES_JSON.is_file():
        pytest.skip(f"{SCOPES_JSON} not generated; run 'just export-scopes'")
    return json.loads(SCOPES_JSON.read_text())


def _resolve(d, key_path: str):
    """Walk a dotted key_path through a dict/list structure."""
    cursor = d
    for part in key_path.split("."):
        if isinstance(cursor, dict):
            cursor = cursor[part]
        elif isinstance(cursor, list):
            cursor = cursor[int(part)]
        else:
            raise TypeError(f"Cannot index {type(cursor).__name__} with {part!r}")
    return cursor


def _detect_dp(value_str: str) -> int:
    """Count decimal places in a formatted numeric string."""
    if "." not in value_str:
        return 0
    return len(value_str.rsplit(".", 1)[1])


# ---------- schema and basic shape ----------


def test_paper_values_schema_version(paper_values):
    assert paper_values["schema_version"] == "1.2"


def test_paper_values_has_macros(paper_values):
    assert paper_values["n_macros"] > 0
    assert paper_values["n_macros_with_provenance"] > 0
    assert len(paper_values["macros"]) == paper_values["n_macros"]


# ---------- macro provenance thresholds ----------
#
# These two bounds lock the current state and force coverage to improve
# monotonically. The provenance floor only goes up. The orphan ceiling only
# goes down. To change either, edit the constants here AND
# `paper-adot/generators/macro_provenance.json` in the same commit.

# Lower bound on macros that have a `source_files`/`key_paths` annotation.
# 135 of 270 = 50.0 percent coverage. Adding a new macro without provenance
# does not fail this test directly. The orphan ceiling below catches that.
MIN_MACROS_WITH_PROVENANCE = 135

# Upper bound on macros that have neither `source_files` nor `key_paths`.
# Adding a new \newcommand without a corresponding macro_provenance.json
# entry pushes the orphan count above 136 and fails this test. Fix by adding
# provenance, or by raising the ceiling in the same commit with a reason.
MAX_ORPHAN_MACROS = 136


def _orphan_macros(paper_values: dict) -> list[dict]:
    return [m for m in paper_values["macros"] if not (m.get("source_files") or [])]


def test_macros_with_provenance_floor(paper_values):
    """At least MIN_MACROS_WITH_PROVENANCE macros must carry source_files +
    key_paths. Coverage can grow; it cannot shrink without explicit consent."""
    n = paper_values["n_macros_with_provenance"]
    assert n >= MIN_MACROS_WITH_PROVENANCE, (
        f"Macro provenance coverage regressed: {n} annotated, "
        f"floor is {MIN_MACROS_WITH_PROVENANCE}. To intentionally lower the "
        f"floor, edit MIN_MACROS_WITH_PROVENANCE in this file with a reason."
    )


def test_orphan_macros_ceiling(paper_values):
    """At most MAX_ORPHAN_MACROS macros may lack a `source_files` annotation.
    A new macro added to data_macros.sty without provenance pushes this
    count up and fails. The fix is either to annotate it in
    paper-adot/generators/macro_provenance.json or to bump the ceiling here
    in the same commit (with reason).
    """
    orphans = _orphan_macros(paper_values)
    n = len(orphans)
    if n > MAX_ORPHAN_MACROS:
        new_orphans = [m["name"] for m in orphans][-(n - MAX_ORPHAN_MACROS) :]
        pytest.fail(
            f"Orphan-macro ceiling exceeded: {n} unannotated, ceiling is "
            f"{MAX_ORPHAN_MACROS}. Likely new orphans: {new_orphans}. "
            f"Add entries to paper-adot/generators/macro_provenance.json or "
            f"bump MAX_ORPHAN_MACROS with a documented reason."
        )


def test_macro_coverage_invariant(paper_values):
    """n_macros == n_macros_with_provenance + n_orphans. Sanity check that
    the bookkeeping fields agree with the macro list."""
    n_total = paper_values["n_macros"]
    n_with = paper_values["n_macros_with_provenance"]
    n_orphan = len(_orphan_macros(paper_values))
    assert n_with + n_orphan == n_total, (
        f"Macro accounting drift: with_provenance={n_with} + orphan={n_orphan} != total={n_total}"
    )


def test_scopes_schema_version(scopes_export):
    assert scopes_export["schema_version"] == "1.0"


# ---------- paper_version consistency with main.tex ----------


def test_paper_version_matches_main_tex(paper_values):
    """paper_version in the export must match main.tex \\paperversion."""
    main_tex = REPO.parent / "paper-adot" / "main.tex"
    if not main_tex.is_file():
        pytest.skip("paper-adot/main.tex not present in this checkout")
    import re

    text = main_tex.read_text()
    m = re.search(r"\\def\\paperversion\{([^}]+)\}", text)
    assert m, "paper-adot/main.tex missing \\paperversion"
    assert paper_values["paper_version"] == m.group(1), (
        f"paper_values.json paper_version={paper_values['paper_version']!r} "
        f"differs from main.tex {m.group(1)!r}; run 'just paper-values'"
    )


# ---------- provenance integrity ----------


def test_source_files_exist(paper_values):
    """Every source_files entry must reference an existing file in results/."""
    missing = []
    for m in paper_values["macros"]:
        for fname in m.get("source_files") or []:
            if not (RESULTS / fname).is_file():
                missing.append(f"{m['name']} -> {fname}")
    assert not missing, "Missing source files:\n" + "\n".join(missing)


def test_key_paths_resolve(paper_values):
    """Every key_paths entry must resolve in the (first) referenced JSON."""
    failures = []
    for m in paper_values["macros"]:
        files = m.get("source_files") or []
        keys = m.get("key_paths") or []
        if not files or not keys:
            continue
        ref = json.loads((RESULTS / files[0]).read_text())
        for k in keys:
            try:
                _resolve(ref, k)
            except (KeyError, IndexError, TypeError) as e:
                failures.append(f"{m['name']}: key {k!r} unresolvable in {files[0]} ({e})")
    assert not failures, "Unresolvable key_paths:\n" + "\n".join(failures)


def test_scopes_referenced_exist(paper_values, scopes_export):
    """Every scope name in paper_values.json must appear in scopes.json
    OR in the macro_provenance.json scopes block (for paper-local scopes
    not in the SCOPES dict, like absorption_cohort_14).

    Skips when the paper repo is not in a sibling checkout: in that case
    paper-local scopes cannot be verified, so the test cannot distinguish
    a typo from a legitimate paper-local scope name."""
    prov_path = REPO.parent / "paper-adot" / "generators" / "macro_provenance.json"
    if not prov_path.is_file():
        pytest.skip(
            "paper-adot/generators/macro_provenance.json not present; "
            "cross-repo scope check requires sibling checkout"
        )
    prov_scopes = set(json.loads(prov_path.read_text()).get("scopes", {}).keys())
    public_scopes = set(scopes_export.get("scopes", {}).keys())
    known = prov_scopes | public_scopes
    unknown = []
    for m in paper_values["macros"]:
        scope = m.get("scope")
        if scope and scope not in known:
            unknown.append(f"{m['name']} -> scope {scope!r}")
    assert not unknown, "Unknown scope references:\n" + "\n".join(unknown)


# ---------- direct-read auto-verification ----------


def test_direct_read_macros_match_jsons(paper_values):
    """For every macro with formula starting with 'direct read', the macro
    value must match the JSON cell at the precision the macro was emitted."""
    failures = []
    for m in paper_values["macros"]:
        formula = (m.get("formula") or "").strip()
        if not formula.startswith("direct read"):
            continue
        files = m.get("source_files") or []
        keys = m.get("key_paths") or []
        if len(files) != 1 or len(keys) != 1:
            continue  # multi-source aggregations are tested by formula audit
        try:
            cell = _resolve(json.loads((RESULTS / files[0]).read_text()), keys[0])
        except Exception as e:  # noqa: BLE001
            failures.append(f"{m['name']}: load/resolve failed ({e})")
            continue
        macro_value = m["value"]
        dp = _detect_dp(macro_value)
        try:
            cell_f = float(cell)
        except (TypeError, ValueError):
            failures.append(f"{m['name']}: cell value {cell!r} is not numeric")
            continue
        formatted = f"{cell_f:.{dp}f}"
        if formatted != macro_value:
            failures.append(
                f"{m['name']}: macro={macro_value!r} JSON={cell_f!r} (formatted at {dp} dp = {formatted!r})"
            )
    assert not failures, "Direct-read mismatches:\n" + "\n".join(failures)


# ---------- idempotency ----------


def _import_paper_export():
    """Lazy-import paper-adot's exporter (only used for idempotency check)."""
    paper_gen = REPO.parent / "paper-adot" / "generators"
    if not (paper_gen / "export_paper_values.py").is_file():
        return None
    sys.path.insert(0, str(paper_gen))
    try:
        import export_paper_values  # type: ignore
    finally:
        sys.path.pop(0)
    return export_paper_values


def test_export_paper_values_idempotent():
    """Calling build_export() twice yields identical content (modulo timestamp)."""
    mod = _import_paper_export()
    if mod is None:
        pytest.skip("paper-adot exporter not present in this checkout")
    a = mod.build_export()
    b = mod.build_export()
    assert mod._strip_volatile(a) == mod._strip_volatile(b), (
        "export_paper_values.build_export() is non-deterministic"
    )


def test_export_scopes_idempotent():
    """Calling build_export() twice yields identical content (modulo timestamp)."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import export_scopes  # type: ignore
    finally:
        sys.path.pop(0)
    a = export_scopes.build_export()
    b = export_scopes.build_export()
    assert export_scopes._strip_volatile(a) == export_scopes._strip_volatile(b), (
        "export_scopes.build_export() is non-deterministic"
    )


# ---------- scopes membership matches the source dict ----------


# ---------- figure_sources.json ----------


@pytest.fixture(scope="module")
def figure_sources() -> dict:
    if not FIGURE_SOURCES.is_file():
        pytest.skip(f"{FIGURE_SOURCES} not generated; run paper-adot 'just figure-sources'")
    return json.loads(FIGURE_SOURCES.read_text())


def test_figure_sources_schema(figure_sources):
    assert figure_sources["schema_version"] == "1.0"
    assert figure_sources["n_figures"] > 0
    assert len(figure_sources["figures"]) == figure_sources["n_figures"]


def test_figure_source_files_exist(figure_sources):
    """Every source_files entry must point to a file in results/."""
    missing = []
    for name, fig in figure_sources["figures"].items():
        for f in fig.get("source_files") or []:
            if not (RESULTS / f).is_file():
                missing.append(f"{name} -> {f}")
    assert not missing, "Missing figure source files:\n" + "\n".join(missing)


def test_figure_alt_text_files_exist(figure_sources):
    """Every alt_text_file must exist next to the PDF in figures/."""
    figures_dir = REPO.parent / "paper-adot" / "figures"
    if not figures_dir.is_dir():
        pytest.skip("paper-adot/figures not present")
    missing = []
    for name, fig in figure_sources["figures"].items():
        alt = fig.get("alt_text_file")
        if alt and not (figures_dir / alt).is_file():
            missing.append(f"{name} -> {alt}")
    assert not missing, "Missing alt-text sidecars:\n" + "\n".join(missing)


def test_figure_pdf_committed(figure_sources):
    """Every figure key must correspond to a committed PDF in figures/."""
    figures_dir = REPO.parent / "paper-adot" / "figures"
    if not figures_dir.is_dir():
        pytest.skip("paper-adot/figures not present")
    missing = [name for name in figure_sources["figures"] if not (figures_dir / name).is_file()]
    assert not missing, "Figure PDFs missing: " + ", ".join(missing)


def test_export_figure_sources_idempotent():
    """Calling build_export() twice yields identical content (modulo timestamp)."""
    paper_gen = REPO.parent / "paper-adot" / "generators"
    if not (paper_gen / "export_figure_sources.py").is_file():
        pytest.skip("paper-adot exporter not present in this checkout")
    sys.path.insert(0, str(paper_gen))
    try:
        import export_figure_sources  # type: ignore
    finally:
        sys.path.pop(0)
    a = export_figure_sources.build_export()
    b = export_figure_sources.build_export()
    assert export_figure_sources._strip_volatile(a) == export_figure_sources._strip_volatile(b), (
        "export_figure_sources.build_export() is non-deterministic"
    )


# ---------- formal schema validation ----------


def _schema_dispatch_module():
    """Lazy-import scripts/validate_schemas.py for its DISPATCH + classifier."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import validate_schemas as vs  # type: ignore
    finally:
        sys.path.pop(0)
    return vs


def _all_result_files() -> list[Path]:
    return sorted((REPO / "results").glob("*.json"))


def _classify_or_skip(path: Path) -> str:
    """Return the schema kind for a result file, or pytest.skip if it should
    not be schema-validated (manifests, exporter outputs)."""
    vs = _schema_dispatch_module()
    kind = vs._classify(path)
    if kind is None:
        if path.name in vs.SKIP_PATTERNS:
            pytest.skip(f"{path.name} is a manifest/exporter output, not result data")
        pytest.fail(
            f"{path.name} has no schema dispatch entry. Add it to "
            f"scripts/validate_schemas.py DISPATCH or to SKIP_PATTERNS."
        )
    return kind


@pytest.mark.parametrize("path", _all_result_files(), ids=lambda p: p.name)
def test_result_json_matches_schema(path):
    """Every result JSON validates against the schema selected by its filename
    pattern. Single source of truth: scripts/validate_schemas.py DISPATCH.
    Catches structural drift and unmatched files in one parametrized sweep."""
    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed")
    kind = _classify_or_skip(path)
    schema_path = REPO / "schema" / f"{kind}.schema.json"
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    data = json.loads(path.read_text())
    errs = list(validator.iter_errors(data))
    assert not errs, f"{path.name} fails {kind} schema:\n" + "\n".join(
        f"  - {'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errs[:5]
    )


# ---------- scopes membership matches the source dict ----------


def test_scopes_match_load_results(scopes_export):
    """Every scope in scopes.json must agree with analysis.load_results.SCOPES."""
    from analysis.load_results import SCOPES

    failures = []
    for name, members in SCOPES.items():
        if name not in scopes_export["scopes"]:
            failures.append(f"scope {name!r} missing from scopes.json")
            continue
        exported = scopes_export["scopes"][name]["models"]
        if members is None:
            if exported is not None:
                failures.append(f"scope {name!r}: source is None, export has list")
            continue
        if sorted(members) != exported:
            failures.append(f"scope {name!r}: membership mismatch")
    assert not failures, "Scope drift:\n" + "\n".join(failures)


# ---------- bridge contract: paths in reports/ resolve in this repo ----------


# File-extension allowlist used to distinguish unambiguous file paths from
# formula text like "learned/random ratio" or "a/b". A token only counts as
# a path if it ends with one of these.
_PATH_EXTENSIONS = (
    ".py",
    ".json",
    ".md",
    ".tex",
    ".sty",
    ".pdf",
    ".txt",
    ".csv",
    ".yaml",
    ".yml",
    ".toml",
    ".cff",
    ".jsonl",
    ".lock",
    ".sh",
    ".png",
)


@pytest.mark.parametrize(
    "path",
    sorted(REPORTS.glob("*.json")),
    ids=lambda p: p.name,
)
def test_reports_paths_resolve_within_repo(path):
    """Every unambiguously path-shaped token in reports/*.json must resolve
    within this repo: no absolute paths, no parent traversal, and any
    leading directory-shaped segment must exist as a directory under REPO.

    A token counts as a path only if it has no whitespace, contains a slash,
    and ends with a recognized file extension. This skips formula text like
    'learned/random ratio'. The rule expresses 'paths must resolve here',
    without naming any sibling repo or offender by hand.

    Cross-repo exemption: when a report's top-level `generated_from_repo`
    field declares a sibling repo as the producer, the `generated_from`
    paths in that report document script locations in the sibling repo
    and are not expected to resolve under this repo's root."""
    if not path.is_file():
        pytest.skip(f"{path.name} not generated")
    data = json.loads(path.read_text())
    cross_repo_paths: set[str] = set()
    if isinstance(data, dict) and data.get("generated_from_repo"):
        if isinstance(data.get("generated_from"), str):
            cross_repo_paths.add(data["generated_from"])
        figures = data.get("figures")
        if isinstance(figures, dict):
            for v in figures.values():
                if isinstance(v, dict) and isinstance(v.get("generated_from"), str):
                    cross_repo_paths.add(v["generated_from"])
    failures: list[str] = []
    for s in _all_strings(data):
        for token in s.split():
            if "/" not in token or not token.endswith(_PATH_EXTENSIONS):
                continue
            if token in cross_repo_paths:
                continue
            if token.startswith("/"):
                failures.append(f"absolute path: {token!r}")
                continue
            if any(part == ".." for part in token.split("/")):
                failures.append(f"parent traversal: {token!r}")
                continue
            head = token.split("/", 1)[0]
            if re.fullmatch(r"[a-z][a-z0-9_-]*", head):
                if not (REPO / head).is_dir():
                    failures.append(f"leading directory {head!r} is not a directory in this repo: {token!r}")
    assert not failures, f"{path.name} carries non-resolvable paths:\n  - " + "\n  - ".join(failures)


# ---------- DISPATCH integrity (validate_schemas.py) ----------


def _dispatch_module():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import validate_schemas as vs  # type: ignore
    finally:
        sys.path.pop(0)
    return vs


def test_dispatch_targets_have_schema_files():
    """Every (suffix -> kind) entry in DISPATCH must point at an existing
    schema file. Prevents 'added a dispatch entry without authoring the
    schema' from passing review and only failing later when a matching JSON
    is validated."""
    vs = _dispatch_module()
    missing = [
        (suffix, kind)
        for suffix, kind in vs.DISPATCH
        if not (REPO / "schema" / f"{kind}.schema.json").is_file()
    ]
    assert not missing, "DISPATCH points at missing schema files:\n" + "\n".join(
        f"  {s} -> {k}.schema.json (not found)" for s, k in missing
    )


def test_dispatch_order_is_longest_suffix_first():
    """Order matters in DISPATCH: a longer suffix that is a tail of a shorter
    one must come first, otherwise it is shadowed and never matches.
    Encodes the invariant that lives only in a comment today."""
    vs = _dispatch_module()
    suffixes = [s for s, _ in vs.DISPATCH]
    bad: list[tuple[str, str]] = []
    for i, s in enumerate(suffixes):
        for longer in suffixes[i + 1 :]:
            if longer.endswith(s) and longer != s:
                bad.append((s, longer))
    assert not bad, "DISPATCH order bug (longer suffix shadowed by earlier shorter one):\n" + "\n".join(
        f"  {s!r} appears before {longer!r}" for s, longer in bad
    )


# ---------- manifest_verification filename convention ----------


def test_manifest_verification_filenames_are_iso_dated():
    """Reports under results/manifest_verification/ must be named
    YYYY-MM-DD.json so that 'the latest' resolves to a sortable predicate
    (lexicographic sort == chronological sort under ISO date)."""
    d = RESULTS / "manifest_verification"
    if not d.is_dir():
        pytest.skip("no manifest verification reports yet")
    bad = [p.name for p in d.glob("*.json") if not re.match(r"^\d{4}-\d{2}-\d{2}\.json$", p.name)]
    assert not bad, (
        "manifest_verification filenames must be YYYY-MM-DD.json "
        f"(lexicographic sort == chronological sort): {bad}"
    )
