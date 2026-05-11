"""Export named scopes from analysis.load_results.SCOPES to reports/scopes.json.

The named scopes (cross_family_14, pythia_controlled_9, ...) are the
Python-side definitions of which models a paper claim covers. Exporting
them as JSON lets downstream consumers read the membership without
importing the analysis package.

Usage:
    uv run python scripts/export_scopes.py            # write
    uv run python scripts/export_scopes.py --check    # content diff
"""

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from analysis.load_results import SCOPES

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "reports" / "scopes.json"

SCHEMA_VERSION = "1.0"

SCOPE_DESCRIPTIONS = {
    "cross_family_14": "Cross-family aggregate scope (paper Sections 3 and 5).",
    "control_sensitivity_14": "Alias for cross_family_14; legacy scope name retained for backward compatibility.",
    "absorption_cohort_14": "14-model cohort for the headline confidence-absorption statistic (paper macro confabsorbmean).",
    "pythia_controlled_9": "Pythia controlled suite (paper Section 4).",
    "all": "All loaded models; no scope filter applied.",
}

VOLATILE_FIELDS = ("generated_at",)


def build_export() -> dict:
    scopes_out = {}
    for name, members in SCOPES.items():
        scopes_out[name] = {
            "description": SCOPE_DESCRIPTIONS.get(name, ""),
            "models": sorted(members) if members is not None else None,
            "n_models": len(members) if members is not None else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "generated_from": "scripts/export_scopes.py",
        "source": "analysis/load_results.py",
        "scopes": scopes_out,
    }


def _strip_volatile(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in VOLATILE_FIELDS}


def _serialize(d: dict) -> str:
    return json.dumps(d, indent=2) + "\n"


def do_write() -> None:
    out = build_export()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(_serialize(out))
    n = sum(1 for v in out["scopes"].values() if v["models"] is not None)
    print(f"Exported {n} named scopes to {OUTPUT.relative_to(REPO)}")


def do_check() -> int:
    if not OUTPUT.is_file():
        print(f"FAIL: {OUTPUT} does not exist; run 'just export-scopes' to generate")
        return 1
    committed = json.loads(OUTPUT.read_text())
    fresh = build_export()
    if _strip_volatile(committed) != _strip_volatile(fresh):
        print(f"FAIL: {OUTPUT.relative_to(REPO)} differs from generated output")
        committed_scopes = committed.get("scopes", {})
        fresh_scopes = fresh.get("scopes", {})
        for name in set(committed_scopes) ^ set(fresh_scopes):
            origin = "committed" if name in committed_scopes else "fresh"
            print(f"  presence: scope {name!r} only in {origin}")
        for name in set(committed_scopes) & set(fresh_scopes):
            if committed_scopes[name] != fresh_scopes[name]:
                print(f"  content: scope {name!r} membership differs")
        print("\nRegenerate via 'just export-scopes' (or 'just sync').")
        return 1
    n = sum(1 for v in fresh["scopes"].values() if v["models"] is not None)
    print(f"OK: {OUTPUT.relative_to(REPO)} matches generated output ({n} named scopes)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed file matches generated output (no writes).",
    )
    args = parser.parse_args()
    if args.check:
        sys.exit(do_check())
    do_write()


if __name__ == "__main__":
    main()
