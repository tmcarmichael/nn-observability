"""Lint for hardcoded data-dependent numbers in .tex files.

Parses data_macros.sty to get all macro values, then scans .tex files
for those same values appearing as literals instead of macro references.
Catches the "someone typed 0.263 instead of \\qThreepcorr" problem.

Usage: cd nn-observability && uv run python analysis/lint_hardcoded.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT.parent / "nn-observability-paper"

# Files to scan (prose sections, not generated tables)
SCAN_DIRS = [
    PAPER_DIR / "sections",
    PAPER_DIR / "appendix",
]

# Skip these files (generated content, handled by check-tables)
SKIP_FILES = set()

# Macros where the value legitimately appears in prose without the macro
# (e.g., the value appears in a different context than the macro's meaning)
KNOWN_COLLISIONS = {
    # +0.286 is both Llama 1B pcorr and GPT-2 Large pcorr
    # The lint can't distinguish context, so these produce false positives
}


def _load_macros() -> dict[str, str]:
    """Parse data_macros.sty into name->value dict."""
    macros = {}
    path = PAPER_DIR / "data_macros.sty"
    for line in path.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", line)
        if m:
            macros[m.group(1)] = m.group(2)
    return macros


def _is_macro_ref(line: str, pos: int) -> bool:
    """Check if the number at position pos is part of a macro reference."""
    # Look backwards for a backslash indicating this is a macro expansion
    before = line[:pos].rstrip()
    if before.endswith("\\"):
        return True
    # Check if we're inside a macro name (e.g., \qThreepcorr expands to +0.263)
    # The expanded form won't have a backslash, but we check if the line
    # contains the macro name near this position
    return False


def _find_hardcoded(macros: dict[str, str]) -> list[tuple[str, int, str, str, list[str]]]:
    """Find literal macro values in tex files. Returns (file, line, value, context, macro_names)."""
    # Build reverse map: value -> list of macro names
    value_to_macros: dict[str, list[str]] = {}
    for name, val in macros.items():
        # Only check numeric values with 3 decimal places (the standard pcorr format)
        val_clean = val.strip()
        if not re.match(r"^[+-]?\d+\.\d{3}$", val_clean):
            continue
        # Skip small absolute values that collide with unrelated numbers
        abs_val = abs(float(val_clean))
        if abs_val < 0.01:
            continue
        value_to_macros.setdefault(val_clean, []).append(name)

    # Deduplicate: track (file, line, position) to avoid reporting +0.263 and 0.263 at same spot
    seen: set[tuple[str, int, int]] = set()
    findings = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for tex_file in sorted(scan_dir.glob("*.tex")):
            if tex_file.name in SKIP_FILES:
                continue
            lines = tex_file.read_text().splitlines()
            for line_num, line in enumerate(lines, 1):
                stripped = line.lstrip()
                if stripped.startswith("%"):
                    continue
                # Skip alt-text in figure includes (descriptive, not data claims)
                if "alt={" in line:
                    continue
                for val, macro_names in value_to_macros.items():
                    pattern = re.escape(val)
                    for match in re.finditer(rf"(?<![0-9.]){pattern}(?![0-9])", line):
                        pos = match.start()
                        # Deduplicate: if we already flagged this position, skip
                        # (handles +0.263 and 0.263 matching the same literal)
                        key = (str(tex_file), line_num, pos if val.startswith(("+", "-")) else pos - 1)
                        if key in seen:
                            continue
                        seen.add(key)
                        # Also mark the alternate form as seen
                        alt_key = (
                            str(tex_file),
                            line_num,
                            pos + 1 if not val.startswith(("+", "-")) else pos,
                        )
                        seen.add(alt_key)

                        is_expanded = any(f"\\{mn}" in line for mn in macro_names)
                        if is_expanded:
                            continue
                        start = max(0, pos - 20)
                        end = min(len(line), pos + len(val) + 20)
                        context = line[start:end].strip()
                        rel_path = tex_file.relative_to(PAPER_DIR)
                        findings.append((str(rel_path), line_num, val, context, macro_names))

    return findings


def main():
    if not PAPER_DIR.exists():
        print(f"ERROR: paper directory not found: {PAPER_DIR}")
        sys.exit(1)

    macros = _load_macros()
    print(f"Loaded {len(macros)} macros from data_macros.sty")

    findings = _find_hardcoded(macros)

    if not findings:
        print("OK: no hardcoded macro values found in tex files")
        sys.exit(0)

    print(f"\nFound {len(findings)} hardcoded value(s) that could use macros:\n")
    for filepath, line_num, val, context, macro_names in findings:
        macros_str = ", ".join(f"\\{n}" for n in macro_names)
        print(f"  {filepath}:{line_num}: {val}")
        print(f"    context: ...{context}...")
        print(f"    macro(s): {macros_str}")
        print()

    print(f"WARN: {len(findings)} hardcoded value(s) found")
    print("  Replace with macros where appropriate, or add to KNOWN_COLLISIONS if intentional")
    # Exit 0 (warning, not failure) since some may be false positives
    # Use --strict to make this a hard failure
    if "--strict" in sys.argv:
        sys.exit(1)


if __name__ == "__main__":
    main()
