"""Each model-loading script must have the layout/manifest preflight block.

Catches two regressions: a preflight removed from an existing script, or a
new script that calls from_pretrained without the canonical preflight.
Also enforces dataset-revision pinning for paper-cited eval datasets.
"""

import re
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

PINNED_DATASETS = {
    "Salesforce/wikitext",
    "allenai/c4",
    "rajpurkar/squad_v2",
    "GBaker/MedQA-USMLE-4-options",
    "truthfulqa/truthful_qa",
    # Cross-domain datasets pinned end-to-end via revision= in
    # src/transformer_observe.py and scripts/controlled_*.py. These producers
    # are not in SCRIPTS_REQUIRING_PREFLIGHT (controlled-training jobs run
    # separately from the canonical paper protocol), so the per-script
    # preflight check does not run on them; the drift test below still
    # enforces this set equals the manifest keys.
    "Skylion007/openwebtext",
    "code_search_net",
}

# Datasets a paper-cited script may load without a manifest entry. Empty by
# default: every paper-cited load_dataset target should be pinned. Add an
# entry here only with a documented reason (e.g. dataset not used downstream).
DATASETS_EXEMPT_FROM_PINNING: set[str] = set()

# Scripts that load HF models and must carry the canonical preflight.
SCRIPTS_REQUIRING_PREFLIGHT = [
    "dump_tokens.py",
    "mechanistic_llama.py",
    "mechanistic_mistral.py",
    "medqa_selective.py",
    "nonlinear_probe.py",
    "pythia_12b_backfill.py",
    "pythia_1.4b_shuffle.py",
    "pythia_checkpoint_dynamics.py",
    "rag_hallucination.py",
    "roc_width_sweep.py",
    "run_model.py",
    "run_residualizer_split.py",
    "run_stream_model.py",
    "split_bootstrap_gpu.py",
    "truthfulqa_hallucination.py",
]

# Scripts that legitimately do not need the preflight: constant-SHA scripts,
# MPS-only local-dev scripts, and tokenizer-only scripts that train models
# from scratch (no weight download).
SCRIPTS_EXEMPT_FROM_PREFLIGHT = {
    "controlled_depth_width.py",  # tokenizer only, trains from scratch
    "controlled_training.py",  # tokenizer only, trains from scratch
    "gpt2_shuffle_test.py",  # constant SHA
    "mistral7b_instruct_full_mps.py",  # MPS-only, local dev
    "phi3_downstream_mps.py",  # MPS-only, local dev
    "phi3_layer_sweep_mps.py",  # MPS-only, local dev
}


@pytest.mark.parametrize("script_name", SCRIPTS_REQUIRING_PREFLIGHT)
def test_script_has_preflight(script_name):
    """Script contains the canonical comment marker and all three sys.exit checks."""
    src = (SCRIPTS_DIR / script_name).read_text()
    assert "# Fail-fast before model download." in src, f"{script_name}: missing preflight comment marker"
    assert 'sys.exit(f"RESULTS_DIR not found' in src, f"{script_name}: missing RESULTS_DIR.is_dir() check"
    assert 'sys.exit(f"Manifest missing' in src, f"{script_name}: missing manifest existence check"
    assert 'sys.exit(f"Model ' in src, f"{script_name}: missing model-in-manifest check"


@pytest.mark.parametrize("script_name", SCRIPTS_REQUIRING_PREFLIGHT)
def test_script_pins_dataset_revisions(script_name):
    """Every load_dataset call for a manifest dataset must pass revision=."""
    src = (SCRIPTS_DIR / script_name).read_text()
    failures = []
    for ds in PINNED_DATASETS:
        pattern = re.compile(rf'load_dataset\(\s*"{re.escape(ds)}"[^)]*\)', re.DOTALL)
        for match in pattern.finditer(src):
            if "revision=" not in match.group():
                failures.append(f'load_dataset("{ds}", ...) missing revision=')
    assert not failures, f"{script_name}:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("script_name", SCRIPTS_REQUIRING_PREFLIGHT)
def test_no_unprefixed_wikitext(script_name):
    """Use canonical 'Salesforce/wikitext'; the unprefixed name depends on an HF redirect."""
    src = (SCRIPTS_DIR / script_name).read_text()
    assert 'load_dataset("wikitext"' not in src, (
        f'{script_name}: use load_dataset("Salesforce/wikitext", ...) not unprefixed "wikitext"'
    )


@pytest.mark.parametrize("script_name", SCRIPTS_REQUIRING_PREFLIGHT)
def test_loaded_datasets_in_manifest(script_name):
    """Every load_dataset target must be in PINNED_DATASETS or DATASETS_EXEMPT_FROM_PINNING."""
    src = (SCRIPTS_DIR / script_name).read_text()
    found = set(re.findall(r'load_dataset\(\s*"([^"]+)"', src))
    unknown = found - PINNED_DATASETS - DATASETS_EXEMPT_FROM_PINNING
    assert not unknown, (
        f"{script_name}: load_dataset target(s) not in manifest or exempt list: {sorted(unknown)}. "
        f"Add to results/dataset_revisions.json (and PINNED_DATASETS) or to DATASETS_EXEMPT_FROM_PINNING."
    )


@pytest.mark.parametrize("script_name", SCRIPTS_REQUIRING_PREFLIGHT)
def test_script_has_dataset_preflight(script_name):
    """Scripts that load any pinned dataset must carry the dataset-manifest fail-fast block."""
    src = (SCRIPTS_DIR / script_name).read_text()
    if not any(f'"{ds}"' in src for ds in PINNED_DATASETS):
        return
    assert 'sys.exit(f"Dataset manifest missing' in src, (
        f"{script_name}: missing dataset_revisions.json existence check"
    )
    assert "DATASET_REVISIONS" in src, f"{script_name}: missing DATASET_REVISIONS binding"


# Scripts that train models from scratch and so cannot pin a HF revision.
SCRIPTS_EXEMPT_FROM_REVISION_PINNING = {
    "controlled_depth_width.py",  # trains models from scratch (LlamaConfig, no weights)
    "controlled_training.py",  # trains models from scratch (LlamaConfig, no weights)
    "gpt2_shuffle_test.py",  # constant-SHA, _resolved_revision spliced in elsewhere
    # Per-checkpoint revisions: revisions come from the CHECKPOINTS list (training
    # steps), not the manifest commit. Pinning happens via revision= directly.
    "pythia_checkpoint_dynamics.py",
    # MPS local-dev scripts that intentionally do not enforce manifest pinning
    "phi3_layer_sweep_mps.py",
    "phi3_downstream_mps.py",
    "mistral7b_instruct_full_mps.py",
}


@pytest.mark.parametrize("script_name", SCRIPTS_REQUIRING_PREFLIGHT)
def test_script_passes_pinned_revision_to_loaders(script_name):
    """Every from_pretrained call in a preflight-required script must receive
    the manifest-pinned revision (directly via revision=... or via **_rev_kw /
    a load_kwargs dict that itself merges _rev_kw). Catches the class of bug
    where a script preflights the manifest but silently downloads HF HEAD."""
    if script_name in SCRIPTS_EXEMPT_FROM_REVISION_PINNING:
        pytest.skip(f"{script_name} is exempt from revision pinning")

    src = (SCRIPTS_DIR / script_name).read_text()

    # Must define or use a revision-source: either _revision_kwargs(...) or an
    # equivalent inline _rev_kw construction reading the manifest.
    has_rev_kw_binding = bool(re.search(r"_rev_kw\s*=", src))
    has_revision_helper = "_revision_kwargs(" in src
    assert has_rev_kw_binding or has_revision_helper, (
        f"{script_name}: no _rev_kw binding or _revision_kwargs() call. "
        f"Manifest-pinned revisions are not threaded into model loading."
    )

    # Every from_pretrained call must reference _rev_kw, **load_kwargs that
    # merges _rev_kw, or pass revision= directly.
    fp_calls = re.findall(r"from_pretrained\([^)]*\)", src, flags=re.DOTALL)
    assert fp_calls, f"{script_name}: no from_pretrained calls found"

    # Build the set of kwargs-dict names that are known to carry _rev_kw.
    # A line like `load_kwargs = {... **_rev_kw ...}` registers `load_kwargs`.
    rev_carrying_dicts = set(re.findall(r"(\w+)\s*=\s*\{[^}]*\*\*_rev_kw[^}]*\}", src))

    for call in fp_calls:
        passes_revision = (
            "_rev_kw" in call
            or "revision=" in call
            or any(f"**{name}" in call for name in rev_carrying_dicts)
        )
        assert passes_revision, (
            f"{script_name}: from_pretrained call does not receive a pinned revision: {call.strip()}"
        )


def test_all_scripts_compile():
    """Catch syntax errors and global-vs-local errors that ast.parse misses
    (e.g. 'name X used prior to global declaration'). py_compile runs the full
    bytecode compile pass, which is what surfaces these."""
    import py_compile

    failures: list[tuple[str, str]] = []
    for path in SCRIPTS_DIR.glob("*.py"):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            failures.append((path.name, str(e)))
    assert not failures, "Scripts fail to compile:\n" + "\n".join(
        f"  {name}: {err}" for name, err in failures
    )


def test_pythia_12b_backfill_checkpoint_matches_run_model_convention():
    """run_model.py names checkpoints as <output_slug>_checkpoint.json where
    output_slug strips _results.json or .json from --output. The backfill
    script must follow the same convention so a partial run_model.py run can
    be resumed by the backfill without manual file renaming."""
    src = (SCRIPTS_DIR / "pythia_12b_backfill.py").read_text()
    # Must derive the checkpoint name from OUTPUT_PATH (or output filename),
    # not hard-code an unrelated string like "pythia_12b_checkpoint.json".
    assert re.search(
        r'CHECKPOINT_PATH\s*=\s*_resolve_out\(\s*f?"[^"]*\{[^}]*\}_checkpoint\.json"',
        src,
    ), (
        "pythia_12b_backfill.py: CHECKPOINT_PATH must be derived from the "
        "output filename to match run_model.py's <slug>_checkpoint.json naming."
    )


def test_no_unlisted_model_loaders():
    """Catch new scripts that use from_pretrained but lack a preflight registration.

    Any script using Hugging Face from_pretrained must either be in
    SCRIPTS_REQUIRING_PREFLIGHT or explicitly exempted via
    SCRIPTS_EXEMPT_FROM_PREFLIGHT. A new script that goes uncovered would
    waste a multi-GB model download on layout errors.
    """
    listed = set(SCRIPTS_REQUIRING_PREFLIGHT) | SCRIPTS_EXEMPT_FROM_PREFLIGHT
    unlisted = []
    for path in SCRIPTS_DIR.glob("*.py"):
        if path.name in listed:
            continue
        if "from_pretrained" in path.read_text():
            unlisted.append(path.name)
    assert not unlisted, (
        f"Scripts use from_pretrained but are not in SCRIPTS_REQUIRING_PREFLIGHT "
        f"or SCRIPTS_EXEMPT_FROM_PREFLIGHT: {unlisted}"
    )


def test_committed_json_producers_are_registered():
    """Every committed result JSON's provenance.script must point at a script
    that is either in SCRIPTS_REQUIRING_PREFLIGHT or SCRIPTS_EXEMPT_FROM_PREFLIGHT.
    Cross-references the canonical script registry against actual producers,
    catching the case where a new script ships JSONs without ever being
    registered."""
    import json
    from pathlib import Path

    results_dir = SCRIPTS_DIR.parent / "results"
    listed = set(SCRIPTS_REQUIRING_PREFLIGHT) | SCRIPTS_EXEMPT_FROM_PREFLIGHT
    skip_files = {"model_revisions.json", "dataset_revisions.json", "figure_sources.json"}
    unregistered: dict[str, list[str]] = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.name in skip_files:
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        script = (data.get("provenance") or {}).get("script", "")
        if not script:
            continue
        # src/ producers (e.g. transformer_observe.py) live outside scripts/
        # and have their own audit trail; they do not belong in the script
        # preflight registry.
        if script.startswith("src/"):
            continue
        name = Path(script).name
        if name not in listed:
            unregistered.setdefault(name, []).append(p.name)
    assert not unregistered, (
        "Committed JSONs reference producer scripts not in "
        "SCRIPTS_REQUIRING_PREFLIGHT or SCRIPTS_EXEMPT_FROM_PREFLIGHT:\n"
        + "\n".join(f"  {s}: {sorted(fs)}" for s, fs in sorted(unregistered.items()))
    )


def test_pinned_datasets_matches_manifest():
    """The hardcoded PINNED_DATASETS set in this file must equal the keys of
    results/dataset_revisions.json. Catches drift between the test registry
    and the dataset manifest, which would otherwise let a new dataset ship
    without dataset-revision pinning enforcement."""
    import json

    manifest_path = SCRIPTS_DIR.parent / "results" / "dataset_revisions.json"
    if not manifest_path.is_file():
        pytest.skip(f"{manifest_path} not present")
    manifest_keys = set(json.loads(manifest_path.read_text())["datasets"].keys())
    in_tests_only = PINNED_DATASETS - manifest_keys
    in_manifest_only = manifest_keys - PINNED_DATASETS
    assert PINNED_DATASETS == manifest_keys, (
        "PINNED_DATASETS drift between tests and dataset_revisions.json:\n"
        f"  in tests but not manifest: {sorted(in_tests_only)}\n"
        f"  in manifest but not tests: {sorted(in_manifest_only)}"
    )
