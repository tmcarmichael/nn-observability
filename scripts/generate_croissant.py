"""Generate croissant.json metadata descriptor from results/ and schema/.

Builds a Croissant 1.1 metadata descriptor (Core + recordSet) per the
official spec at https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html.

Sources of truth:
  - pyproject.toml: name, version, description, keywords, license
  - CITATION.cff: citation text, DOI, date
  - LICENSE: SPDX license name
  - results/*.json: distribution file inventory + sha256
  - results/manifest_verification/*.json: latest verification report
  - schema/*.schema.json: per-record-type field definitions

Usage:
  uv run python scripts/generate_croissant.py            # write croissant.json
  uv run python scripts/generate_croissant.py --check    # diff against committed file
  uv run python scripts/generate_croissant.py --quiet    # suppress progress lines
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SCHEMA = REPO / "schema"
OUT = REPO / "croissant.json"

# Map filename suffix -> (file-type id, schema name, human description).
# Mirrors scripts/validate_schemas.py DISPATCH so a new file type added there
# is visible here too.
FILE_TYPES: list[tuple[str, str, str, str]] = [
    (
        "_main.json",
        "main-results",
        "main",
        "Per-model output of the canonical 350 ex/dim 7-seed observability protocol.",
    ),
    (
        "_nonlinear-probe-multilayer.json",
        "nonlinear-probe-multilayer-results",
        "nonlinear-probe",
        "Multi-layer nonlinear-probe sweep on Llama 3B.",
    ),
    (
        "_nonlinear-probe.json",
        "nonlinear-probe-results",
        "nonlinear-probe",
        "Matched-HP and swept-HP MLP-vs-linear comparison per model.",
    ),
    (
        "_dynamics.json",
        "dynamics-results",
        "dynamics",
        "Pythia checkpoint trajectory: pcorr and oc_residual across training steps.",
    ),
    (
        "_residualizer-split.json",
        "residualizer-split-results",
        "residualizer-split",
        "Split-fit OLS residualizer test (separate fit pool from probe pool).",
    ),
    (
        "_mechanistic.json",
        "mechanistic-results",
        "mechanistic",
        "Mean-ablation patching by layer/component (sign-only interpretation).",
    ),
    (
        "_squad-rag.json",
        "squad-results",
        "downstream",
        "SQuAD 2.0 reading-comprehension exclusive-catch rates by flag rate.",
    ),
    (
        "_medqa.json",
        "medqa-results",
        "downstream",
        "MedQA-USMLE multiple-choice exclusive-catch rates by flag rate.",
    ),
    (
        "_truthfulqa.json",
        "truthfulqa-results",
        "downstream",
        "TruthfulQA generation exclusive-catch rates and AUC by flag rate.",
    ),
    (
        "_shuffle-control.json",
        "shuffle-control-results",
        "shuffle-control",
        "Shuffled-label probe null distribution (10 permutations).",
    ),
    (
        "_bootstrap.json",
        "bootstrap-results",
        "bootstrap",
        "Document-level bootstrap on Qwen 7B (30 resamples).",
    ),
    (
        "_width-sweep.json",
        "width-sweep-results",
        "width-sweep",
        "Output-side MLP width sweep (64-512 units) on Qwen 7B.",
    ),
]
LEGACY_PATTERNS = (
    "_sae-comparison.json",
    "_bottleneck-scaling.json",
    "_exdim-1000.json",
    "_exdim-sweep.json",
    "transformer_observe.json",
)
SKIP = {"model_revisions.json", "dataset_revisions.json", "figure_sources.json"}

# JSON Schema "type" -> Croissant dataType IRI.
TYPE_MAP = {
    "string": "sc:Text",
    "integer": "sc:Integer",
    "number": "sc:Float",
    "boolean": "sc:Boolean",
    "array": "sc:Text",  # arrays serialized inline
    "object": "sc:Text",  # nested objects serialized inline
}

# Croissant 1.1 Appendix 1 with one URL-scheme adjustment: schema.org is
# referenced as https://schema.org/ rather than http://. RDF treats them as
# the same vocabulary, but the mlcroissant validator (built before the 1.1
# spec landed) does strict string matching on Dataset URI and rejects http://.
CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "dct": "http://purl.org/dc/terms/",
    "annotation": "cr:annotation",
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "excludes": "cr:excludes",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "readLines": "cr:readLines",
    "sdVersion": "cr:sdVersion",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "unArchive": "cr:unArchive",
    "value": "cr:value",
}

CONFORMS_TO = [
    "http://mlcommons.org/croissant/1.1",
    "http://mlcommons.org/croissant/RAI/1.0",
]
SD_VERSION = "1.1"


def _read_pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]


def _read_citation() -> dict:
    text = (REPO / "CITATION.cff").read_text()
    fields = {}
    for line in text.splitlines():
        m = re.match(r'^(version|date-released|doi|repository-code|title): "?([^"]+?)"?$', line)
        if m:
            fields[m.group(1)] = m.group(2)
        m_orcid = re.match(r'^\s*orcid:\s*"(https://orcid\.org/[\w-]+)"\s*$', line)
        if m_orcid:
            fields["orcid"] = m_orcid.group(1)
    return fields


def _creator(citation: dict) -> dict:
    creator: dict = {"@type": "Person", "name": "Thomas Carmichael"}
    orcid = citation.get("orcid")
    if orcid:
        creator["sameAs"] = orcid
        creator["identifier"] = orcid
    return creator


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify(path: Path) -> tuple[str, str, str, str] | None:
    """Return (fileset_id, schema_name, description, suffix) for a result file, or None."""
    name = path.name
    if name in SKIP or name.startswith("."):
        return None
    for suffix, fileset_id, schema_name, desc in FILE_TYPES:
        if name.endswith(suffix):
            return fileset_id, schema_name, desc, suffix
    if any(name.endswith(p) for p in LEGACY_PATTERNS):
        return (
            "legacy-results",
            "legacy",
            "Pre-v3.0.0 phase-taxonomy outputs preserved as provenance.",
            "legacy",
        )
    return None


def _load_schema(name: str) -> dict | None:
    path = SCHEMA / f"{name}.schema.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _schema_to_fields(schema: dict, fileset_id: str) -> list[dict]:
    """Map JSON Schema top-level properties to Croissant Field entries."""
    properties = schema.get("properties", {})
    fields = []
    for name, spec in properties.items():
        js_type = spec.get("type", "string")
        if isinstance(js_type, list):
            js_type = next((t for t in js_type if t != "null"), "string")
        data_type = TYPE_MAP.get(js_type, "sc:Text")
        field = {
            "@type": "cr:Field",
            "@id": f"{fileset_id}-records/{name}",
            "name": name,
            "description": spec.get("description", "").strip() or f"Field {name} from {fileset_id}.",
            "dataType": data_type,
            "source": {
                "fileSet": {"@id": fileset_id},
                "extract": {"jsonPath": f"$.{name}"},
            },
        }
        if js_type in ("array", "object"):
            field["description"] = (field["description"] + " (JSON-serialized)").strip()
        fields.append(field)
    return fields


def _archive_sha256(grouped: dict[str, list[Path]]) -> str:
    """Deterministic merkle hash over (filename, sha256) pairs of every distribution file.

    Verifies dataset integrity without downloading a tarball: rerun the
    generator on a clean clone and the same hash appears.
    """
    pairs: list[tuple[str, str]] = []
    for paths in grouped.values():
        for p in paths:
            pairs.append((p.name, _sha256(p)))
    for special in ("model_revisions.json", "dataset_revisions.json"):
        pairs.append((special, _sha256(RESULTS / special)))
    for p in sorted((RESULTS / "manifest_verification").glob("*.json")):
        pairs.append((p.name, _sha256(p)))
    h = hashlib.sha256()
    for name, sha in sorted(pairs):
        h.update(f"{name}:{sha}\n".encode())
    return h.hexdigest()


def _hf_record_sets() -> list[dict]:
    """First-class Croissant RecordSets enumerating the pinned HF models and datasets.

    Each upstream model and dataset is exposed as a queryable record with the
    canonical HF URL, pinned commit SHA, license identifier, and URL to the
    upstream license text. This lets a Croissant-aware tool rebuild the full
    provenance cohort from the descriptor alone, without re-parsing
    model_revisions.json / dataset_revisions.json.
    """
    models = json.loads((RESULTS / "model_revisions.json").read_text())["models"]
    datasets = json.loads((RESULTS / "dataset_revisions.json").read_text())["datasets"]

    def _fields(rs_id: str, specs: list[tuple[str, str]]) -> list[dict]:
        # specs: list of (field_name, description); @id is rs_id/field_name.
        return [
            {
                "@type": "cr:Field",
                "@id": f"{rs_id}/{name}",
                "name": name,
                "description": desc,
                "dataType": "sc:Text",
            }
            for name, desc in specs
        ]

    def _records(rs_id: str, items: list[dict]) -> list[dict]:
        # mlcroissant inline records key fields by @id, not name.
        return [{f"{rs_id}/{k}": v for k, v in item.items()} for item in items]

    model_fields = [
        ("model_id", "Hugging Face model identifier (org/name)."),
        ("commit", "Pinned Hugging Face commit SHA (40-char hex git revision)."),
        ("url", "Canonical Hugging Face URL for the model repository."),
        ("contentUrl", "Hugging Face tree URL pinned to the recorded commit."),
        ("license", "License identifier (SPDX where available, otherwise the upstream label)."),
        ("license_url", "URL to the upstream license text."),
    ]
    dataset_fields = [
        ("dataset_id", "Hugging Face dataset identifier (org/name)."),
        ("config", "Dataset config / subset name, or empty string if unconfigured."),
        ("commit", "Pinned Hugging Face commit SHA (40-char hex git revision)."),
        ("url", "Canonical Hugging Face URL for the dataset repository."),
        ("contentUrl", "Hugging Face tree URL pinned to the recorded commit."),
        ("license", "License identifier (SPDX where available, otherwise the upstream label)."),
        ("license_url", "URL to the upstream license text."),
    ]

    model_items = [
        {
            "model_id": mid,
            "commit": entry["commit"],
            "url": entry["url"],
            "contentUrl": f"{entry['url']}/tree/{entry['commit']}",
            "license": entry.get("license", ""),
            "license_url": entry.get("license_url", ""),
        }
        for mid, entry in sorted(models.items())
    ]
    dataset_items = [
        {
            "dataset_id": did,
            "config": entry.get("config") or "",
            "commit": entry["commit"],
            "url": entry["url"],
            "contentUrl": f"{entry['url']}/tree/{entry['commit']}",
            "license": entry.get("license", ""),
            "license_url": entry.get("license_url", ""),
        }
        for did, entry in sorted(datasets.items())
    ]

    return [
        {
            "@type": "cr:RecordSet",
            "@id": "hugging-face-models",
            "name": "hugging-face-models",
            "description": (
                "All Hugging Face transformer checkpoints evaluated in the paper. Each "
                "record carries the model identifier, pinned commit SHA, family-level "
                "license, and the URL to the upstream license text. The cross-family "
                "cohort and Pythia controlled suite are defined as named scopes in "
                "analysis/load_results.py."
            ),
            "key": {"@id": "hugging-face-models/model_id"},
            "field": _fields("hugging-face-models", model_fields),
            "data": _records("hugging-face-models", model_items),
        },
        {
            "@type": "cr:RecordSet",
            "@id": "hugging-face-datasets",
            "name": "hugging-face-datasets",
            "description": (
                "All Hugging Face datasets used as eval corpora or producer inputs in "
                "the paper. Each record carries the dataset identifier, config name, "
                "pinned commit SHA, license, and the URL to the upstream license text. "
                "Producer scripts pass the recorded commit via the revision= argument "
                "to load_dataset, enforced by tests/test_script_preflight.py."
            ),
            "key": {"@id": "hugging-face-datasets/dataset_id"},
            "field": _fields("hugging-face-datasets", dataset_fields),
            "data": _records("hugging-face-datasets", dataset_items),
        },
    ]


def _file_objects(citation: dict, grouped: dict[str, list[Path]]) -> list[dict]:
    """Parent archive + singleton FileObjects for the pinned manifests."""
    repo_url = citation["repository-code"].rstrip("/")
    version = citation["version"]
    manifests = sorted((RESULTS / "manifest_verification").glob("*.json"))
    latest = manifests[-1] if manifests else None
    out = [
        {
            "@type": "cr:FileObject",
            "@id": "nn-observability-archive",
            "name": "nn-observability-archive",
            "description": (
                f"Source repository at v{version}. The sha256 is a merkle hash over "
                "(filename, sha256) pairs of every distribution file under results/, "
                "computed by scripts/generate_croissant.py; rerun the generator on a "
                "clean clone to verify."
            ),
            "contentUrl": f"{repo_url}/archive/refs/tags/v{version}.tar.gz",
            "encodingFormat": "application/x-tar",
            "sha256": _archive_sha256(grouped),
        },
        {
            "@type": "cr:FileObject",
            "@id": "model-revisions",
            "name": "model-revisions",
            "description": (
                "Hugging Face model IDs, pinned commit hashes, and per-model license "
                "attribution for every evaluated model. Llama 3.2 / 3.1 entries carry the "
                "Llama Community License; Gemma 3 entries carry the Gemma license; the "
                "remaining five families ship under MIT or Apache 2.0. See MODELS.md for the "
                "license-compatibility table and downstream-derivative constraints."
            ),
            "containedIn": {"@id": "nn-observability-archive"},
            "contentUrl": "results/model_revisions.json",
            "encodingFormat": "application/json",
            "sha256": _sha256(RESULTS / "model_revisions.json"),
        },
        {
            "@type": "cr:FileObject",
            "@id": "dataset-revisions",
            "name": "dataset-revisions",
            "description": (
                "Hugging Face dataset IDs, pinned commit hashes, and per-dataset license "
                "attribution for every paper-cited evaluation corpus. Per-entry licenses span "
                "MIT, Apache 2.0, CC0, CC BY 4.0, CC BY-SA 4.0, and ODC-BY 1.0. See DATA.md "
                "for the license-compatibility table and downstream-derivative constraints."
            ),
            "containedIn": {"@id": "nn-observability-archive"},
            "contentUrl": "results/dataset_revisions.json",
            "encodingFormat": "application/json",
            "sha256": _sha256(RESULTS / "dataset_revisions.json"),
        },
    ]
    if latest is not None:
        out.append(
            {
                "@type": "cr:FileObject",
                "@id": "manifest-verification",
                "name": "manifest-verification",
                "description": "Programmatic verification of every model_revisions entry "
                "against the live Hugging Face API.",
                "containedIn": {"@id": "nn-observability-archive"},
                "contentUrl": f"results/manifest_verification/{latest.name}",
                "encodingFormat": "application/json",
                "sha256": _sha256(latest),
            }
        )
    return out


def _file_sets(grouped: dict[str, list[Path]], fileset_meta: dict[str, str]) -> list[dict]:
    """One FileSet per file-type group, each containedIn the parent archive."""
    out = []
    for fileset_id in sorted(grouped):
        out.append(
            {
                "@type": "cr:FileSet",
                "@id": fileset_id,
                "name": fileset_id,
                "description": fileset_meta[fileset_id],
                "containedIn": {"@id": "nn-observability-archive"},
                "encodingFormat": "application/json",
                "includes": [f"results/{p.name}" for p in sorted(grouped[fileset_id])],
            }
        )
    return out


def _record_sets(grouped: dict[str, list[Path]], schemas: dict[str, str]) -> list[dict]:
    """One RecordSet per file type, fields derived from the matching JSON Schema."""
    out = []
    for fileset_id in sorted(grouped):
        schema_name = schemas[fileset_id]
        schema = _load_schema(schema_name)
        if schema is None:
            continue
        out.append(
            {
                "@type": "cr:RecordSet",
                "@id": f"{fileset_id}-records",
                "name": f"{fileset_id}-records",
                "description": (schema.get("description") or f"Records derived from {fileset_id}."),
                "key": {"@id": f"{fileset_id}-records/model"}
                if "model" in schema.get("properties", {})
                else None,
                "field": _schema_to_fields(schema, fileset_id),
            }
        )
        # Drop the optional "key" if there's no "model" field.
        if out[-1]["key"] is None:
            del out[-1]["key"]
    return out


def _walk_results() -> tuple[dict[str, list[Path]], dict[str, str], dict[str, str]]:
    """Group results files by file-type id. Return (groups, descriptions, schemas)."""
    groups: dict[str, list[Path]] = {}
    descriptions: dict[str, str] = {}
    schemas: dict[str, str] = {}
    unmatched: list[Path] = []
    for p in sorted(RESULTS.glob("*.json")):
        info = _classify(p)
        if info is None:
            if p.name not in SKIP:
                unmatched.append(p)
            continue
        fileset_id, schema_name, desc, _ = info
        groups.setdefault(fileset_id, []).append(p)
        descriptions[fileset_id] = desc
        schemas[fileset_id] = schema_name
    if unmatched:
        names = ", ".join(p.name for p in unmatched)
        sys.exit(f"FAIL: unmatched result file(s): {names}")
    return groups, descriptions, schemas


def build() -> dict:
    pyproj = _read_pyproject()
    citation = _read_citation()
    groups, descriptions, schemas = _walk_results()

    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CONFORMS_TO,
        "sdVersion": SD_VERSION,
        "name": pyproj["name"],
        "description": pyproj["description"],
        "url": citation["repository-code"],
        "documentation": [
            f"{citation['repository-code']}/blob/main/DATA.md",
            f"{citation['repository-code']}/blob/main/MODELS.md",
        ],
        "version": citation["version"],
        "datePublished": citation["date-released"],
        "license": "https://opensource.org/licenses/MIT",
        "rai:dataCollection": (
            "This repository is a consumer of seven public Hugging Face datasets, not a "
            "creator. No data collection occurred here. Each dataset is loaded at a pinned "
            "revision (results/dataset_revisions.json) and used as the source for tokenized "
            "text inputs and downstream-task items. Upstream collection methods are "
            "documented on each dataset's Hugging Face card and source publication; see "
            "DATA.md for per-dataset role and reference links."
        ),
        "rai:dataCollectionType": "https://schema.org/Dataset",
        "rai:dataPreprocessingProtocol": (
            "Per-dataset preprocessing is documented in DATA.md (subset and transforms per "
            "dataset). Probe targets are partial-correlation residuals after controlling for "
            "max-softmax confidence and activation norm, computed by analysis code shipped "
            "in src/probe.py and src/observe.py and locked at 1e-12 tolerance by "
            "tests/test_producer_invariants.py."
        ),
        "rai:annotationsPerItem": (
            "No annotations are produced by this repository. Persisted artifacts are derived "
            "statistics (per-token loss values, observer scores, partial correlations, "
            "exclusive-catch counts), not labels assigned to source-text items."
        ),
        "rai:dataAnnotationProtocol": (
            "Not applicable. No annotations are produced by this repository. Upstream "
            "datasets carry their own annotation protocols, documented in their respective "
            "Hugging Face cards and source publications (see DATA.md per-dataset entries "
            "for citations)."
        ),
        "rai:personalSensitiveInformation": (
            "Per-dataset PII posture mirrors DATA.md (PII and sensitive content section). "
            "WikiText-103, SQuAD v2, MedQA-USMLE, and TruthfulQA are low-risk: encyclopedic "
            "or synthetic content with no human subjects beyond the upstream Wikipedia or "
            "exam-board sources. C4 and OpenWebText carry web-scrape residuals from public "
            "internet text. CodeSearchNet (Python) carries author-identifier residuals from "
            "open-source GitHub comments and string literals. This repository does not "
            "collect, redistribute, or expose source-text content; persisted artifacts are "
            "derived statistics that are not text-recoverable."
        ),
        "rai:dataUseCases": (
            "Recommended: reproducing the observability protocol on the same models, "
            "extending it to additional autoregressive transformers under the same probe "
            "class, using the WikiText-trained observer as a starting point for "
            "methodological hardening, and probing-validity research that builds on the "
            "confidence-controlled and output-controlled framework. Out-of-scope: "
            "production or regulated deployment in clinical, legal, financial, hiring, "
            "lending, or content-moderation contexts; generalization to multilingual or "
            "non-text settings without re-validation; detection of fluent confident "
            "falsehoods (the paper reports near-chance TruthfulQA AUC across three "
            "production instruct models); adversarial-robustness claims under adaptive "
            "attacks. See DATA.md and MODELS.md for the full statements."
        ),
        "rai:dataLimitations": (
            "All seven datasets are English-language. The paper's findings are scoped to "
            "English text and the populations represented (Wikipedia editors, web users, "
            "crowdworkers, US medical board examiners, open-source Python developers, "
            "adversarial fact-pattern selection). Cross-language and cross-demographic "
            "generalization is not measured. Activation-monitor robustness under adaptive "
            "attacks is not measured (McGuinness et al., 2025). Upstream creators do not "
            "release training-data composition for several model families (Llama, Gemma, "
            "Mistral, Qwen, Phi); cross-recipe and cross-family contrasts are observational, "
            "not causal."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "Versioned releases tagged on GitHub and snapshotted on Zenodo (concept DOI in "
            "this descriptor's `sameAs`). Every release pins per-model and per-dataset "
            "Hugging Face SHAs verified against the live HF API by "
            "scripts/verify_manifest_revisions.py; the latest verification report is "
            "exposed through the manifest-verification FileObject. Schema validation runs "
            "on every push; result-JSON drift is blocked by CI. The Croissant descriptor is "
            "regenerated on every release."
        ),
        "usageInfo": (
            "Repository source code, result schemas, and human-readable documentation are MIT-"
            "licensed (this descriptor's `license` field). Result JSON values are statistical "
            "summaries derived from frozen activations of upstream public models; model "
            "weights are never redistributed, and downstream use of result JSON values must "
            "respect the upstream terms recorded in NOTICE. Per-model and per-dataset license attribution "
            "is recorded in results/model_revisions.json and results/dataset_revisions.json "
            "(also exposed through the model-revisions and dataset-revisions FileObjects, "
            "and as queryable records in the hugging-face-models and hugging-face-datasets "
            "RecordSets). Built with Llama: this repository evaluates Llama 3.2 1B, 3B, "
            "1B-Instruct, and Llama 3.1 8B; downstream redistributors of derived artifacts "
            'include the "Built with Llama" attribution required by the Llama Community '
            "License and remain bound by Meta's Acceptable Use Policy. Gemma 3 derivatives "
            "are governed by the Gemma Terms of Use, including the Prohibited Use Policy. "
            "Qwen 2.5 3B base and 3B Instruct are released under the Qwen Research License "
            "(custom, non-commercial); derivatives that depend on these two checkpoints "
            "inherit the non-commercial restriction. The other 25 evaluated checkpoints "
            "ship under permissive licenses (MIT or Apache 2.0). See MODELS.md, DATA.md, "
            "and the top-level NOTICE file for the per-family compatibility tables and "
            "required attribution."
        ),
        "citeAs": (
            f"Carmichael, T. ({citation['date-released'][:4]}). "
            f"{citation['title']}. Zenodo. https://doi.org/{citation['doi']}"
        ),
        "creator": _creator(citation),
        "keywords": pyproj["keywords"],
        "sameAs": f"https://doi.org/{citation['doi']}",
        "isLiveDataset": False,
        "distribution": _file_objects(citation, groups) + _file_sets(groups, descriptions),
        "recordSet": _hf_record_sets() + _record_sets(groups, schemas),
    }


def _emit(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check", action="store_true", help="Diff regenerated content against committed croissant.json."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args(argv)

    text = _emit(build())

    if args.check:
        if not OUT.is_file():
            print("FAIL: croissant.json missing; run without --check to generate.")
            return 1
        committed = OUT.read_text()
        if committed != text:
            print("FAIL: croissant.json out of date. Re-run scripts/generate_croissant.py.")
            return 1
        if not args.quiet:
            print(f"OK: croissant.json matches generator (len={len(text)} bytes).")
        return 0

    OUT.write_text(text)
    if not args.quiet:
        n_files = sum(len(v) for v in _walk_results()[0].values())
        print(f"Wrote {OUT.relative_to(REPO)} ({len(text)} bytes, {n_files} result files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
