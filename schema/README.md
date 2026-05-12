# Result schemas

Every committed result file in this repository validates against a JSON Schema defined here. The schemas cover major result families like main probe runs, checkpoint dynamics, downstream task evaluations, and residualizer split runs. Dispatch happens by filename suffix and the routing table lives in `scripts/validate_schemas.py`. Run `just validate-schemas` from the repo root to check the whole tree at once.
