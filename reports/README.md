# Reports

This folder holds derived metadata that bridges the paper to the code. `paper_values.json` lists every macro the paper cites with its value, the result files it draws from, and the formula that recomputes it. `scopes.json` mirrors the named model cohorts from `analysis/load_results.py`, and `figure_sources.json` maps each committed figure to its generator script in the paper repository. The paper repository regenerates these files, and the tests in `tests/test_paper_values.py` keep the committed copies aligned with what a fresh regen would produce.
