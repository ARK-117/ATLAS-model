# ATLAS Model

A standalone, standard-library Python research package for testing the mathematics behind ATLAS.
It contains ordinary least squares and ridge regression, time-ordered feature/label construction,
a five-bar breakout baseline, portfolio risk checks, and cost-aware historical paper evaluation.
The examples generate synthetic data. No pretrained weights or proven profitable model are supplied.

## Run

Install Git, Python 3.11 or newer, and Node.js 24 for the development checks.

~~~powershell
git clone https://github.com/ARK-117/ATLAS-model.git
cd ATLAS-model
npm ci --ignore-scripts
python -B -S -m atlas.checks
python -B -S -m atlas.backtest_cli
python -B -S -m atlas.regression_cli
python -B -S -m atlas.regression_cli --trace-date 2026-05-21
~~~

The runtime requires only Python's standard library. Node.js installs the pinned development type checker.
The full check runs syntax, types, and the model's Python tests. JSON research output is printed in the terminal.
VS Code also includes model, regression, explanation, and quality tasks.

See [the model guide](docs/MODEL_GUIDE.md) for formulas, inputs, assumptions, and an independent-testing checklist.
The public package includes model code, synthetic fixtures, tests, and development tooling.
The private application separately maintains the dashboard, data acquisition, research exports, journals,
and operating evidence. Live orders remain locked.

## Version provenance

EXPORT_MANIFEST.json lists the originating private commit and SHA-256 of every exported file.
Copied model and test files retain their source bytes. Public changes should be reviewed in pull requests;
they do not automatically modify the private application. Each later export is an explicit reviewed update.
