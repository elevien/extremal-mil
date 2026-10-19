# extremal-mil

Code to reproduce the figures of

> E. Levien and T. GrandPre, *Learning Through Freezing Phase Transitions in an
> Extremal Regression Problem*.

The package `extremal_mil` implements the multiple-instance regression model
(softmax pooling with a learnable inverse temperature λ), full-batch gradient
descent on it, and the asymptotic gradient flow derived in the paper. Each
figure has a data generator, which writes one `.npz` file per cell under
`notebooks/*_data/`, and a notebook that reads those files and writes the PDF
to `figures/`. No data is included; run the generators first.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Reproducing the figures

Generators skip any cell already on disk, so an interrupted run resumes. Each
cell's runtime is logged to `notebooks/*_data/manifest.jsonl`.

| Figure | Generate data | Notebook | Peak memory | Runtime |
|---|---|---|---|---|
| Fig. 1 (schematic) | none | `fig_intro.ipynb` | — | seconds |
| Figs. 2 and 3 | `python -m extremal_mil.fig2_experiment` | `fig2.ipynb`, `fig3.ipynb` | < 1 GB | hours (d = 150, 1.5×10⁵ steps) |
| Fig. 4 | `python -m extremal_mil.fig4_experiment` | `fig4.ipynb` | < 1 GB | seconds |
| SI, gradient field and λ sweep | `OMP_NUM_THREADS=1 python -m extremal_mil.si_experiment` | `si_figures.ipynb` | ~14 GB (d = 225) | minutes |
| SI, convergence of the initial λ̇ | `OMP_NUM_THREADS=1 python -m extremal_mil.gradient_convergence` | `si_gradient_convergence.ipynb` | set by chunk size | ~1 h, mostly d = 255 and 270 |

Run the notebooks from the `notebooks/` directory. Each one adds `../src` to
the path, so they also work without installing the package.

Every cell is seeded, and the seeds match the runs behind the published
figures.

## Layout

```
src/extremal_mil/
  model.py, data.py, gd.py       model, synthetic data, gradient descent
  theory.py, ode.py              phase boundaries and the asymptotic gradient flow
  convergence.py, sweeps.py      empirical vs. asymptotic gradients; parallel GD sweeps
  fig2_experiment.py             data for Figs. 2 and 3
  fig4_experiment.py             data for Fig. 4
  si_experiment.py               data for the SI gradient-field figures
  gradient_convergence.py        data for the SI convergence figure
  plotting.py, manifest.py       shared plotting style; per-cell run log
notebooks/                       one notebook per figure
tests/                           unit tests
```
