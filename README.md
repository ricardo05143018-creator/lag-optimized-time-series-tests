# Lag-Optimized Time-Series Tests

This repository contains the code, generated outputs, and manuscript for *Lag-Optimized Time-Series Tests*.

It was created as a compact reproducibility repository for the paper. The earlier development of the project is documented separately in [Cross-Domain Error Dynamics](https://github.com/ricardo05143018-creator/cross-domain-error-dynamics).

## Background

The paper grew out of a question I first explored in *Cross-Domain Error Dynamics*: whether prediction errors from unrelated domains could show meaningful synchronous or lagged relationships.

As that project developed, the focus shifted from finding correlations to asking whether those correlations were statistically trustworthy. In particular, scanning across multiple lags and working with serially dependent time series raised problems that the earlier exploratory methods did not fully handle.

This paper develops that statistical question more directly.

## Repository contents

- `src/` — analysis and figure-generation scripts
- `output/figures/` — figures used in the paper
- `output/tables/` — numerical results from the simulations
- `paper/` — the manuscript
- `requirements.txt` — Python dependencies

## Reproducing the analysis

Install the required dependencies with:

```bash
python -m pip install -r requirements.txt
```

The analysis scripts are in `src/`. Generated figures and numerical outputs are included in `output/` so they can be compared with the results reported in the manuscript.

## Note on repository history

Most of the research development took place before this repository was assembled. This repository therefore records the final paper materials and reproducibility files rather than the full development process.

For the earlier version-by-version development, see [Cross-Domain Error Dynamics](https://github.com/ricardo05143018-creator/cross-domain-error-dynamics).
