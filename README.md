# EOF-based sensor placement for marine monitoring

Python implementation used for EOF-based three-dimensional sensor placement and sparse reconstruction of temperature and salinity fields in the Weser Estuary case study.

The workflow compares seven sensor-placement approaches:

- QDEIM
- D-optimal design
- A-optimal design
- uncertainty-based placement (U-opt)
- greedy RMSE selection
- hybrid D-optimal/RMSE selection
- random placement

The analysis includes candidate-site screening, EOF construction, sparse reconstruction, observation uncertainty, reconstruction metrics, computational timing, and CSV/Shapefile exports.

## Repository contents

```text
.
├── eof_sensor_placement.py
├── data/
│   ├── topo.nc
│   └── 3D_mean_2016-03.nc
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

The NetCDF files are tracked with Git LFS because model output can exceed the normal GitHub file-size limit.

## Input data

By default, the script reads:

- `data/topo.nc` — model grid/topography file
- `data/3D_mean_2016-03.nc` — three-dimensional hydrodynamic model output

The topography file must contain `lonc` and `latc`.

The three-dimensional model file must contain `time`, `bathymetry`, `temp`, `salt`, and `hn`. The variable `hv` is optional and is used only when a positive `HV_MIN` threshold is configured.

Alternative files can be supplied with `--topo` and `--data`, so the code does not depend on a particular server or directory layout.

## Installation

The calculations reported in the manuscript were run with Python 3.9.18, NumPy 1.26.4, SciPy 1.13.1, and xarray 2024.7.0.

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```text
.venv\Scripts\activate
```

## Running the analysis

Run one placement method using the default March data:

```bash
python eof_sensor_placement.py dopt --candidates voxel
```

Run all placement methods:

```bash
python eof_sensor_placement.py --all_methods --candidates voxel
```

Use data stored elsewhere:

```bash
python eof_sensor_placement.py dopt \
  --topo /path/to/topo.nc \
  --data /path/to/model_output.nc \
  --start 2016-03-01 \
  --end 2016-03-05 \
  --output /path/to/results
```

The default output directory is `outputs/` in the repository root.

## Main settings

The principal analysis settings are grouped near the top of `eof_sensor_placement.py`. They include the number of retained EOF modes, maximum number of sensors, voxel dimensions, observation uncertainties, ridge regularization, and temperature/salinity weighting.

For the manuscript analysis, the default candidate spacing is 500 m × 500 m × 1 m and the maximum sensor count is 21.

## Output

Results are written by placement method and candidate mode. Depending on the selected settings, outputs include:

- selected sensor locations
- reconstruction metrics by sensor count
- computational timings
- per-snapshot and per-location errors
- uncertainty estimates
- EOF and reconstruction diagnostics
- CSV files for figures and further analysis
- Shapefiles for spatial post-processing

Generated results are excluded from version control.

## Data source

The hydrodynamic fields used in the case study were generated with the General Estuarine Transport Model (GETM). Details of the Weser Estuary model configuration are given in the manuscript and the cited model publications.

## Citation

If you use this code, please cite the accompanying manuscript:

Morandage, S., Rummel, K., and Prien, R. *Model-Informed EOF-Based Sensor Placement for Marine Monitoring*.


## License

This repository is released under the MIT License. Users should retain the copyright and license notices when redistributing the code.
