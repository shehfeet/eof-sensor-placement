# EOF-based sensor placement for marine monitoring

Python scripts for EOF-based sensor placement and reconstruction of temperature and salinity in the Weser Estuary.

The revised analysis uses days 1–2 for training, day 3 for validation and day 4 for testing.

## Input data

Place the required NetCDF files in a `Data` folder beside the scripts:

```text
Data/
    topo.nc
    3D_mean_2016-03.nc
    3D_mean_2016-06.nc
    3D_mean_2016-09.nc
    3D_mean_2016-12.nc
```

The March file and topography are needed for the main analysis. The adaptive analysis also requires June, September and December.

The model-data record is available on [Zenodo](https://doi.org/10.5281/zenodo.24334506). Model files and generated results are not included in this repository.

## Installation

Install the required packages:

```sh
python -m pip install -r requirements.txt
```

## Running the analysis

Keep all Python scripts in the same folder. Each part can be run separately.

D-opt reference-design selection and measurement-noise evaluation:

```sh
python run_dopt_design.py
```

Comparison of the six optimized placement methods:

```sh
python run_methods.py
```

Random baseline using 100 layouts:

```sh
python run_random.py
```

D-opt parameter sensitivity:

```sh
python run_sensitivity.py
```

Seasonal comparison:

```sh
python run_adaptive.py
```

After the Methods and Random runs finish, the additional noise-weighting check can be run:

```sh
python check_noise_weights.py
```

The shared analysis settings are specified in `settings.py`. Random runs independently using the same EOF count and ridge parameter as the optimized methods.

Results are saved in separate folders under `outputs/`. To resume an interrupted run, repeat the same command without changing the scripts or input files.

## Citation

Morandage, S., Rummel, K., and Prien, R. *Model-Informed EOF-Based Sensor Placement for Marine Monitoring.*
