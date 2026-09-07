#!/usr/bin/env python3
"""
3D sensor selection + EOF reconstruction pipeline (with sensor uncertainty)
+ metric suite per added sensor
+ benchmarking (time + hardware)
+ CSV exports for later plotting

Run examples:
  python eof_op.py
  python eof_op.py qdeim
  python eof_op.py hybrid --candidates voxel
  python eof_op.py --all_methods --candidates voxel

Notes:
- method          : optimization method (qdeim|dopt|aopt|rmse|uopt|hybrid|random)
- candidate_mode  : candidate pool builder (all|voxel)

Sensor uncertainty (3 levels):
  Level 1 (NOISE_ADD_OBS): Treat model output as "truth", add Gaussian observation noise at sensors.
  Level 2 (NOISE_AWARE_SELECTION + NOISE_USE_WLS):
    - Selection uses noise-aware whitening (prefer lower-noise locations).
    - PC fitting uses weighted least squares (WLS) using sensor sigmas.
  Level 3 (UNCERTAINTY_BANDS):
    - Propagate measurement noise to obtain approximate reconstruction uncertainty (space-wise std).
"""

from __future__ import annotations

import os
import time
import argparse
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import xarray as xr

from joblib import Parallel, delayed
from scipy.linalg import qr as scipy_qr


# =============================================================================
# USER SETTINGS (edit only this section)
# =============================================================================

# Data
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DATA_TOPO = DATA_DIR / "topo.nc"
DATA_3D = DATA_DIR / "3D_mean_2016-03.nc"
TIME_START = "2016-03-01"
TIME_END   = "2016-03-05"

# Optimization method: "qdeim" | "dopt" | "aopt" | "rmse" | "uopt" | "hybrid" | "random"
DEFAULT_METHOD = "qdeim"

# Candidate pool builder: "all" | "voxel"
CANDIDATE_MODE = "voxel"

# Core params
RANDOM_SEED = 42
MAX_SENSORS = 21
N_MODES_T = 10
N_MODES_S = 10
RIDGE = 1e-3
WT = 1.0
WS = 1.0

# -------------------------------------------------------------------------
# Combined temperature--salinity objective
# -------------------------------------------------------------------------
# A direct RMSE_T + RMSE_S objective mixes degrees C and
# g/kg.  The default objective below uses a density-anomaly proxy based on
# the linearized equation of state:
#     rho' / rho0 ~= beta*S - alpha*T
# The selected score is normalized by the anomaly standard deviation of this
# density proxy over the full reference domain and period.
#
# Options:
#   "density_proxy_nrmse"     : RMSE(beta*dS - alpha*dT) / std(beta*S' - alpha*T')
#   "density_component_nrmse" : (alpha*RMSE_T + beta*RMSE_S) /
#                               (alpha*std(T') + beta*std(S'))
#   "weighted_nrmse"          : weighted average of RMSE_T/std(T') and RMSE_S/std(S')
#   "raw_sum"                 : legacy objective, not dimensionally consistent
COMBINED_OBJECTIVE = "density_proxy_nrmse"
ALPHA_THERMAL = 1.7e-4   # thermal expansion coefficient [degC^-1]
BETA_HALINE = 7.6e-4     # haline contraction coefficient [kg/g if S is g/kg]
RHO0 = 1025.0            # reference density [kg m^-3], used only for reporting
USE_DENSITY_WEIGHTS_FOR_INFO_METHODS = True

# Wetness threshold for hn [m]
HN_MIN = 0.01

# Optional 2D gate
HV_VAR = "hv"
HV_MIN = 0.0

# Candidate pool parameters
N_OPT_POINTS = -1          # used for "all" mode (<=0 keeps all)
MAX_CANDIDATES = None      # cap after candidate creation

# Filtering strictness across time:
REQUIRE_WET_ALL_TIME = True
REQUIRE_FINITE_ALL_TIME = True

# Hybrid method settings
PRESELECT_FRAC = 0.10
PRESELECT_MIN  = 50
PRESELECT_MAX  = None
N_JOBS = -1

# Voxel sizes (meters) for "voxel" candidate_mode
VOXEL_DX_M = 500.0
VOXEL_DY_M = 500.0
VOXEL_DZ_M = 1.0

# Outputs
# Note: kept as OUT_ROOT for backward compatibility, but this script no longer generates plots.
OUT_ROOT = REPO_ROOT / "outputs"
SAVE_SENSORS_TXT = True
SAVE_METRICS_TXT = True

# CSV exports
EXPORT_CSV = True

# Shapefile exports (AUTO behavior happens in main()).
EXPORT_CANDIDATES_SHP: Optional[bool] = None  # None => AUTO
EXPORT_SELECTED_SHP: Optional[bool] = None    # None => AUTO

CANDIDATES_SHP_NAME = "candidates_3D.shp"
SELECTED_SHP_NAME = "selected_sensors_3D.shp"
CANDIDATES_CRS = "EPSG:4326"  # lon/lat

# -------------------------------------------------------------------------
# Sensor uncertainty controls (OSSE)
# -------------------------------------------------------------------------
NOISE_ADD_OBS = True
NOISE_AWARE_SELECTION = True
NOISE_USE_WLS = True
UNCERTAINTY_BANDS = True

SIGMA_MODEL = "constant"  # "constant" | "depth_linear"
SIGMA_LEVEL = "low"      # "low" | "medium" | "high" | "custom"

SIGMA_LEVELS_TEMP = {"low": 0.01, "medium": 0.02, "high": 0.05}
SIGMA_LEVELS_SALT = {"low": 0.005, "medium": 0.01, "high": 0.02}

SIGMA_TEMP_CUSTOM = 0.02
SIGMA_SALT_CUSTOM = 0.01

DEPTH_SLOPE_TEMP = 0.00
DEPTH_SLOPE_SALT = 0.00

NOISE_SEED = 1234
UNCERTAINTY_FACTOR = 2.0


# --- Additional uncertainty components ---
# Model/truncation error: accounts for unresolved EOF modes / representation error.
MODEL_ERROR_INCLUDE = True
MODEL_ERROR_FACTOR_TEMP = 1.0   # multiplies truncation residual std (Temp)
MODEL_ERROR_FACTOR_SALT = 1.0   # multiplies truncation residual std (Salt)
MODEL_ERROR_FLOOR_TEMP = 0.0    # absolute floor added in quadrature [Temp units]
MODEL_ERROR_FLOOR_SALT = 0.0    # absolute floor added in quadrature [Salt units]

# --- Optional bias / drift in sensor observations (in addition to Gaussian noise) ---
# Bias: constant offset per sensor across time.
NOISE_ADD_BIAS = False
BIAS_STD_FACTOR_TEMP = 0.5   # bias std = factor * sigma_sel (Temp)
BIAS_STD_FACTOR_SALT = 0.5   # bias std = factor * sigma_sel (Salt)

# Drift: slow time-varying offset per sensor (random walk).
NOISE_ADD_DRIFT = False
DRIFT_STEP_FACTOR_TEMP = 0.05  # drift step std = factor * sigma_sel per time step (Temp)
DRIFT_STEP_FACTOR_SALT = 0.05  # drift step std = factor * sigma_sel per time step (Salt)

# -------------------------------------------------------------------------
# Post-processing exports (NO PLOTS)
# -------------------------------------------------------------------------
# All quantities that used to be shown as plots are exported as CSV (x,y) and/or
# Shapefiles are exported for spatial post-processing.
EXPORT_POSTPROC = True

# Time-series export selection
TS_FIXED_SENSOR_NUMBERS = [1, 5, 10, 15]  # 1-based sensor numbers from the selected list
TS_RANDOM_N = 5
TS_RANDOM_SEED = 20260128


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="3D sensor selection + EOF reconstruction + metrics/benchmarks")
    p.add_argument(
        "method", nargs="?", default=DEFAULT_METHOD,
        choices=["qdeim", "dopt", "aopt", "rmse", "uopt", "hybrid", "random"],
        help="Optimization method"
    )
    p.add_argument(
        "--all_methods", action="store_true",
        help="Run all methods and export comparison CSVs"
    )
    p.add_argument(
        "--candidates", dest="candidate_mode", default=CANDIDATE_MODE,
        choices=["all", "voxel"],
        help="Candidate pool mode"
    )
    p.add_argument(
        "--topo", default=str(DATA_TOPO),
        help="Path to the model grid/topography NetCDF file"
    )
    p.add_argument(
        "--data", default=str(DATA_3D),
        help="Path to the 3D model-output NetCDF file"
    )
    p.add_argument(
        "--output", default=str(OUT_ROOT),
        help="Directory for exported results"
    )
    p.add_argument(
        "--start", default=TIME_START,
        help="Start date/time passed to xarray selection (default: %(default)s)"
    )
    p.add_argument(
        "--end", default=TIME_END,
        help="End date/time passed to xarray selection (default: %(default)s)"
    )
    return p.parse_args()


# =============================================================================
# DATA IMPORT + FILTERING + CANDIDATES
# =============================================================================

def load_and_prepare(
    data_topo: str,
    data_3d: str,
    time_start: str,
    time_end: str,
    require_wet_all_time: bool = True,
    require_finite_all_time: bool = True,
    hn_min: float = 0.0,
    hv_min: float = 0.0,
    hv_var: str = "hv",
) -> Dict[str, Any]:
    topo = xr.open_dataset(data_topo)
    ds = xr.open_dataset(data_3d).sel(time=slice(time_start, time_end))

    lon2d = topo["lonc"].values
    lat2d = topo["latc"].values
    time_arr = ds["time"].values

    bath = ds["bathymetry"].values
    temp = ds["temp"].values
    salt = ds["salt"].values
    hn = ds["hn"].values

    nt, nz, ny, nx = temp.shape
    bath2d = bath[0] if bath.ndim == 3 else bath

    hv_min = float(hv_min)
    if hv_min > 0.0:
        if hv_var in ds.variables:
            hv_raw = ds[hv_var].values
            hv2d = hv_raw[0] if hv_raw.ndim == 3 else hv_raw
        else:
            hv2d = bath2d
        hv_ok_2d = np.isfinite(hv2d) & (hv2d >= hv_min)
    else:
        hv_ok_2d = np.isfinite(bath2d)

    # z(t,k,y,x)
    z = np.cumsum(hn, axis=1) + 0.5 * hn - bath2d[None, None, :, :]
    z0 = z[0]

    # Polygon / wedge mask defining the study region
    lon1, lat1 = 8.32943858828, 53.6033896612
    lon2, lat2 = 8.49082102239, 53.673277974
    dx1, dy1 = lon2 - lon1, lat2 - lat1
    cross1_2d = dx1 * (lat2d - lat1) - dy1 * (lon2d - lon1)

    lon3, lat3 = 8.33014025104, 53.610466959
    lon4p, lat4p = 8.33785854137, 53.3762893856
    dx2, dy2 = lon3 - lon4p, lat3 - lat4p
    cross2_2d = dx2 * (lat2d - lat3) - dy2 * (lon2d - lon3)

    ymin2 = min(lat3, lat4p)
    mask2_constraint = np.ones_like(lat2d, dtype=bool)
    apply_zone = lat2d >= ymin2
    mask2_constraint[apply_zone] = (cross2_2d[apply_zone] <= 0)
    mask2d = (cross1_2d <= 0) & mask2_constraint

    mask2d = mask2d & hv_ok_2d
    mask3d = np.broadcast_to(mask2d[None, :, :], (nz, ny, nx))

    if require_finite_all_time:
        finite3d = np.isfinite(temp).all(axis=0) & np.isfinite(salt).all(axis=0)
    else:
        finite3d = np.isfinite(temp).any(axis=0) & np.isfinite(salt).any(axis=0)

    hn_min = float(hn_min)
    if require_wet_all_time:
        wet3d = np.isfinite(hn).all(axis=0) & (hn > hn_min).all(axis=0)
    else:
        wet3d = np.isfinite(hn).any(axis=0) & (hn > hn_min).any(axis=0)

    valid3d = mask3d & finite3d & wet3d

    temp_snap = temp[:, valid3d]
    salt_snap = salt[:, valid3d]

    lon3d = np.broadcast_to(lon2d[None, :, :], (nz, ny, nx))
    lat3d = np.broadcast_to(lat2d[None, :, :], (nz, ny, nx))

    lon_sp = lon3d[valid3d]
    lat_sp = lat3d[valid3d]
    z_sp = z0[valid3d]
    z_rep = np.nanmedian(z[:, valid3d], axis=0)

    k_sp, iy_sp, ix_sp = np.where(valid3d)
    valid2d_for_map = mask2d & finite3d.any(axis=0)

    return {
        "time": time_arr,
        "temp_snap": temp_snap,
        "salt_snap": salt_snap,
        "lon_sp": lon_sp,
        "lat_sp": lat_sp,
        "z_sp": z_sp,
        "z_rep": z_rep,
        "lon2d": lon2d,
        "lat2d": lat2d,
        "mask2d": mask2d,
        "valid2d_for_map": valid2d_for_map,
        "n_snap": temp_snap.shape[0],
        "n_space": temp_snap.shape[1],
        "ix_sp": ix_sp.astype(np.int64),
        "iy_sp": iy_sp.astype(np.int64),
        "k_sp":  k_sp.astype(np.int64),
    }


def _lonlat_to_xy_m(lon_deg, lat_deg):
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)

    good = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(good):
        return np.full_like(lon, np.nan), np.full_like(lat, np.nan)

    try:
        from pyproj import Transformer
        lon0 = float(np.nanmedian(lon[good]))
        lat0 = float(np.nanmedian(lat[good]))
        zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
        epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except Exception:
        lon0 = float(np.nanmedian(lon[good]))
        lat0 = float(np.nanmedian(lat[good]))
        R = 6371000.0
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        lon0r = np.deg2rad(lon0)
        lat0r = np.deg2rad(lat0)
        x_m = (lon_rad - lon0r) * np.cos(lat0r) * R
        y_m = (lat_rad - lat0r) * R
        return x_m, y_m


def make_candidates(
    data: Dict[str, Any],
    seed: int,
    candidate_mode: str,
    n_opt_points: int,
    voxel_dx_m: float,
    voxel_dy_m: float,
    voxel_dz_m: float,
    max_candidates: Optional[int]
):
    rng = np.random.RandomState(int(seed))
    n_space = int(data["n_space"])

    if candidate_mode == "all":
        n_opt_points = int(n_opt_points)
        if n_opt_points <= 0 or n_opt_points >= n_space:
            idx = np.arange(n_space, dtype=int)
        else:
            idx = np.sort(rng.choice(n_space, size=n_opt_points, replace=False)).astype(int)

        if max_candidates is not None and idx.size > int(max_candidates):
            idx = np.sort(rng.choice(idx, size=int(max_candidates), replace=False)).astype(int)
        return idx

    if candidate_mode != "voxel":
        raise ValueError(f"Unknown candidate_mode={candidate_mode!r}. Use 'all' or 'voxel'.")

    lon = np.asarray(data["lon_sp"], dtype=float)
    lat = np.asarray(data["lat_sp"], dtype=float)
    z   = np.asarray(data["z_rep"], dtype=float)
    x_m, y_m = _lonlat_to_xy_m(lon, lat)

    dx = float(voxel_dx_m); dy = float(voxel_dy_m); dz = float(voxel_dz_m)
    x0 = float(np.nanmin(x_m)); y0 = float(np.nanmin(y_m)); z0 = float(np.nanmin(z))

    vix = np.floor((x_m - x0) / dx).astype(np.int64)
    viy = np.floor((y_m - y0) / dy).astype(np.int64)
    viz = np.floor((z   - z0) / dz).astype(np.int64)

    cx = x0 + (vix + 0.5) * dx
    cy = y0 + (viy + 0.5) * dy
    cz = z0 + (viz + 0.5) * dz
    d2 = (x_m - cx)**2 + (y_m - cy)**2 + (z - cz)**2

    order = np.lexsort((d2, viz, viy, vix))

    vox = np.empty(order.size, dtype=[("vix", np.int64), ("viy", np.int64), ("viz", np.int64)])
    vox["vix"] = vix[order]
    vox["viy"] = viy[order]
    vox["viz"] = viz[order]

    _, first = np.unique(vox, return_index=True)
    rep_idx = np.sort(order[first]).astype(int)

    if max_candidates is not None and rep_idx.size > int(max_candidates):
        rep_idx = np.sort(rng.choice(rep_idx, size=int(max_candidates), replace=False)).astype(int)

    return rep_idx


def map_sensors(selected, lon_sp, lat_sp, z_sp):
    sel = np.asarray(selected, dtype=int)
    return sel, lon_sp[sel], lat_sp[sel], z_sp[sel]


# =============================================================================
# SHAPEFILE EXPORTS (with attributes)
# =============================================================================

def export_points_shp(path: str, data: Dict[str, Any], idx, add_vox=False,
                      voxel_dx=None, voxel_dy=None, voxel_dz=None) -> None:
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        raise RuntimeError("geopandas/shapely not available. Install or disable shapefile export.") from e

    idx = np.asarray(idx, dtype=np.int64)

    lon  = np.asarray(data["lon_sp"][idx], dtype=float)
    lat  = np.asarray(data["lat_sp"][idx], dtype=float)
    zrep = np.asarray(data["z_rep"][idx], dtype=float)

    k  = np.asarray(data["k_sp"][idx], dtype=np.int64)
    ix = np.asarray(data["ix_sp"][idx], dtype=np.int64)
    iy = np.asarray(data["iy_sp"][idx], dtype=np.int64)

    attrs = {"idx": idx, "lon": lon, "lat": lat, "zrep": zrep, "k": k, "ix": ix, "iy": iy}

    if add_vox:
        # Voxel indices use a consistent origin (x0, y0, z0) across exports.
        # across exports; otherwise candidate vs selected shapefiles will not align.
        if voxel_dx is None or voxel_dy is None or voxel_dz is None:
            raise ValueError("add_vox=True requires voxel_dx, voxel_dy, voxel_dz.")

        lon_all = np.asarray(data["lon_sp"], dtype=float)
        lat_all = np.asarray(data["lat_sp"], dtype=float)
        z_all = np.asarray(data["z_rep"], dtype=float)
        x_all, y_all = _lonlat_to_xy_m(lon_all, lat_all)
        x0 = float(np.nanmin(x_all)); y0 = float(np.nanmin(y_all)); z0 = float(np.nanmin(z_all))

        x_m, y_m = _lonlat_to_xy_m(lon, lat)
        attrs["vix"] = np.floor((x_m - x0) / float(voxel_dx)).astype(np.int64)
        attrs["viy"] = np.floor((y_m - y0) / float(voxel_dy)).astype(np.int64)
        attrs["viz"] = np.floor((zrep - z0) / float(voxel_dz)).astype(np.int64)

    gdf = gpd.GeoDataFrame(
        attrs,
        geometry=[Point(float(x), float(y)) for x, y in zip(lon, lat)],
        crs=CANDIDATES_CRS,
    )
    gdf.to_file(path, driver="ESRI Shapefile")


# =============================================================================
# SENSOR NOISE HELPERS
# =============================================================================

def _resolve_sigma_levels() -> Tuple[float, float]:
    if SIGMA_LEVEL == "custom":
        return float(SIGMA_TEMP_CUSTOM), float(SIGMA_SALT_CUSTOM)
    if SIGMA_LEVEL not in SIGMA_LEVELS_TEMP or SIGMA_LEVEL not in SIGMA_LEVELS_SALT:
        raise ValueError(
            f"SIGMA_LEVEL={SIGMA_LEVEL!r} not in allowed keys "
            f"{sorted(SIGMA_LEVELS_TEMP.keys())} (or use 'custom')."
        )
    return float(SIGMA_LEVELS_TEMP[SIGMA_LEVEL]), float(SIGMA_LEVELS_SALT[SIGMA_LEVEL])


def build_sigma_space(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    baseT, baseS = _resolve_sigma_levels()
    z = np.asarray(data["z_rep"], dtype=float)
    if SIGMA_MODEL == "constant":
        sigT = np.full_like(z, baseT, dtype=float)
        sigS = np.full_like(z, baseS, dtype=float)
    elif SIGMA_MODEL == "depth_linear":
        sigT = baseT * (1.0 + float(DEPTH_SLOPE_TEMP) * np.abs(z))
        sigS = baseS * (1.0 + float(DEPTH_SLOPE_SALT) * np.abs(z))
    else:
        raise ValueError(f"Unknown SIGMA_MODEL={SIGMA_MODEL!r}. Use 'constant' or 'depth_linear'.")
    sigT = np.maximum(sigT, 1e-12)
    sigS = np.maximum(sigS, 1e-12)
    return sigT, sigS


def add_sensor_noise(
    y_true: np.ndarray,
    sigma_sel: np.ndarray,
    rng: np.random.RandomState,
    *,
    add_bias: bool = False,
    bias_std_factor: float = 0.0,
    add_drift: bool = False,
    drift_step_factor: float = 0.0,
) -> np.ndarray:
    """Add observation noise to sensor time series.

    y_true: (nt, k)
    sigma_sel: (k,) or scalar
    bias: constant offset per sensor across all times
    drift: random-walk offset per sensor over time
    """
    sigma_sel = np.asarray(sigma_sel, dtype=float)
    if sigma_sel.ndim == 0:
        sigma_sel = np.full((y_true.shape[1],), float(sigma_sel))

    nt, k = y_true.shape
    y = y_true + rng.normal(0.0, sigma_sel[None, :], size=y_true.shape)

    if add_bias and bias_std_factor > 0:
        bstd = bias_std_factor * sigma_sel
        bias = rng.normal(0.0, bstd, size=(k,))
        y = y + bias[None, :]

    if add_drift and drift_step_factor > 0:
        step_std = drift_step_factor * sigma_sel
        steps = rng.normal(0.0, step_std[None, :], size=(nt, k))
        drift = np.cumsum(steps, axis=0)
        y = y + drift

    return y



# =============================================================================
# EOF + RECONSTRUCTION UTILS
# =============================================================================

def compute_eofs_timecov(X: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    n_snap, _n_space = X.shape
    if n_snap < 2:
        raise ValueError("Need at least 2 snapshots to compute EOFs.")

    C = (X @ X.T) / (n_snap - 1)
    evals, U = np.linalg.eigh(C)

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    U = U[:, order]

    r = min(int(n_modes), n_snap)
    U_r = U[:, :r]
    evals_r = np.maximum(evals[:r], 0.0)

    # Drop numerically tiny modes to avoid exploding EOF amplitudes
    s_r = np.sqrt(evals_r * (n_snap - 1))
    if s_r.size > 0:
        tol = 1e-8 * np.max(s_r)
        keep = s_r > tol
        U_r = U_r[:, keep]
        s_r = s_r[keep]

    EOFs = (X.T @ U_r) / (s_r[None, :] + 1e-12)

    return EOFs, s_r

def explained_variance_fraction(s: np.ndarray) -> np.ndarray:
    v = s**2
    denom = float(v.sum())
    if denom <= 0:
        return np.full_like(v, np.nan)
    return v / denom


def fit_pcs_from_sensors_ols(EOFs: np.ndarray, mean_space: np.ndarray,
                            y_obs: np.ndarray, sensor_idx: np.ndarray,
                            ridge: float = 0.0) -> np.ndarray:
    r = EOFs.shape[1]
    A = EOFs[sensor_idx, :]
    b = y_obs - mean_space[sensor_idx]
    AtA = A.T @ A
    if ridge > 0:
        AtA = AtA + float(ridge) * np.eye(r)
    rhs = A.T @ b.T
    PCs_hat = np.linalg.solve(AtA, rhs).T
    return PCs_hat


def fit_pcs_from_sensors_wls(EOFs: np.ndarray, mean_space: np.ndarray,
                            y_obs: np.ndarray, sensor_idx: np.ndarray,
                            sigma_sel: Optional[np.ndarray],
                            ridge: float = 0.0) -> np.ndarray:
    if sigma_sel is None:
        return fit_pcs_from_sensors_ols(EOFs, mean_space, y_obs, sensor_idx, ridge=ridge)

    r = EOFs.shape[1]
    A = EOFs[sensor_idx, :]
    b = y_obs - mean_space[sensor_idx]

    sigma_sel = np.asarray(sigma_sel, dtype=float)
    if sigma_sel.ndim == 0:
        sigma_sel = np.full((A.shape[0],), float(sigma_sel))
    w = 1.0 / np.maximum(sigma_sel**2, 1e-30)

    AtA = A.T @ (A * w[:, None])
    rhs = A.T @ (b.T * w[:, None])

    if ridge > 0:
        AtA = AtA + float(ridge) * np.eye(r)

    PCs_hat = np.linalg.solve(AtA, rhs).T
    return PCs_hat


def reconstruct_from_pcs(EOFs: np.ndarray, mean_space: np.ndarray, PCs: np.ndarray) -> np.ndarray:
    return mean_space[None, :] + PCs @ EOFs.T


def rmse_global(Xhat: np.ndarray, X: np.ndarray) -> float:
    return float(np.sqrt(((Xhat - X) ** 2).mean()))


def rmse_per_snapshot(Xhat: np.ndarray, X: np.ndarray) -> np.ndarray:
    # RMSE at each time snapshot (over space)
    return np.sqrt(((Xhat - X) ** 2).mean(axis=1))


def rmse_per_space(Xhat: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.sqrt(((Xhat - X) ** 2).mean(axis=0))


def anomaly_reference_scale(X: np.ndarray) -> float:
    """One scalar anomaly standard deviation for a reference field.

    The temporal mean is removed at each valid spatial grid point first.
    The returned value is then the RMS anomaly over all times and valid
    spatial locations. This is the recommended reference variability scale
    for EOF-based anomaly reconstruction.
    """
    X = np.asarray(X, dtype=np.float64)
    mean_space = np.nanmean(X, axis=0, keepdims=True)
    X_anom = X - mean_space
    scale = float(np.sqrt(np.nanmean(X_anom ** 2)))
    return max(scale, 1e-12)


def effective_density_coefficients(wt: float = 1.0, ws: float = 1.0) -> Tuple[float, float]:
    """Return effective alpha and beta coefficients including optional priorities.

    WT and WS remain available as scientific priority multipliers. With the
    default WT=WS=1, salinity receives beta/alpha ~= 4.47 times the density
    influence of temperature per unit error.
    """
    alpha_eff = float(wt) * float(ALPHA_THERMAL)
    beta_eff = float(ws) * float(BETA_HALINE)
    return alpha_eff, beta_eff


def density_proxy(temp: np.ndarray, salt: np.ndarray, *, wt: float = 1.0, ws: float = 1.0, rho0: float = 1.0) -> np.ndarray:
    """Linear density-anomaly proxy rho0 * (beta*S - alpha*T).

    Use rho0=1 for a dimensionless proxy and rho0=RHO0 for kg m^-3.
    """
    alpha_eff, beta_eff = effective_density_coefficients(wt=wt, ws=ws)
    return float(rho0) * (beta_eff * np.asarray(salt, dtype=np.float64)
                          - alpha_eff * np.asarray(temp, dtype=np.float64))


def density_proxy_error_rmse(T_hat: np.ndarray, S_hat: np.ndarray, T: np.ndarray, S: np.ndarray,
                             *, wt: float = 1.0, ws: float = 1.0, rho0: float = 1.0) -> float:
    """RMSE of the linearized density-proxy error."""
    e = density_proxy(T_hat, S_hat, wt=wt, ws=ws, rho0=rho0) - density_proxy(T, S, wt=wt, ws=ws, rho0=rho0)
    return float(np.sqrt(np.nanmean(e ** 2)))


def compute_reference_scales(temp_truth: np.ndarray, salt_truth: np.ndarray,
                             *, wt: float = 1.0, ws: float = 1.0) -> Dict[str, float]:
    """Compute all reference standard deviations used for normalization.

    These values are computed once for the full valid 3D reference domain and
    the full analysis period. They are not recomputed for each candidate, each
    sensor count, or each time step.
    """
    alpha_eff, beta_eff = effective_density_coefficients(wt=wt, ws=ws)
    scale_T = anomaly_reference_scale(temp_truth)
    scale_S = anomaly_reference_scale(salt_truth)
    rho_proxy_truth = density_proxy(temp_truth, salt_truth, wt=wt, ws=ws, rho0=1.0)
    scale_rho_proxy = anomaly_reference_scale(rho_proxy_truth)
    scale_rho_kgm3 = float(RHO0) * scale_rho_proxy

    return {
        "alpha": float(ALPHA_THERMAL),
        "beta": float(BETA_HALINE),
        "alpha_eff": float(alpha_eff),
        "beta_eff": float(beta_eff),
        "beta_over_alpha": float(BETA_HALINE / ALPHA_THERMAL),
        "effective_beta_over_alpha": float(beta_eff / max(alpha_eff, 1e-30)),
        "rho0": float(RHO0),
        "std_T_anomaly": float(scale_T),
        "std_S_anomaly": float(scale_S),
        "std_density_proxy_anomaly": float(scale_rho_proxy),
        "std_density_anomaly_kg_m3": float(scale_rho_kgm3),
        "component_density_scale": float(alpha_eff * scale_T + beta_eff * scale_S),
    }


def effective_information_weights(wt: float, ws: float, ref_scales: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    """Weights used by QDEIM/D-opt/A-opt style multi-variable criteria.

    Those methods do not use physical RMSE directly, but their combined
    temperature/salinity objective still needs a defensible variable priority.
    We therefore use the density sensitivity ratio beta/alpha when requested.
    """
    if not bool(USE_DENSITY_WEIGHTS_FOR_INFO_METHODS):
        return float(wt), float(ws)
    return float(wt), float(ws) * float(BETA_HALINE / ALPHA_THERMAL)


def combined_reconstruction_score(
    temp_truth: np.ndarray,
    salt_truth: np.ndarray,
    T_hat: np.ndarray,
    S_hat: np.ndarray,
    rmse_t: float,
    rmse_s: float,
    ref_scales: Dict[str, float],
    *,
    wt: float = 1.0,
    ws: float = 1.0,
    objective: str = COMBINED_OBJECTIVE,
) -> float:
    """Dimensionally consistent combined score for optimization/reporting."""
    objective = str(objective).lower()
    if objective == "density_proxy_nrmse":
        rmse_rho_proxy = density_proxy_error_rmse(T_hat, S_hat, temp_truth, salt_truth, wt=wt, ws=ws, rho0=1.0)
        return float(rmse_rho_proxy / max(ref_scales["std_density_proxy_anomaly"], 1e-12))

    if objective == "density_component_nrmse":
        alpha_eff = float(ref_scales["alpha_eff"])
        beta_eff = float(ref_scales["beta_eff"])
        denom = max(float(ref_scales["component_density_scale"]), 1e-12)
        return float((alpha_eff * float(rmse_t) + beta_eff * float(rmse_s)) / denom)

    if objective == "weighted_nrmse":
        denom = max(float(wt) + float(ws), 1e-12)
        return float((float(wt) * float(rmse_t) / max(ref_scales["std_T_anomaly"], 1e-12)
                      + float(ws) * float(rmse_s) / max(ref_scales["std_S_anomaly"], 1e-12)) / denom)

    if objective == "raw_sum":
        return float(float(wt) * float(rmse_t) + float(ws) * float(rmse_s))

    raise ValueError(f"Unknown COMBINED_OBJECTIVE={objective!r}")


def combined_uncertainty_score(stdT: np.ndarray, stdS: np.ndarray, ref_scales: Dict[str, float],
                               *, wt: float = 1.0, ws: float = 1.0,
                               objective: str = COMBINED_OBJECTIVE) -> float:
    """Dimensionally consistent uncertainty objective for U-opt.

    For the density-proxy objective we assume independent T and S uncertainty
    contributions and combine them in quadrature.
    """
    objective = str(objective).lower()
    stdT = np.asarray(stdT, dtype=float)
    stdS = np.asarray(stdS, dtype=float)

    if objective == "density_proxy_nrmse":
        alpha_eff = float(ref_scales["alpha_eff"])
        beta_eff = float(ref_scales["beta_eff"])
        std_rho = np.sqrt((alpha_eff * stdT) ** 2 + (beta_eff * stdS) ** 2)
        return float(np.nanmean(std_rho) / max(ref_scales["std_density_proxy_anomaly"], 1e-12))

    if objective == "density_component_nrmse":
        alpha_eff = float(ref_scales["alpha_eff"])
        beta_eff = float(ref_scales["beta_eff"])
        denom = max(float(ref_scales["component_density_scale"]), 1e-12)
        return float((alpha_eff * float(np.nanmean(stdT)) + beta_eff * float(np.nanmean(stdS))) / denom)

    if objective == "weighted_nrmse":
        denom = max(float(wt) + float(ws), 1e-12)
        return float((float(wt) * float(np.nanmean(stdT)) / max(ref_scales["std_T_anomaly"], 1e-12)
                      + float(ws) * float(np.nanmean(stdS)) / max(ref_scales["std_S_anomaly"], 1e-12)) / denom)

    if objective == "raw_sum":
        return float(float(wt) * float(np.nanmean(stdT)) + float(ws) * float(np.nanmean(stdS)))

    raise ValueError(f"Unknown COMBINED_OBJECTIVE={objective!r}")


def compute_density_proxy_metrics(temp_truth: np.ndarray, salt_truth: np.ndarray,
                                  T_hat: np.ndarray, S_hat: np.ndarray,
                                  ref_scales: Dict[str, float],
                                  *, wt: float = 1.0, ws: float = 1.0) -> Dict[str, float]:
    """Metrics for the density-anomaly proxy used in the combined objective."""
    rho_truth = density_proxy(temp_truth, salt_truth, wt=wt, ws=ws, rho0=1.0)
    rho_recon = density_proxy(T_hat, S_hat, wt=wt, ws=ws, rho0=1.0)
    m = compute_metrics_global(rho_truth, rho_recon)
    rmse_proxy = float(m["rmse"])
    return {
        "rmse_density_proxy": rmse_proxy,
        "nrmse_density_proxy": float(rmse_proxy / max(ref_scales["std_density_proxy_anomaly"], 1e-12)),
        "rmse_density_kg_m3": float(RHO0 * rmse_proxy),
        "std_density_kg_m3": float(ref_scales["std_density_anomaly_kg_m3"]),
        "corr_r_density_proxy": float(m["corr_r"]),
        "var_ratio_density_proxy": float(m["var_ratio"]),
        "nse_density_proxy": float(m["nse"]),
    }


def print_reference_scales(ref_scales: Dict[str, float], *, method: str, candidate_mode: str) -> None:
    """Print the normalization constants so they are visible in log files."""
    print("\n=== Combined-objective reference scales ===")
    print(f"method={method}, candidates={candidate_mode}")
    print(f"COMBINED_OBJECTIVE: {COMBINED_OBJECTIVE}")
    print(f"alpha: {ref_scales['alpha']:.6g} degC^-1")
    print(f"beta : {ref_scales['beta']:.6g} kg/g")
    print(f"beta/alpha: {ref_scales['beta_over_alpha']:.6g}")
    print(f"effective beta/alpha including WT/WS: {ref_scales['effective_beta_over_alpha']:.6g}")
    print(f"std(T anomaly): {ref_scales['std_T_anomaly']:.8g} degC")
    print(f"std(S anomaly): {ref_scales['std_S_anomaly']:.8g} g kg^-1")
    print(f"std(beta*S' - alpha*T'): {ref_scales['std_density_proxy_anomaly']:.8g}")
    print(f"rho0*std(beta*S' - alpha*T'): {ref_scales['std_density_anomaly_kg_m3']:.8g} kg m^-3")
    print("===========================================\n")


def truncation_residual_std(X_anom: np.ndarray, EOFs: np.ndarray) -> np.ndarray:
    """Per-space residual std due to EOF truncation (numerically stable)."""
    X_anom = np.asarray(X_anom, dtype=np.float64)

    if EOFs is None or np.size(EOFs) == 0:
        resid = X_anom
    else:
        EOFs = np.asarray(EOFs, dtype=np.float64)
        PCs = X_anom @ EOFs
        X_rec = PCs @ EOFs.T
        resid = X_anom - X_rec

    scale = np.nanmax(np.abs(resid), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    rms = np.sqrt(np.nanmean((resid / scale) ** 2, axis=0)) * scale
    return np.asarray(rms, dtype=float)

def build_eofs(temp_snap: np.ndarray, salt_snap: np.ndarray, n_modes_t: int, n_modes_s: int):
    temp_mean = temp_snap.mean(axis=0)
    salt_mean = salt_snap.mean(axis=0)

    temp_anom = temp_snap - temp_mean[None, :]
    salt_anom = salt_snap - salt_mean[None, :]

    EOF_T, sT = compute_eofs_timecov(temp_anom, n_modes_t)
    EOF_S, sS = compute_eofs_timecov(salt_anom, n_modes_s)

    evT = explained_variance_fraction(sT)
    evS = explained_variance_fraction(sS)

    # Truncation residual std (per space): useful as a model/representation error term.
    resT_std = truncation_residual_std(temp_anom, EOF_T)
    resS_std = truncation_residual_std(salt_anom, EOF_S)

    return temp_mean, salt_mean, EOF_T, EOF_S, evT, evS, resT_std, resS_std


def pc_covariance_wls(EOFs: np.ndarray, sensor_idx: np.ndarray, sigma_sel: Optional[np.ndarray], ridge: float) -> np.ndarray:
    """Posterior covariance of PCs for WLS with ridge regularization (numerically stable)."""
    A = np.asarray(EOFs, dtype=float)[np.asarray(sensor_idx, dtype=int), :]
    r = A.shape[1]

    if sigma_sel is None:
        AtA = A.T @ A
    else:
        sigma = np.asarray(sigma_sel, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full((A.shape[0],), float(sigma))
        finite = np.isfinite(sigma)
        if not np.all(finite):
            med = np.nanmedian(sigma[finite]) if np.any(finite) else 1.0
            sigma = np.where(finite, sigma, med)
        sigma = np.maximum(sigma, 1e-6)
        w = 1.0 / (sigma ** 2)
        AtA = A.T @ (w[:, None] * A)

    ridge = float(max(ridge, 1e-8))
    M = AtA + ridge * np.eye(r)

    try:
        L = np.linalg.cholesky(M)
        Linv = np.linalg.solve(L, np.eye(r))
        Cov_p = Linv.T @ Linv
    except np.linalg.LinAlgError:
        Cov_p = np.linalg.pinv(M, rcond=1e-10)

    return Cov_p

def field_std_from_pc_cov(EOFs: np.ndarray, Cov_p: np.ndarray) -> np.ndarray:
    V = EOFs @ Cov_p
    var = np.sum(V * EOFs, axis=1)
    return np.sqrt(np.maximum(var, 0.0))


def reconstruct_and_score(
    temp_truth: np.ndarray,
    salt_truth: np.ndarray,
    EOF_T: np.ndarray,
    temp_mean: np.ndarray,
    EOF_S: np.ndarray,
    salt_mean: np.ndarray,
    selected: np.ndarray,
    ridge: float,
    sigmaT_space: Optional[np.ndarray] = None,
    sigmaS_space: Optional[np.ndarray] = None,
    resT_std: Optional[np.ndarray] = None,
    resS_std: Optional[np.ndarray] = None,
    add_obs_noise: bool = False,
    use_wls: bool = False,
    noise_seed: int = 0,
):
    """Reconstruct fields from selected sensors and compute RMSE + uncertainty bands."""
    sel = np.asarray(selected, dtype=int)
    rng = np.random.RandomState(int(noise_seed))

    yT_true = temp_truth[:, sel]
    yS_true = salt_truth[:, sel]

    sigT_sel = None if sigmaT_space is None else np.asarray(sigmaT_space[sel], dtype=float)
    sigS_sel = None if sigmaS_space is None else np.asarray(sigmaS_space[sel], dtype=float)

    if add_obs_noise:
        yT_obs = add_sensor_noise(
            yT_true,
            sigT_sel if sigT_sel is not None else 0.0,
            rng,
            add_bias=bool(NOISE_ADD_BIAS),
            bias_std_factor=float(BIAS_STD_FACTOR_TEMP),
            add_drift=bool(NOISE_ADD_DRIFT),
            drift_step_factor=float(DRIFT_STEP_FACTOR_TEMP),
        )
        yS_obs = add_sensor_noise(
            yS_true,
            sigS_sel if sigS_sel is not None else 0.0,
            rng,
            add_bias=bool(NOISE_ADD_BIAS),
            bias_std_factor=float(BIAS_STD_FACTOR_SALT),
            add_drift=bool(NOISE_ADD_DRIFT),
            drift_step_factor=float(DRIFT_STEP_FACTOR_SALT),
        )
    else:
        yT_obs = yT_true
        yS_obs = yS_true

    if use_wls:
        PCsT_hat = fit_pcs_from_sensors_wls(EOF_T, temp_mean, yT_obs, sel, sigma_sel=sigT_sel, ridge=ridge)
        PCsS_hat = fit_pcs_from_sensors_wls(EOF_S, salt_mean, yS_obs, sel, sigma_sel=sigS_sel, ridge=ridge)
    else:
        PCsT_hat = fit_pcs_from_sensors_ols(EOF_T, temp_mean, yT_obs, sel, ridge=ridge)
        PCsS_hat = fit_pcs_from_sensors_ols(EOF_S, salt_mean, yS_obs, sel, ridge=ridge)

    T_hat = reconstruct_from_pcs(EOF_T, temp_mean, PCsT_hat)
    S_hat = reconstruct_from_pcs(EOF_S, salt_mean, PCsS_hat)

    rmse_t = rmse_global(T_hat, temp_truth)
    rmse_s = rmse_global(S_hat, salt_truth)

    stdT_space = None
    stdS_space = None
    if UNCERTAINTY_BANDS and use_wls and (sigmaT_space is not None) and (sigmaS_space is not None):
        Cov_p_T = pc_covariance_wls(EOF_T, sel, sigT_sel, ridge=ridge)
        Cov_p_S = pc_covariance_wls(EOF_S, sel, sigS_sel, ridge=ridge)
        stdT_space = field_std_from_pc_cov(EOF_T, Cov_p_T)
        stdS_space = field_std_from_pc_cov(EOF_S, Cov_p_S)

    if MODEL_ERROR_INCLUDE and (resT_std is not None) and (resS_std is not None):
        resT = np.asarray(resT_std, dtype=float)
        resS = np.asarray(resS_std, dtype=float)

        if stdT_space is None:
            stdT_space = np.zeros_like(resT, dtype=float)
        if stdS_space is None:
            stdS_space = np.zeros_like(resS, dtype=float)

        stdT_space = np.sqrt(
            np.maximum(stdT_space, 0.0) ** 2
            + (float(MODEL_ERROR_FACTOR_TEMP) * np.maximum(resT, 0.0)) ** 2
            + float(MODEL_ERROR_FLOOR_TEMP) ** 2
        )
        stdS_space = np.sqrt(
            np.maximum(stdS_space, 0.0) ** 2
            + (float(MODEL_ERROR_FACTOR_SALT) * np.maximum(resS, 0.0)) ** 2
            + float(MODEL_ERROR_FLOOR_SALT) ** 2
        )

    return rmse_t, rmse_s, T_hat, S_hat, yT_obs, yS_obs, stdT_space, stdS_space

def _safe_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.var(x, ddof=0))


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.std(x, ddof=0))


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    aa = a[m]; bb = b[m]
    sa = np.std(aa); sb = np.std(bb)
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def compute_metrics_global(truth: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    """
    Compute standard metrics comparing truth vs recon.
    Inputs are (nt, nspace).
    """
    truth = np.asarray(truth, dtype=float)
    recon = np.asarray(recon, dtype=float)
    err = recon - truth

    mse = float(np.nanmean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.nanmean(np.abs(err)))
    bias = float(np.nanmean(err))

    var_t = _safe_var(truth)
    var_r = _safe_var(recon)
    var_e = _safe_var(err)

    std_t = _safe_std(truth)
    std_r = _safe_std(recon)

    # explained variance captured by recon relative to truth variability:
    # EV = 1 - var(error)/var(truth)
    ev_capture = float("nan")
    if np.isfinite(var_t) and var_t > 0 and np.isfinite(var_e):
        ev_capture = float(1.0 - (var_e / var_t))

    var_ratio = float("nan")
    if np.isfinite(var_t) and var_t > 0 and np.isfinite(var_r):
        var_ratio = float(var_r / var_t)

    nrmse = float("nan")
    if np.isfinite(std_t) and std_t > 0:
        nrmse = float(rmse / std_t)

    r = _safe_corr(truth, recon)
    r2 = float("nan") if not np.isfinite(r) else float(r**2)

    # NSE (Nash–Sutcliffe efficiency)
    # NSE = 1 - sum((recon-truth)^2)/sum((truth-mean(truth))^2)
    nse = float("nan")
    denom = float(np.nanmean((truth - np.nanmean(truth))**2))
    if np.isfinite(denom) and denom > 0:
        nse = float(1.0 - (mse / denom))

    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "bias": bias,
        "std_truth": std_t,
        "std_recon": std_r,
        "var_truth": var_t,
        "var_recon": var_r,
        "var_ratio": var_ratio,
        "ev_capture": ev_capture,
        "corr_r": r,
        "r2": r2,
        "nse": nse,
    }


def collect_hardware_info() -> Dict[str, str]:
    info: Dict[str, str] = {}
    info["platform"] = platform.platform()
    info["python"] = platform.python_version()
    info["processor"] = platform.processor() or "unknown"
    info["machine"] = platform.machine()
    info["system"] = platform.system()
    info["release"] = platform.release()

    try:
        import psutil
        info["cpu_physical_cores"] = str(psutil.cpu_count(logical=False))
        info["cpu_logical_cores"] = str(psutil.cpu_count(logical=True))
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = f"{vm.total/1e9:.2f}"
    except Exception:
        info["cpu_physical_cores"] = "unknown"
        info["cpu_logical_cores"] = "unknown"
        info["ram_total_gb"] = "unknown"

    try:
        import numpy as _np
        info["numpy"] = _np.__version__
    except Exception:
        info["numpy"] = "unknown"

    try:
        import scipy as _sp
        info["scipy"] = _sp.__version__
    except Exception:
        info["scipy"] = "unknown"

    try:
        import xarray as _xr
        info["xarray"] = _xr.__version__
    except Exception:
        info["xarray"] = "unknown"

    return info


def _write_csv_rows(path: str, header: List[str], rows: List[List[Any]]) -> None:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# =============================================================================
# OPTIMIZATION METHODS (noise-aware option)
# =============================================================================

def select_sensors_qdeim(cand_idx, k_max, EOF_T, EOF_S, wt=1.0, ws=1.0,
                        sigmaT_space: Optional[np.ndarray] = None,
                        sigmaS_space: Optional[np.ndarray] = None,
                        noise_aware: bool = False):
    cand_idx = np.asarray(cand_idx, dtype=int)
    PhiT = EOF_T[cand_idx, :]
    PhiS = EOF_S[cand_idx, :]

    if noise_aware and (sigmaT_space is not None) and (sigmaS_space is not None):
        sigT = np.asarray(sigmaT_space[cand_idx], dtype=float)
        sigS = np.asarray(sigmaS_space[cand_idx], dtype=float)
        PhiT = PhiT / sigT[:, None]
        PhiS = PhiS / sigS[:, None]

    fracT = float(wt) / max(float(wt) + float(ws), 1e-12)
    kT = max(1, int(round(int(k_max) * fracT)))
    kS = max(0, int(k_max) - kT)

    _Q, _R, pivT = scipy_qr(PhiT.T, mode="economic", pivoting=True)
    idxT_local = np.array(pivT[:min(kT, pivT.size)], dtype=int)

    if kS > 0:
        _Q, _R, pivS = scipy_qr(PhiS.T, mode="economic", pivoting=True)
        idxS_local = np.array(pivS[:min(kS, pivS.size)], dtype=int)
    else:
        idxS_local = np.array([], dtype=int)

    chosen = []
    chosen_set = set()

    for j in cand_idx[idxT_local]:
        jj = int(j)
        if len(chosen) < int(k_max) and jj not in chosen_set:
            chosen.append(jj); chosen_set.add(jj)

    for j in cand_idx[idxS_local]:
        jj = int(j)
        if len(chosen) >= int(k_max):
            break
        if jj not in chosen_set:
            chosen.append(jj); chosen_set.add(jj)

    if len(chosen) < int(k_max):
        for j in cand_idx:
            jj = int(j)
            if len(chosen) >= int(k_max):
                break
            if jj not in chosen_set:
                chosen.append(jj); chosen_set.add(jj)

    return np.array(chosen[:int(k_max)], dtype=int)


def select_sensors_dopt(cand_idx, k_max, EOF_T, EOF_S, wt=1.0, ws=1.0, ridge=1e-6,
                        sigmaT_space: Optional[np.ndarray] = None,
                        sigmaS_space: Optional[np.ndarray] = None,
                        noise_aware: bool = False):
    rT = EOF_T.shape[1]
    rS = EOF_S.shape[1]
    I_T = np.eye(rT, dtype=float)
    I_S = np.eye(rS, dtype=float)
    ridge = float(ridge)

    selected = []
    selected_set = set()
    AT = np.zeros((rT, rT), dtype=float)
    AS = np.zeros((rS, rS), dtype=float)

    cand_idx = np.asarray(cand_idx, dtype=int)

    for _ in range(int(k_max)):
        best_c = None
        best_score = -np.inf

        for c in cand_idx:
            c = int(c)
            if c in selected_set:
                continue

            vT = EOF_T[c, :].reshape(-1, 1)
            vS = EOF_S[c, :].reshape(-1, 1)

            if noise_aware and (sigmaT_space is not None) and (sigmaS_space is not None):
                vT = vT / float(sigmaT_space[c])
                vS = vS / float(sigmaS_space[c])

            ATc = AT + (vT @ vT.T)
            ASc = AS + (vS @ vS.T)

            sT = np.linalg.slogdet(ATc + ridge * I_T)[1]
            sS = np.linalg.slogdet(ASc + ridge * I_S)[1]
            s = float(wt) * float(sT) + float(ws) * float(sS)

            if s > best_score:
                best_score = s
                best_c = c

        if best_c is None:
            raise RuntimeError("D-opt: failed to select next sensor (no candidate left).")

        vT = EOF_T[best_c, :].reshape(-1, 1)
        vS = EOF_S[best_c, :].reshape(-1, 1)
        if noise_aware and (sigmaT_space is not None) and (sigmaS_space is not None):
            vT = vT / float(sigmaT_space[best_c])
            vS = vS / float(sigmaS_space[best_c])

        AT = AT + (vT @ vT.T)
        AS = AS + (vS @ vS.T)
        selected.append(best_c)
        selected_set.add(best_c)

    return np.array(selected, dtype=int)


def select_sensors_aopt(cand_idx, k_max, EOF_T, EOF_S, wt=1.0, ws=1.0, ridge=1e-6,
                        sigmaT_space: Optional[np.ndarray] = None,
                        sigmaS_space: Optional[np.ndarray] = None,
                        noise_aware: bool = False):
    rT = EOF_T.shape[1]
    rS = EOF_S.shape[1]
    selected = []
    selected_set = set()
    AT = float(ridge) * np.eye(rT)
    AS = float(ridge) * np.eye(rS)
    cand_idx = np.asarray(cand_idx, dtype=int)

    for _ in range(int(k_max)):
        AT_inv = np.linalg.inv(AT)
        AS_inv = np.linalg.inv(AS)
        tr_AT_inv = float(np.trace(AT_inv))
        tr_AS_inv = float(np.trace(AS_inv))

        best_c = None
        best_score = np.inf

        for c in cand_idx:
            c = int(c)
            if c in selected_set:
                continue

            vT = EOF_T[c, :].reshape(-1, 1)
            vS = EOF_S[c, :].reshape(-1, 1)
            if noise_aware and (sigmaT_space is not None) and (sigmaS_space is not None):
                vT = vT / float(sigmaT_space[c])
                vS = vS / float(sigmaS_space[c])

            denomT = 1.0 + float((vT.T @ AT_inv @ vT).item())
            denomS = 1.0 + float((vS.T @ AS_inv @ vS).item())

            vT_Ainv2_vT = float((vT.T @ (AT_inv @ AT_inv) @ vT).item())
            vS_Ainv2_vS = float((vS.T @ (AS_inv @ AS_inv) @ vS).item())

            ATc_inv_trace = tr_AT_inv - (vT_Ainv2_vT / denomT)
            ASc_inv_trace = tr_AS_inv - (vS_Ainv2_vS / denomS)

            s = float(wt) * ATc_inv_trace + float(ws) * ASc_inv_trace
            if s < best_score:
                best_score = s
                best_c = c

        if best_c is None:
            raise RuntimeError("A-opt: failed to select next sensor (no candidate left).")

        vT = EOF_T[best_c, :].reshape(-1, 1)
        vS = EOF_S[best_c, :].reshape(-1, 1)
        if noise_aware and (sigmaT_space is not None) and (sigmaS_space is not None):
            vT = vT / float(sigmaT_space[best_c])
            vS = vS / float(sigmaS_space[best_c])

        AT = AT + (vT @ vT.T)
        AS = AS + (vS @ vS.T)

        selected.append(best_c)
        selected_set.add(best_c)

    return np.array(selected, dtype=int)


def _eval_one(c, selected_list,
              temp_truth, salt_truth,
              EOF_T, temp_mean,
              EOF_S, salt_mean,
              ridge,
              sigmaT_space, sigmaS_space,
              resT_std, resS_std,
              add_obs_noise, use_wls,
              noise_seed,
              ref_scales, wt, ws, combined_objective):
    sel = np.array(selected_list + [int(c)], dtype=int)

    seed_raw = int(noise_seed) + 1000003 * int(sel.size) + 9176 * int(c)
    seed = int(seed_raw % (2**32))

    rmse_t, rmse_s, _T_hat, _S_hat, _yT_obs, _yS_obs, _stdT, _stdS = reconstruct_and_score(
        temp_truth, salt_truth,
        EOF_T, temp_mean,
        EOF_S, salt_mean,
        sel, ridge,
        sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
        resT_std=resT_std, resS_std=resS_std,
        add_obs_noise=add_obs_noise,
        use_wls=use_wls,
        noise_seed=seed
    )
    score = combined_reconstruction_score(
        temp_truth, salt_truth, _T_hat, _S_hat, rmse_t, rmse_s, ref_scales,
        wt=wt, ws=ws, objective=combined_objective
    )
    return score, int(c), rmse_t, rmse_s


def select_sensors_greedy_rmse(cand_idx, k_max,
                              temp_truth, salt_truth,
                              EOF_T, temp_mean,
                              EOF_S, salt_mean,
                              ridge=1e-6,
                              n_jobs=-1,
                              sigmaT_space: Optional[np.ndarray] = None,
                              sigmaS_space: Optional[np.ndarray] = None,
                              resT_std: Optional[np.ndarray] = None,
                              resS_std: Optional[np.ndarray] = None,
                              add_obs_noise: bool = False,
                              use_wls: bool = False,
                              noise_seed: int = 0,
                              ref_scales: Optional[Dict[str, float]] = None,
                              wt: float = 1.0,
                              ws: float = 1.0,
                              combined_objective: str = COMBINED_OBJECTIVE):
    cand_idx = np.asarray(cand_idx, dtype=int)
    if ref_scales is None:
        ref_scales = compute_reference_scales(temp_truth, salt_truth, wt=wt, ws=ws)
    selected = []
    selected_set = set()

    for _ in range(int(k_max)):
        remaining = np.array([c for c in cand_idx if int(c) not in selected_set], dtype=int)
        if remaining.size == 0:
            raise RuntimeError("Greedy: no candidates left to select.")

        res = Parallel(n_jobs=n_jobs)(
            delayed(_eval_one)(
                c, selected,
                temp_truth, salt_truth,
                EOF_T, temp_mean,
                EOF_S, salt_mean,
                ridge,
                sigmaT_space, sigmaS_space,
                resT_std, resS_std,
                add_obs_noise, use_wls,
                noise_seed,
                ref_scales, wt, ws, combined_objective
            )
            for c in remaining
        )
        best = min(res, key=lambda x: x[0])
        best_c = int(best[1])
        selected.append(best_c)
        selected_set.add(best_c)

    return np.array(selected, dtype=int)


def _eval_one_uncertainty(
    c: int,
    selected: List[int],
    EOF_T: np.ndarray,
    EOF_S: np.ndarray,
    ridge: float,
    sigmaT_space: Optional[np.ndarray],
    sigmaS_space: Optional[np.ndarray],
    resT_std: Optional[np.ndarray],
    resS_std: Optional[np.ndarray],
    wt: float,
    ws: float,
    ref_scales: Dict[str, float],
    combined_objective: str,
) -> Tuple[float, int]:
    idx = np.asarray(list(selected) + [int(c)], dtype=int)

    sigT_sel = None if sigmaT_space is None else np.asarray(sigmaT_space[idx], dtype=float)
    sigS_sel = None if sigmaS_space is None else np.asarray(sigmaS_space[idx], dtype=float)

    Cov_p_T = pc_covariance_wls(EOF_T, idx, sigT_sel, ridge=ridge)
    Cov_p_S = pc_covariance_wls(EOF_S, idx, sigS_sel, ridge=ridge)

    stdT = field_std_from_pc_cov(EOF_T, Cov_p_T)
    stdS = field_std_from_pc_cov(EOF_S, Cov_p_S)

    if MODEL_ERROR_INCLUDE and (resT_std is not None) and (resS_std is not None):
        stdT = np.sqrt(
            np.maximum(stdT, 0.0)**2
            + (float(MODEL_ERROR_FACTOR_TEMP) * np.maximum(np.asarray(resT_std, dtype=float), 0.0))**2
            + float(MODEL_ERROR_FLOOR_TEMP)**2
        )
        stdS = np.sqrt(
            np.maximum(stdS, 0.0)**2
            + (float(MODEL_ERROR_FACTOR_SALT) * np.maximum(np.asarray(resS_std, dtype=float), 0.0))**2
            + float(MODEL_ERROR_FLOOR_SALT)**2
        )

    obj = combined_uncertainty_score(
        stdT, stdS, ref_scales, wt=wt, ws=ws, objective=combined_objective
    )
    return obj, int(c)


def select_sensors_greedy_uncertainty(
    cand_idx, k_max,
    EOF_T: np.ndarray,
    EOF_S: np.ndarray,
    *,
    wt: float = 1.0,
    ws: float = 1.0,
    ridge: float = 1e-6,
    n_jobs: int = -1,
    sigmaT_space: Optional[np.ndarray] = None,
    sigmaS_space: Optional[np.ndarray] = None,
    resT_std: Optional[np.ndarray] = None,
    resS_std: Optional[np.ndarray] = None,
    ref_scales: Optional[Dict[str, float]] = None,
    combined_objective: str = COMBINED_OBJECTIVE,
) -> np.ndarray:
    """Greedy sensor placement minimizing predicted reconstruction uncertainty (no truth required)."""
    cand_idx = np.asarray(cand_idx, dtype=int)
    if ref_scales is None:
        raise ValueError("U-opt with a normalized/density objective requires ref_scales from the reference fields.")
    selected: List[int] = []
    selected_set = set()

    for _ in range(int(k_max)):
        remaining = np.array([c for c in cand_idx if int(c) not in selected_set], dtype=int)
        if remaining.size == 0:
            raise RuntimeError("Greedy-UOPT: no candidates left to select.")

        res = Parallel(n_jobs=n_jobs)(
            delayed(_eval_one_uncertainty)(
                int(c), selected,
                EOF_T, EOF_S,
                ridge,
                sigmaT_space, sigmaS_space,
                resT_std, resS_std,
                wt, ws,
                ref_scales, combined_objective
            )
            for c in remaining
        )
        best = min(res, key=lambda x: x[0])
        best_c = int(best[1])
        selected.append(best_c)
        selected_set.add(best_c)

    return np.array(selected, dtype=int)


def select_sensors_hybrid(
    cand_idx, k_max,
    temp_truth, salt_truth,
    EOF_T, temp_mean,
    EOF_S, salt_mean,
    wt=1.0, ws=1.0,
    ridge=1e-6,
    preselect_frac=0.10,
    preselect_min=50,
    preselect_max=None,
    n_jobs=-1,
    sigmaT_space: Optional[np.ndarray] = None,
    sigmaS_space: Optional[np.ndarray] = None,
    resT_std: Optional[np.ndarray] = None,
    resS_std: Optional[np.ndarray] = None,
    add_obs_noise: bool = False,
    use_wls: bool = False,
    noise_seed: int = 0,
    noise_aware_preselect: bool = False,
    ref_scales: Optional[Dict[str, float]] = None,
    combined_objective: str = COMBINED_OBJECTIVE,
):
    cand_idx = np.asarray(cand_idx, dtype=int)
    if cand_idx.size == 0:
        raise ValueError("cand_idx is empty: no candidates available for hybrid selection.")
    if ref_scales is None:
        ref_scales = compute_reference_scales(temp_truth, salt_truth, wt=wt, ws=ws)

    pre_wt, pre_ws = effective_information_weights(wt, ws, ref_scales)

    pre_n = int(np.ceil(float(preselect_frac) * cand_idx.size))
    if preselect_min is not None:
        pre_n = max(int(preselect_min), pre_n)
    if preselect_max is not None:
        pre_n = min(int(preselect_max), pre_n)

    pre_n = min(pre_n, cand_idx.size)
    pre_n = max(pre_n, int(k_max))

    pre_pool = select_sensors_dopt(
        cand_idx, pre_n,
        EOF_T, EOF_S,
        wt=pre_wt, ws=pre_ws,
        ridge=ridge,
        sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
        noise_aware=noise_aware_preselect
    )

    sel = select_sensors_greedy_rmse(
        pre_pool, int(k_max),
        temp_truth, salt_truth,
        EOF_T, temp_mean,
        EOF_S, salt_mean,
        ridge=ridge,
        n_jobs=n_jobs,
        sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
        resT_std=resT_std, resS_std=resS_std,
        add_obs_noise=add_obs_noise,
        use_wls=use_wls,
        noise_seed=noise_seed,
        ref_scales=ref_scales,
        wt=wt, ws=ws,
        combined_objective=combined_objective
    )
    return np.asarray(sel, dtype=int)


def select_sensors_random(cand_idx, k_max, seed: int = 0) -> np.ndarray:
    cand_idx = np.asarray(cand_idx, dtype=int)
    if cand_idx.size < int(k_max):
        raise ValueError(f"Not enough candidates ({cand_idx.size}) for k_max={int(k_max)}.")
    rng = np.random.RandomState(int(seed))
    return np.sort(rng.choice(cand_idx, size=int(k_max), replace=False)).astype(int)


# =============================================================================
# OUTPUTS + PLOTTING
# =============================================================================

def save_sensors_txt(path, lon, lat, z, sigmaT=None, sigmaS=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Selected 3D sensors (lon, lat, z):\n")
        for i in range(len(lon)):
            if sigmaT is None or sigmaS is None:
                f.write(f"{i+1:2d}: lon={lon[i]:.6f}, lat={lat[i]:.6f}, z={z[i]:.3f}\n")
            else:
                f.write(
                    f"{i+1:2d}: lon={lon[i]:.6f}, lat={lat[i]:.6f}, z={z[i]:.3f}, "
                    f"sigmaT={sigmaT[i]:.4g}, sigmaS={sigmaS[i]:.4g}\n"
                )


def save_metrics_txt(path, method_name, candidate_mode, n_snap, n_space,
                     rmse_t, rmse_s, evT, evS,
                     noise_add_obs, noise_use_wls, noise_aware_selection,
                     sigma_model, sigma_level,
                     ref_scales: Optional[Dict[str, float]] = None,
                     density_metrics: Optional[Dict[str, float]] = None,
                     combined_score: Optional[float] = None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Method: {method_name}\n")
        f.write(f"Candidate mode: {candidate_mode}\n")
        f.write(f"n_snap: {n_snap}\n")
        f.write(f"n_space: {n_space}\n")
        f.write(f"Final RMSE (Temp): {rmse_t:.6g}\n")
        f.write(f"Final RMSE (Salt): {rmse_s:.6g}\n")
        if density_metrics is not None:
            f.write(f"Final density-proxy RMSE: {density_metrics['rmse_density_proxy']:.6g}\n")
            f.write(f"Final density-proxy NRMSE: {density_metrics['nrmse_density_proxy']:.6g}\n")
            f.write(f"Final density-equivalent RMSE: {density_metrics['rmse_density_kg_m3']:.6g} kg m^-3\n")
        if combined_score is not None:
            f.write(f"Final combined objective score ({COMBINED_OBJECTIVE}): {combined_score:.6g}\n")
        f.write(f"NOISE_ADD_OBS: {noise_add_obs}\n")
        f.write(f"NOISE_USE_WLS: {noise_use_wls}\n")
        f.write(f"NOISE_AWARE_SELECTION: {noise_aware_selection}\n")
        f.write(f"SIGMA_MODEL: {sigma_model}\n")
        f.write(f"SIGMA_LEVEL: {sigma_level}\n")
        f.write(f"COMBINED_OBJECTIVE: {COMBINED_OBJECTIVE}\n")
        if ref_scales is not None:
            f.write(f"alpha: {ref_scales['alpha']:.8g} degC^-1\n")
            f.write(f"beta: {ref_scales['beta']:.8g} kg/g\n")
            f.write(f"beta/alpha: {ref_scales['beta_over_alpha']:.8g}\n")
            f.write(f"effective beta/alpha: {ref_scales['effective_beta_over_alpha']:.8g}\n")
            f.write(f"Reference std T anomaly: {ref_scales['std_T_anomaly']:.8g} degC\n")
            f.write(f"Reference std S anomaly: {ref_scales['std_S_anomaly']:.8g} g kg^-1\n")
            f.write(f"Reference std density proxy anomaly: {ref_scales['std_density_proxy_anomaly']:.8g}\n")
            f.write(f"Reference std density anomaly: {ref_scales['std_density_anomaly_kg_m3']:.8g} kg m^-3\n")
        f.write("Explained variance (Temp): " + " ".join([f"{x:.4f}" for x in evT]) + "\n")
        f.write("Explained variance (Salt): " + " ".join([f"{x:.4f}" for x in evS]) + "\n")


def _time_to_iso(t) -> str:
    """Convert numpy datetime64 / cftime / python datetime to ISO string."""
    try:
        # numpy datetime64
        if isinstance(t, np.datetime64):
            return str(np.datetime_as_string(t, unit="s"))
    except Exception:
        pass
    return str(t)


def _try_export_value_shp(path: str, lon: np.ndarray, lat: np.ndarray, attrs: Dict[str, np.ndarray], crs: str) -> None:
    """Export a point shapefile with arbitrary numeric/string attributes.

    This is a best-effort helper: if geopandas/shapely are unavailable, it raises RuntimeError
    so the caller can decide whether to fail or skip.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        raise RuntimeError("geopandas/shapely not available. Install or disable shapefile export.") from e

    geom = [Point(float(x), float(y)) for x, y in zip(lon, lat)]
    gdf = gpd.GeoDataFrame(attrs, geometry=geom, crs=crs)
    gdf.to_file(path, driver="ESRI Shapefile")


def export_postproc_data(
    *,
    out_dir: str,
    csv_dir: str,
    method: str,
    candidate_mode: str,
    data: Dict[str, Any],
    cand_idx: np.ndarray,
    selected_idx: np.ndarray,
    temp_truth: np.ndarray,
    salt_truth: np.ndarray,
    T_hat: np.ndarray,
    S_hat: np.ndarray,
    yT_obs: np.ndarray,
    yS_obs: np.ndarray,
    EOF_T: np.ndarray,
    EOF_S: np.ndarray,
    sigmaT_space: np.ndarray,
    sigmaS_space: np.ndarray,
    stdT_space: Optional[np.ndarray],
    stdS_space: Optional[np.ndarray],
    add_obs_noise: bool,
    use_wls: bool,
    noise_seed: int,
    ref_scales: Optional[Dict[str, float]] = None,
) -> None:
    """Export all data needed for later plotting/post-processing.

    - x/y plot data exported to CSV
    - map-like scatter data exported as Shapefiles (best effort)
    """
    if not EXPORT_POSTPROC:
        return

    os.makedirs(csv_dir, exist_ok=True)

    n_space = int(data["n_space"])
    lon = np.asarray(data["lon_sp"], dtype=float)
    lat = np.asarray(data["lat_sp"], dtype=float)
    zrep = np.asarray(data["z_rep"], dtype=float)
    zsp = np.asarray(data["z_sp"], dtype=float)
    k_sp = np.asarray(data["k_sp"], dtype=np.int64)
    ix = np.asarray(data["ix_sp"], dtype=np.int64)
    iy = np.asarray(data["iy_sp"], dtype=np.int64)
    if ref_scales is None:
        ref_scales = compute_reference_scales(temp_truth, salt_truth, wt=WT, ws=WS)

    # ------------------------------------------------------------------
    # 1) Master per-space table (coordinates + meta + sigmas)
    # ------------------------------------------------------------------
    master_rows = [
        [
            int(i),
            float(lon[i]), float(lat[i]), float(zrep[i]), float(zsp[i]),
            int(k_sp[i]), int(ix[i]), int(iy[i]),
            float(sigmaT_space[i]), float(sigmaS_space[i]),
            int(i in set(map(int, cand_idx))),
            int(i in set(map(int, selected_idx))),
        ]
        for i in range(n_space)
    ]
    _write_csv_rows(
        os.path.join(csv_dir, f"space_table_{method}.csv"),
        [
            "space_idx", "lon", "lat", "z_rep", "z_sp",
            "k", "ix", "iy", "sigma_T", "sigma_S",
            "is_candidate", "is_selected",
        ],
        master_rows,
    )

    # ------------------------------------------------------------------
    # 2) Per-space error & uncertainty (final k)
    # ------------------------------------------------------------------
    rmseT_space = rmse_per_space(T_hat, temp_truth)
    rmseS_space = rmse_per_space(S_hat, salt_truth)
    rho_err = density_proxy(T_hat, S_hat, wt=WT, ws=WS, rho0=1.0) - density_proxy(temp_truth, salt_truth, wt=WT, ws=WS, rho0=1.0)
    rmseRho_space = np.sqrt(np.nanmean(rho_err ** 2, axis=0))
    rmseRhoKg_space = float(RHO0) * rmseRho_space

    stdRho_space = None
    if (stdT_space is not None) and (stdS_space is not None):
        alpha_eff = float(ref_scales["alpha_eff"])
        beta_eff = float(ref_scales["beta_eff"])
        stdRho_space = np.sqrt((alpha_eff * np.asarray(stdT_space, dtype=float)) ** 2
                               + (beta_eff * np.asarray(stdS_space, dtype=float)) ** 2)

    err_rows = []
    for i in range(n_space):
        stdT = float(stdT_space[i]) if stdT_space is not None else float("nan")
        stdS = float(stdS_space[i]) if stdS_space is not None else float("nan")
        stdRho = float(stdRho_space[i]) if stdRho_space is not None else float("nan")
        err_rows.append([
            int(i),
            float(rmseT_space[i]), float(rmseS_space[i]), float(rmseRho_space[i]), float(rmseRhoKg_space[i]),
            stdT, stdS, stdRho, float(RHO0) * stdRho if np.isfinite(stdRho) else float("nan"),
            float(UNCERTAINTY_FACTOR) * stdT if np.isfinite(stdT) else float("nan"),
            float(UNCERTAINTY_FACTOR) * stdS if np.isfinite(stdS) else float("nan"),
            float(UNCERTAINTY_FACTOR) * stdRho if np.isfinite(stdRho) else float("nan"),
        ])
    _write_csv_rows(
        os.path.join(csv_dir, f"space_error_uncertainty_{method}.csv"),
        [
            "space_idx",
            "rmse_T_space", "rmse_S_space", "rmse_density_proxy_space", "rmse_density_kg_m3_space",
            "std_T_space", "std_S_space", "std_density_proxy_space", "std_density_kg_m3_space",
            f"band_T_pm_{UNCERTAINTY_FACTOR:g}sigma",
            f"band_S_pm_{UNCERTAINTY_FACTOR:g}sigma",
            f"band_density_proxy_pm_{UNCERTAINTY_FACTOR:g}sigma",
        ],
        err_rows,
    )

    # ------------------------------------------------------------------
    # 3) RMSE vs predicted uncertainty (scatter data)
    # ------------------------------------------------------------------
    if (stdT_space is not None) and (stdS_space is not None):
        rows_sc = [[
            int(i),
            float(UNCERTAINTY_FACTOR) * float(stdT_space[i]), float(rmseT_space[i]),
            float(UNCERTAINTY_FACTOR) * float(stdS_space[i]), float(rmseS_space[i]),
            float(UNCERTAINTY_FACTOR) * float(stdRho_space[i]) if stdRho_space is not None else float("nan"),
            float(rmseRho_space[i]),
        ] for i in range(n_space)]
        _write_csv_rows(
            os.path.join(csv_dir, f"rmse_vs_uncertainty_{method}.csv"),
            [
                "space_idx",
                "temp_band_pm", "temp_rmse_space",
                "salt_band_pm", "salt_rmse_space",
                "density_proxy_band_pm", "density_proxy_rmse_space",
            ],
            rows_sc,
        )

    # ------------------------------------------------------------------
    # 4) Depth distribution (histogram data)
    # ------------------------------------------------------------------
    z_all = np.asarray(data["z_rep"], dtype=float)
    z_sel = np.asarray(data["z_rep"], dtype=float)[np.asarray(selected_idx, dtype=int)]
    dz = 1.0
    zmin = float(np.nanmin(z_all)); zmax = float(np.nanmax(z_all))
    nb = int(np.clip(np.ceil((zmax - zmin) / dz), 10, 80))
    edges = np.linspace(zmin, zmax, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    all_counts, _ = np.histogram(z_all[np.isfinite(z_all)], bins=edges)
    sel_counts, _ = np.histogram(z_sel[np.isfinite(z_sel)], bins=edges)
    hist_rows = [[
        float(edges[i]), float(edges[i + 1]), float(centers[i]), int(all_counts[i]), int(sel_counts[i])
    ] for i in range(nb)]
    _write_csv_rows(
        os.path.join(csv_dir, f"depth_histogram_{method}.csv"),
        ["bin_left", "bin_right", "bin_center", "count_all", "count_selected"],
        hist_rows,
    )

    # ------------------------------------------------------------------
    # 5) EOF mode maps (export as CSV + per-layer shapefiles)
    # ------------------------------------------------------------------
    # CSV for mode-1 (matches previous "eof_mode1_layers" plot)
    eof1_rows = [[
        int(i), float(EOF_T[i, 0] if EOF_T.shape[1] > 0 else float("nan")),
        float(EOF_S[i, 0] if EOF_S.shape[1] > 0 else float("nan"))
    ] for i in range(n_space)]
    _write_csv_rows(
        os.path.join(csv_dir, f"eof_mode1_per_space_{method}.csv"),
        ["space_idx", "EOF_T_mode1", "EOF_S_mode1"],
        eof1_rows,
    )

    # Optional shapefile exports for EOF1 and uncertainty maps by selected layers
    layers = [0, 9, 19]
    for layer in layers:
        mask = (k_sp == int(layer))
        if not np.any(mask):
            continue

        attrs = {
            "space_idx": np.asarray(np.where(mask)[0], dtype=np.int64),
            "k": k_sp[mask].astype(np.int64),
            "ix": ix[mask].astype(np.int64),
            "iy": iy[mask].astype(np.int64),
            "z_rep": zrep[mask].astype(float),
            "eofT1": (EOF_T[mask, 0] if EOF_T.shape[1] > 0 else np.full(mask.sum(), np.nan)).astype(float),
            "eofS1": (EOF_S[mask, 0] if EOF_S.shape[1] > 0 else np.full(mask.sum(), np.nan)).astype(float),
        }
        if stdT_space is not None:
            attrs["stdT"] = np.asarray(stdT_space, dtype=float)[mask]
            attrs["bandT"] = float(UNCERTAINTY_FACTOR) * np.asarray(stdT_space, dtype=float)[mask]
        if stdS_space is not None:
            attrs["stdS"] = np.asarray(stdS_space, dtype=float)[mask]
            attrs["bandS"] = float(UNCERTAINTY_FACTOR) * np.asarray(stdS_space, dtype=float)[mask]
        if stdRho_space is not None:
            attrs["stdRho"] = np.asarray(stdRho_space, dtype=float)[mask]
            attrs["bandRho"] = float(UNCERTAINTY_FACTOR) * np.asarray(stdRho_space, dtype=float)[mask]

        shp_path = os.path.join(out_dir, f"layer_{int(layer):02d}_maps_{method}.shp")
        try:
            _try_export_value_shp(shp_path, lon[mask], lat[mask], attrs, crs=CANDIDATES_CRS)
        except RuntimeError:
            # Skip optional shapefile export if geospatial dependencies are unavailable.
            pass

    # ------------------------------------------------------------------
    # 6) Time-series exports as CSV
    # ------------------------------------------------------------------
    time_arr = np.asarray(data["time"])
    time_iso = [_time_to_iso(t) for t in time_arr]

    fixed_numbers = [int(x) for x in TS_FIXED_SENSOR_NUMBERS]
    fixed_pos = [i - 1 for i in fixed_numbers if 1 <= i <= len(selected_idx)]
    fixed_space_idx = [int(selected_idx[p]) for p in fixed_pos]

    rng = np.random.RandomState(int(TS_RANDOM_SEED))
    forbidden = set(int(x) for x in fixed_space_idx)
    pool = np.array([i for i in range(n_space) if i not in forbidden], dtype=int)
    rand_space_idx = rng.choice(pool, size=min(int(TS_RANDOM_N), pool.size), replace=False).tolist() if pool.size > 0 else []
    panel_idx = (fixed_space_idx + rand_space_idx)[:10]

    for j in panel_idx:
        j = int(j)
        pos = np.where(np.asarray(selected_idx, dtype=int) == j)[0]
        if pos.size == 1:
            p = int(pos[0])
            t_obs = yT_obs[:, p]
            s_obs = yS_obs[:, p]
            is_sel = 1
        else:
            t_obs = np.full((temp_truth.shape[0],), np.nan)
            s_obs = np.full((salt_truth.shape[0],), np.nan)
            is_sel = 0

        rows_ts = [[
            time_iso[i], int(i), j,
            float(temp_truth[i, j]), float(T_hat[i, j]), float(t_obs[i]) if np.isfinite(t_obs[i]) else float("nan"),
            float(salt_truth[i, j]), float(S_hat[i, j]), float(s_obs[i]) if np.isfinite(s_obs[i]) else float("nan"),
            is_sel,
        ] for i in range(temp_truth.shape[0])]
        _write_csv_rows(
            os.path.join(csv_dir, f"timeseries_spaceidx_{j:06d}_{method}.csv"),
            [
                "time", "t_index", "space_idx",
                "temp_truth", "temp_recon", "temp_obs",
                "salt_truth", "salt_recon", "salt_obs",
                "is_selected_sensor",
            ],
            rows_ts,
        )

    # ------------------------------------------------------------------
    # 7) Export flags used (for reproducibility)
    # ------------------------------------------------------------------
    cfg_rows = [
        ["method", method],
        ["candidate_mode", candidate_mode],
        ["NOISE_ADD_OBS", str(bool(add_obs_noise))],
        ["NOISE_USE_WLS", str(bool(use_wls))],
        ["NOISE_SEED", str(int(noise_seed))],
        ["SIGMA_MODEL", str(SIGMA_MODEL)],
        ["SIGMA_LEVEL", str(SIGMA_LEVEL)],
        ["UNCERTAINTY_FACTOR", str(float(UNCERTAINTY_FACTOR))],
        ["COMBINED_OBJECTIVE", str(COMBINED_OBJECTIVE)],
        ["ALPHA_THERMAL", str(float(ALPHA_THERMAL))],
        ["BETA_HALINE", str(float(BETA_HALINE))],
        ["BETA_OVER_ALPHA", str(float(BETA_HALINE / ALPHA_THERMAL))],
        ["RHO0", str(float(RHO0))],
        ["std_T_anomaly", str(float(ref_scales["std_T_anomaly"]))],
        ["std_S_anomaly", str(float(ref_scales["std_S_anomaly"]))],
        ["std_density_proxy_anomaly", str(float(ref_scales["std_density_proxy_anomaly"]))],
        ["std_density_anomaly_kg_m3", str(float(ref_scales["std_density_anomaly_kg_m3"]))],
    ]
    _write_csv_rows(os.path.join(csv_dir, f"config_{method}.csv"), ["key", "value"], cfg_rows)


# =============================================================================
# RUN HELPERS
# =============================================================================

def _auto_export_flags(candidate_mode: str) -> Tuple[bool, bool]:
    if EXPORT_CANDIDATES_SHP is None:
        export_candidates = (candidate_mode != "all")
    else:
        export_candidates = bool(EXPORT_CANDIDATES_SHP)

    if EXPORT_SELECTED_SHP is None:
        export_selected = True
    else:
        export_selected = bool(EXPORT_SELECTED_SHP)

    return export_candidates, export_selected


def run_one_method(
    method: str,
    candidate_mode: str,
    data_topo: str | Path = DATA_TOPO,
    data_3d: str | Path = DATA_3D,
    out_root: str | Path = OUT_ROOT,
    time_start: str = TIME_START,
    time_end: str = TIME_END,
) -> Dict[str, Any]:
    """
    Run one method end-to-end, including:
    - selection
    - metrics_by_k (k=1..MAX_SENSORS)
    - per-snapshot RMSE exports
    - benchmark timings
    Returns a dict summary (for combined comparison).
    """
    np.random.seed(int(RANDOM_SEED))

    out_dir = os.path.join(str(out_root), method, f"cand_{candidate_mode}")
    os.makedirs(out_dir, exist_ok=True)
    csv_dir = os.path.join(out_dir, "csv_exports")
    os.makedirs(csv_dir, exist_ok=True)

    # Load data + EOFs (shared within this run)
    data_topo = Path(data_topo).expanduser().resolve()
    data_3d = Path(data_3d).expanduser().resolve()
    if not data_topo.is_file():
        raise FileNotFoundError(f"Topography file not found: {data_topo}")
    if not data_3d.is_file():
        raise FileNotFoundError(f"Model data file not found: {data_3d}")

    t0 = time.perf_counter()
    data = load_and_prepare(
        str(data_topo), str(data_3d),
        time_start, time_end,
        require_wet_all_time=REQUIRE_WET_ALL_TIME,
        require_finite_all_time=REQUIRE_FINITE_ALL_TIME,
        hn_min=HN_MIN,
        hv_min=HV_MIN,
        hv_var=HV_VAR
    )
    temp_truth = data["temp_snap"]
    salt_truth = data["salt_snap"]

    # Reference scales for dimensionally consistent combined objectives.
    # Computed once over the full valid 3D reference domain and analysis period.
    ref_scales = compute_reference_scales(temp_truth, salt_truth, wt=WT, ws=WS)
    print_reference_scales(ref_scales, method=method, candidate_mode=candidate_mode)
    info_wt, info_ws = effective_information_weights(WT, WS, ref_scales)

    if EXPORT_CSV:
        _write_csv_rows(
            os.path.join(csv_dir, f"reference_scales_{method}.csv"),
            ["key", "value"],
            [[k, v] for k, v in ref_scales.items()]
            + [["COMBINED_OBJECTIVE", COMBINED_OBJECTIVE],
               ["info_weight_T", info_wt],
               ["info_weight_S", info_ws]]
        )

    sigmaT_space, sigmaS_space = build_sigma_space(data)

    temp_mean, salt_mean, EOF_T, EOF_S, evT, evS, resT_std, resS_std = build_eofs(
        temp_truth, salt_truth, N_MODES_T, N_MODES_S
    )

    cand_idx = make_candidates(
        data=data,
        seed=RANDOM_SEED,
        candidate_mode=candidate_mode,
        n_opt_points=N_OPT_POINTS,
        voxel_dx_m=VOXEL_DX_M,
        voxel_dy_m=VOXEL_DY_M,
        voxel_dz_m=VOXEL_DZ_M,
        max_candidates=MAX_CANDIDATES,
    )

    export_candidates, export_selected = _auto_export_flags(candidate_mode)
    if export_candidates:
        export_points_shp(
            path=os.path.join(out_dir, CANDIDATES_SHP_NAME),
            data=data,
            idx=cand_idx,
            add_vox=(candidate_mode == "voxel"),
            voxel_dx=VOXEL_DX_M,
            voxel_dy=VOXEL_DY_M,
            voxel_dz=VOXEL_DZ_M,
        )

    noise_aware = bool(NOISE_AWARE_SELECTION)
    add_obs_noise = bool(NOISE_ADD_OBS)
    use_wls = bool(NOISE_USE_WLS)
    noise_seed = int(NOISE_SEED)

    # Selection timing
    t_sel0 = time.perf_counter()
    if method == "qdeim":
        selected = select_sensors_qdeim(
            cand_idx, MAX_SENSORS, EOF_T, EOF_S,
            wt=info_wt, ws=info_ws,
            sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
            noise_aware=noise_aware
        )
    elif method == "dopt":
        selected = select_sensors_dopt(
            cand_idx, MAX_SENSORS, EOF_T, EOF_S,
            wt=info_wt, ws=info_ws, ridge=RIDGE,
            sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
            noise_aware=noise_aware
        )
    elif method == "aopt":
        selected = select_sensors_aopt(
            cand_idx, MAX_SENSORS, EOF_T, EOF_S,
            wt=info_wt, ws=info_ws, ridge=RIDGE,
            sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
            noise_aware=noise_aware
        )
    elif method == "rmse":
        selected = select_sensors_greedy_rmse(
            cand_idx, MAX_SENSORS,
            temp_truth, salt_truth,
            EOF_T, temp_mean,
            EOF_S, salt_mean,
            ridge=RIDGE, n_jobs=N_JOBS,
            sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
            resT_std=resT_std, resS_std=resS_std,
            add_obs_noise=add_obs_noise,
            use_wls=use_wls,
            noise_seed=noise_seed,
            ref_scales=ref_scales,
            wt=WT, ws=WS,
            combined_objective=COMBINED_OBJECTIVE
        )

    elif method == "uopt":
        selected = select_sensors_greedy_uncertainty(
            cand_idx, MAX_SENSORS,
            EOF_T, EOF_S,
            wt=WT, ws=WS,
            ridge=RIDGE, n_jobs=N_JOBS,
            sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
            resT_std=resT_std, resS_std=resS_std,
            ref_scales=ref_scales,
            combined_objective=COMBINED_OBJECTIVE,
        )
    elif method == "hybrid":
        selected = select_sensors_hybrid(
            cand_idx, MAX_SENSORS,
            temp_truth, salt_truth,
            EOF_T, temp_mean,
            EOF_S, salt_mean,
            wt=WT, ws=WS,
            ridge=RIDGE,
            preselect_frac=PRESELECT_FRAC,
            preselect_min=PRESELECT_MIN,
            preselect_max=PRESELECT_MAX,
            n_jobs=N_JOBS,
            sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
            resT_std=resT_std, resS_std=resS_std,
            add_obs_noise=add_obs_noise,
            use_wls=use_wls,
            noise_seed=noise_seed,
            noise_aware_preselect=noise_aware,
            ref_scales=ref_scales,
            combined_objective=COMBINED_OBJECTIVE
        )
    elif method == "random":
        selected = select_sensors_random(cand_idx, MAX_SENSORS, seed=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown method={method!r}")
    t_sel1 = time.perf_counter()
    selection_time_s = t_sel1 - t_sel0

    selected_idx, sensor_lon, sensor_lat, sensor_z = map_sensors(
        selected, data["lon_sp"], data["lat_sp"], data["z_sp"]
    )

    # Score final (k=MAX_SENSORS) once
    rmse_t, rmse_s, T_hat, S_hat, yT_obs, yS_obs, stdT_space, stdS_space = reconstruct_and_score(
        temp_truth, salt_truth,
        EOF_T, temp_mean,
        EOF_S, salt_mean,
        selected_idx, RIDGE,
        sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
        resT_std=resT_std, resS_std=resS_std,
        add_obs_noise=add_obs_noise,
        use_wls=use_wls,
        noise_seed=noise_seed
    )
    density_metrics_final = compute_density_proxy_metrics(
        temp_truth, salt_truth, T_hat, S_hat, ref_scales, wt=WT, ws=WS
    )
    combined_score_final = combined_reconstruction_score(
        temp_truth, salt_truth, T_hat, S_hat, rmse_t, rmse_s, ref_scales,
        wt=WT, ws=WS, objective=COMBINED_OBJECTIVE
    )

    if SAVE_SENSORS_TXT:
        sigT_sel = sigmaT_space[selected_idx]
        sigS_sel = sigmaS_space[selected_idx]
        save_sensors_txt(os.path.join(out_dir, "selected_sensors_3D.txt"),
                         sensor_lon, sensor_lat, sensor_z, sigmaT=sigT_sel, sigmaS=sigS_sel)

    if SAVE_METRICS_TXT:
        save_metrics_txt(
            os.path.join(out_dir, "final_metrics_3D.txt"),
            method, candidate_mode, data["n_snap"], data["n_space"],
            rmse_t, rmse_s, evT, evS,
            add_obs_noise, use_wls, noise_aware,
            SIGMA_MODEL, SIGMA_LEVEL,
            ref_scales=ref_scales,
            density_metrics=density_metrics_final,
            combined_score=combined_score_final
        )

    if export_selected:
        export_points_shp(
            path=os.path.join(out_dir, SELECTED_SHP_NAME),
            data=data,
            idx=selected_idx,
            add_vox=(candidate_mode == "voxel"),
            voxel_dx=VOXEL_DX_M,
            voxel_dy=VOXEL_DY_M,
            voxel_dz=VOXEL_DZ_M,
        )

    # ----------------------------
    # Metrics-by-k (main request)
    # ----------------------------
    metrics_rows: List[List[Any]] = []
    header = [
        "method", "candidate_mode", "k",
        "rmse_T", "nrmse_T", "mae_T", "bias_T", "std_truth_T", "std_recon_T",
        "var_truth_T", "var_recon_T", "var_ratio_T", "ev_capture_T", "corr_r_T", "r2_T", "nse_T",
        "rmse_S", "nrmse_S", "mae_S", "bias_S", "std_truth_S", "std_recon_S",
        "var_truth_S", "var_recon_S", "var_ratio_S", "ev_capture_S", "corr_r_S", "r2_S", "nse_S",
        "nrmse_T_ref_anomaly", "nrmse_S_ref_anomaly",
        "rmse_density_proxy", "nrmse_density_proxy", "rmse_density_kg_m3",
        "combined_objective_score", "ref_std_T_anomaly", "ref_std_S_anomaly",
        "ref_std_density_proxy_anomaly", "ref_std_density_kg_m3",
        "recon_time_s"
    ]

    recon_total = 0.0
    for k in range(1, int(MAX_SENSORS) + 1):
        tk0 = time.perf_counter()
        rt, rs, Th, Sh, *_ = reconstruct_and_score(
            temp_truth, salt_truth,
            EOF_T, temp_mean,
            EOF_S, salt_mean,
            selected_idx[:k], RIDGE,
            sigmaT_space=sigmaT_space, sigmaS_space=sigmaS_space,
            resT_std=resT_std, resS_std=resS_std,
            add_obs_noise=add_obs_noise,
            use_wls=use_wls,
            noise_seed=noise_seed + 9973 * k
        )
        tk1 = time.perf_counter()
        dt = tk1 - tk0
        recon_total += dt

        mT = compute_metrics_global(temp_truth, Th)
        mS = compute_metrics_global(salt_truth, Sh)
        mRho = compute_density_proxy_metrics(temp_truth, salt_truth, Th, Sh, ref_scales, wt=WT, ws=WS)
        combined_score_k = combined_reconstruction_score(
            temp_truth, salt_truth, Th, Sh, rt, rs, ref_scales,
            wt=WT, ws=WS, objective=COMBINED_OBJECTIVE
        )

        metrics_rows.append([
            method, candidate_mode, k,
            mT["rmse"], mT["nrmse"], mT["mae"], mT["bias"], mT["std_truth"], mT["std_recon"],
            mT["var_truth"], mT["var_recon"], mT["var_ratio"], mT["ev_capture"], mT["corr_r"], mT["r2"], mT["nse"],
            mS["rmse"], mS["nrmse"], mS["mae"], mS["bias"], mS["std_truth"], mS["std_recon"],
            mS["var_truth"], mS["var_recon"], mS["var_ratio"], mS["ev_capture"], mS["corr_r"], mS["r2"], mS["nse"],
            float(rt) / max(ref_scales["std_T_anomaly"], 1e-12),
            float(rs) / max(ref_scales["std_S_anomaly"], 1e-12),
            mRho["rmse_density_proxy"], mRho["nrmse_density_proxy"], mRho["rmse_density_kg_m3"],
            combined_score_k, ref_scales["std_T_anomaly"], ref_scales["std_S_anomaly"],
            ref_scales["std_density_proxy_anomaly"], ref_scales["std_density_anomaly_kg_m3"],
            dt
        ])

    if EXPORT_CSV:
        _write_csv_rows(
            os.path.join(csv_dir, f"metrics_by_k_{method}.csv"),
            header, metrics_rows
        )

        # Per-space uncertainty (final k), if available
        if (stdT_space is not None) and (stdS_space is not None):
            bandT = float(UNCERTAINTY_FACTOR) * np.asarray(stdT_space, dtype=float)
            bandS = float(UNCERTAINTY_FACTOR) * np.asarray(stdS_space, dtype=float)
            alpha_eff = float(ref_scales["alpha_eff"])
            beta_eff = float(ref_scales["beta_eff"])
            stdRho = np.sqrt((alpha_eff * np.asarray(stdT_space, dtype=float)) ** 2
                             + (beta_eff * np.asarray(stdS_space, dtype=float)) ** 2)
            bandRho = float(UNCERTAINTY_FACTOR) * stdRho
            unc_rows = [[int(i), float(stdT_space[i]), float(stdS_space[i]), float(stdRho[i]), float(RHO0 * stdRho[i]),
                         float(bandT[i]), float(bandS[i]), float(bandRho[i]), float(RHO0 * bandRho[i])]
                        for i in range(int(data["n_space"]))]
            _write_csv_rows(
                os.path.join(csv_dir, f"uncertainty_per_space_final_{method}.csv"),
                ["space_idx", "std_T", "std_S", "std_density_proxy", "std_density_kg_m3",
                 "band_T_pm", "band_S_pm", "band_density_proxy_pm", "band_density_kg_m3_pm"],
                unc_rows
            )

        # Per-snapshot RMSE for final k
        rmseT_snap = rmse_per_snapshot(T_hat, temp_truth)
        rmseS_snap = rmse_per_snapshot(S_hat, salt_truth)
        rho_err_snap = density_proxy(T_hat, S_hat, wt=WT, ws=WS, rho0=1.0) - density_proxy(temp_truth, salt_truth, wt=WT, ws=WS, rho0=1.0)
        rmseRho_snap = np.sqrt(np.nanmean(rho_err_snap ** 2, axis=1))
        snap_rows = [[method, candidate_mode, i, float(rmseT_snap[i]), float(rmseS_snap[i]),
                      float(rmseRho_snap[i]), float(RHO0 * rmseRho_snap[i]),
                      float(rmseRho_snap[i] / max(ref_scales["std_density_proxy_anomaly"], 1e-12))]
                     for i in range(rmseT_snap.size)]
        _write_csv_rows(
            os.path.join(csv_dir, f"metrics_per_snapshot_{method}.csv"),
            ["method", "candidate_mode", "t_index", "rmse_T_snapshot", "rmse_S_snapshot",
             "rmse_density_proxy_snapshot", "rmse_density_kg_m3_snapshot", "nrmse_density_proxy_snapshot"],
            snap_rows
        )

        # Hardware CSV
        hw = collect_hardware_info()
        hw_rows = [[k, v] for k, v in hw.items()]
        _write_csv_rows(os.path.join(csv_dir, f"hardware_{method}.csv"), ["key", "value"], hw_rows)

    # Post-processing exports (no figures)
    export_postproc_data(
        out_dir=out_dir,
        csv_dir=csv_dir,
        method=method,
        candidate_mode=candidate_mode,
        data=data,
        cand_idx=np.asarray(cand_idx, dtype=int),
        selected_idx=np.asarray(selected_idx, dtype=int),
        temp_truth=temp_truth,
        salt_truth=salt_truth,
        T_hat=T_hat,
        S_hat=S_hat,
        yT_obs=yT_obs,
        yS_obs=yS_obs,
        EOF_T=EOF_T,
        EOF_S=EOF_S,
        sigmaT_space=sigmaT_space,
        sigmaS_space=sigmaS_space,
        stdT_space=stdT_space,
        stdS_space=stdS_space,
        add_obs_noise=add_obs_noise,
        use_wls=use_wls,
        noise_seed=noise_seed,
        ref_scales=ref_scales,
    )

    t1 = time.perf_counter()
    total_time_s = t1 - t0

    # Run summary CSV
    if EXPORT_CSV:
        summary_rows = [[
            method, candidate_mode,
            int(data["n_snap"]), int(data["n_space"]), int(np.asarray(cand_idx).size),
            float(rmse_t), float(rmse_s),
            density_metrics_final["rmse_density_proxy"], density_metrics_final["nrmse_density_proxy"],
            density_metrics_final["rmse_density_kg_m3"], combined_score_final,
            ref_scales["std_T_anomaly"], ref_scales["std_S_anomaly"],
            ref_scales["std_density_proxy_anomaly"], ref_scales["std_density_anomaly_kg_m3"],
            float(selection_time_s), float(recon_total), float(total_time_s),
            bool(add_obs_noise), bool(use_wls), bool(noise_aware),
            SIGMA_MODEL, SIGMA_LEVEL, COMBINED_OBJECTIVE
        ]]
        _write_csv_rows(
            os.path.join(csv_dir, f"run_summary_{method}.csv"),
            ["method", "candidate_mode", "n_snap", "n_space", "n_candidates",
             "final_rmse_T", "final_rmse_S",
             "final_rmse_density_proxy", "final_nrmse_density_proxy",
             "final_rmse_density_kg_m3", "final_combined_objective_score",
             "ref_std_T_anomaly", "ref_std_S_anomaly",
             "ref_std_density_proxy_anomaly", "ref_std_density_kg_m3",
             "selection_time_s", "recon_eval_total_time_s", "total_time_s",
             "NOISE_ADD_OBS", "NOISE_USE_WLS", "NOISE_AWARE_SELECTION",
             "SIGMA_MODEL", "SIGMA_LEVEL", "COMBINED_OBJECTIVE"],
            summary_rows
        )

    return {
        "method": method,
        "candidate_mode": candidate_mode,
        "n_snap": int(data["n_snap"]),
        "n_space": int(data["n_space"]),
        "n_candidates": int(np.asarray(cand_idx).size),
        "final_rmse_T": float(rmse_t),
        "final_rmse_S": float(rmse_s),
        "final_rmse_density_proxy": float(density_metrics_final["rmse_density_proxy"]),
        "final_nrmse_density_proxy": float(density_metrics_final["nrmse_density_proxy"]),
        "final_rmse_density_kg_m3": float(density_metrics_final["rmse_density_kg_m3"]),
        "final_combined_objective_score": float(combined_score_final),
        "ref_std_T_anomaly": float(ref_scales["std_T_anomaly"]),
        "ref_std_S_anomaly": float(ref_scales["std_S_anomaly"]),
        "ref_std_density_proxy_anomaly": float(ref_scales["std_density_proxy_anomaly"]),
        "ref_std_density_kg_m3": float(ref_scales["std_density_anomaly_kg_m3"]),
        "selection_time_s": float(selection_time_s),
        "recon_eval_total_time_s": float(recon_total),
        "total_time_s": float(total_time_s),
    }


def main(
    method=DEFAULT_METHOD,
    candidate_mode=CANDIDATE_MODE,
    run_all: bool = False,
    data_topo: str | Path = DATA_TOPO,
    data_3d: str | Path = DATA_3D,
    out_root: str | Path = OUT_ROOT,
    time_start: str = TIME_START,
    time_end: str = TIME_END,
):
    methods = ["qdeim", "dopt", "aopt", "uopt", "hybrid", "random", "rmse"] if run_all else [method]
    summaries: List[Dict[str, Any]] = []
    for m in methods:
        summaries.append(run_one_method(m, candidate_mode, data_topo, data_3d, out_root, time_start, time_end))

    # Combined comparison CSV (one row per method)
    if EXPORT_CSV and len(summaries) > 1:
        out_dir = os.path.join(str(out_root), "_comparison", f"cand_{candidate_mode}")
        os.makedirs(out_dir, exist_ok=True)
        csv_dir = os.path.join(out_dir, "csv_exports")
        os.makedirs(csv_dir, exist_ok=True)

        header = [
            "method", "candidate_mode", "n_snap", "n_space", "n_candidates",
            "final_rmse_T", "final_rmse_S",
            "final_rmse_density_proxy", "final_nrmse_density_proxy",
            "final_rmse_density_kg_m3", "final_combined_objective_score",
            "ref_std_T_anomaly", "ref_std_S_anomaly",
            "ref_std_density_proxy_anomaly", "ref_std_density_kg_m3",
            "selection_time_s", "recon_eval_total_time_s", "total_time_s"
        ]
        rows = [[
            s["method"], s["candidate_mode"], s["n_snap"], s["n_space"], s["n_candidates"],
            s["final_rmse_T"], s["final_rmse_S"],
            s["final_rmse_density_proxy"], s["final_nrmse_density_proxy"],
            s["final_rmse_density_kg_m3"], s["final_combined_objective_score"],
            s["ref_std_T_anomaly"], s["ref_std_S_anomaly"],
            s["ref_std_density_proxy_anomaly"], s["ref_std_density_kg_m3"],
            s["selection_time_s"], s["recon_eval_total_time_s"], s["total_time_s"]
        ] for s in summaries]

        _write_csv_rows(
            os.path.join(csv_dir, "comparison_summary_all_methods.csv"),
            header, rows
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        method=args.method,
        candidate_mode=args.candidate_mode,
        run_all=bool(args.all_methods),
        data_topo=args.topo,
        data_3d=args.data,
        out_root=args.output,
        time_start=args.start,
        time_end=args.end,
    )