"""EOF fitting, sensor selection, weighted reconstruction and uncertainty propagation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.linalg import qr as scipy_qr

DATA_TOPO = str(Path(__file__).resolve().parent / "Data/topo.nc")
DATA_3D = str(Path(__file__).resolve().parent / "Data/3D_mean_2016-03.nc")
TIME_START = "2016-03-01"
TIME_END = "2016-03-02"
VALIDATION_DAY = "2016-03-03"
TEST_DAY = "2016-03-04"
VALIDATION_MODES = [(9, 9)]
VALIDATION_RIDGES = [0.0001]
RANDOM_SEED = 42
MAX_SENSORS = 21
WT = 1.0
WS = 1.0
COMBINED_OBJECTIVE = "density_proxy_nrmse"
ALPHA_THERMAL = 0.00017
BETA_HALINE = 0.00076
RHO0 = 1025.0
USE_DENSITY_WEIGHTS_FOR_INFO_METHODS = True
HN_MIN = 0.01
HV_VAR = "hv"
HV_MIN = 0.0
N_OPT_POINTS = -1
MAX_CANDIDATES = None
REQUIRE_WET_ALL_TIME = True
REQUIRE_FINITE_ALL_TIME = True
PRESELECT_FRAC = 0.1
PRESELECT_MIN = 50
PRESELECT_MAX = None
N_JOBS = -1
VOXEL_DX_M = 500.0
VOXEL_DY_M = 500.0
VOXEL_DZ_M = 1.0
OUT_ROOT = str(Path(__file__).resolve().parent / "outputs/methods")
OUTPUT_PREFIX = ""
NOISE_ADD_OBS = True
NOISE_AWARE_SELECTION = True
NOISE_USE_WLS = True
UNCERTAINTY_BANDS = True
SIGMA_MODEL = "constant"
SIGMA_LEVEL = "low"
SIGMA_LEVELS_TEMP = {"low": 0.01, "medium": 0.02, "high": 0.05}
SIGMA_LEVELS_SALT = {"low": 0.005, "medium": 0.01, "high": 0.02}
SIGMA_TEMP_CUSTOM = 0.02
SIGMA_SALT_CUSTOM = 0.01
DEPTH_SLOPE_TEMP = 0.0
DEPTH_SLOPE_SALT = 0.0
MC_CONFIG_KEYS = (
    "NOISE_USE_WLS",
    "UNCERTAINTY_BANDS",
    "MODEL_ERROR_INCLUDE",
    "MODEL_ERROR_FACTOR_TEMP",
    "MODEL_ERROR_FACTOR_SALT",
    "MODEL_ERROR_FLOOR_TEMP",
    "MODEL_ERROR_FLOOR_SALT",
    "NOISE_ADD_BIAS",
    "NOISE_ADD_DRIFT",
    "BIAS_STD_FACTOR_TEMP",
    "BIAS_STD_FACTOR_SALT",
    "DRIFT_STEP_FACTOR_TEMP",
    "DRIFT_STEP_FACTOR_SALT",
    "ALPHA_THERMAL",
    "BETA_HALINE",
    "RHO0",
    "HN_MIN",
)
MC_NOISE_LEVELS = ("low", "medium", "high")
NOISE_SEED = 1234
MODEL_ERROR_INCLUDE = True
MODEL_ERROR_FACTOR_TEMP = 1.0
MODEL_ERROR_FACTOR_SALT = 1.0
MODEL_ERROR_FLOOR_TEMP = 0.0
MODEL_ERROR_FLOOR_SALT = 0.0
NOISE_ADD_BIAS = False
BIAS_STD_FACTOR_TEMP = 0.5
BIAS_STD_FACTOR_SALT = 0.5
NOISE_ADD_DRIFT = False
DRIFT_STEP_FACTOR_TEMP = 0.05
DRIFT_STEP_FACTOR_SALT = 0.05


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
    """Read the training fields and construct the fixed eligible spatial grid."""
    topo = xr.open_dataset(data_topo)
    ds = xr.open_dataset(data_3d).sel(time=slice(time_start, time_end))
    lon2d = topo["lonc"].values
    lat2d = topo["latc"].values
    time_arr = ds["time"].values
    require_hours(time_arr, time_start, 48)
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
    z = np.cumsum(hn, axis=1) - 0.5 * hn - bath2d[None, None, :, :]
    z0 = z[0]
    lon1, lat1 = (8.32943858828, 53.6033896612)
    lon2, lat2 = (8.49082102239, 53.673277974)
    dx1, dy1 = (lon2 - lon1, lat2 - lat1)
    cross1_2d = dx1 * (lat2d - lat1) - dy1 * (lon2d - lon1)
    lon3, lat3 = (8.33014025104, 53.610466959)
    lon4p, lat4p = (8.33785854137, 53.3762893856)
    dx2, dy2 = (lon3 - lon4p, lat3 - lat4p)
    cross2_2d = dx2 * (lat2d - lat3) - dy2 * (lon2d - lon3)
    ymin2 = min(lat3, lat4p)
    mask2_constraint = np.ones_like(lat2d, dtype=bool)
    apply_zone = lat2d >= ymin2
    mask2_constraint[apply_zone] = cross2_2d[apply_zone] <= 0
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
    topo.close()
    ds.close()
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
        "k_sp": k_sp.astype(np.int64),
    }


def load_heldout(data, day, data_3d=None):
    """Mask unavailable held-out cells without changing the training-grid indices."""
    with xr.open_dataset(data_3d or DATA_3D) as ds:
        ds = ds.sel(time=slice(day, day))
        require_hours(ds.time.values, day, 24)
        idx = (slice(None), data["k_sp"], data["iy_sp"], data["ix_sp"])
        temp, salt, hn = (ds[v].values[idx] for v in ("temp", "salt", "hn"))
        if temp.shape[0] == 0:
            raise ValueError(f"{day}: no snapshots found in {DATA_3D}.")
        valid = np.isfinite(temp) & np.isfinite(salt) & np.isfinite(hn) & (hn > HN_MIN)
        if not valid.any(axis=1).all():
            raise ValueError(f"{day}: a snapshot has no valid wet reference cells.")
        return dict(
            data,
            time=ds.time.values,
            temp_snap=np.where(valid, temp, np.nan),
            salt_snap=np.where(valid, salt, np.nan),
            n_snap=temp.shape[0],
        )


def _lonlat_to_xy_m(lon_deg, lat_deg):
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    good = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(good):
        return (np.full_like(lon, np.nan), np.full_like(lat, np.nan))
    try:
        from pyproj import Transformer

        lon0 = float(np.nanmedian(lon[good]))
        lat0 = float(np.nanmedian(lat[good]))
        zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
        epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
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
        return (x_m, y_m)


def make_candidates(
    data: Dict[str, Any],
    seed: int,
    candidate_mode: str,
    n_opt_points: int,
    voxel_dx_m: float,
    voxel_dy_m: float,
    voxel_dz_m: float,
    max_candidates: Optional[int],
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
    z = np.asarray(data["z_rep"], dtype=float)
    x_m, y_m = _lonlat_to_xy_m(lon, lat)
    dx = float(voxel_dx_m)
    dy = float(voxel_dy_m)
    dz = float(voxel_dz_m)
    x0 = float(np.nanmin(x_m))
    y0 = float(np.nanmin(y_m))
    z0 = float(np.nanmin(z))
    vix = np.floor((x_m - x0) / dx).astype(np.int64)
    viy = np.floor((y_m - y0) / dy).astype(np.int64)
    viz = np.floor((z - z0) / dz).astype(np.int64)
    cx = x0 + (vix + 0.5) * dx
    cy = y0 + (viy + 0.5) * dy
    cz = z0 + (viz + 0.5) * dz
    d2 = (x_m - cx) ** 2 + (y_m - cy) ** 2 + (z - cz) ** 2
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


def _resolve_sigma_levels() -> Tuple[float, float]:
    if SIGMA_LEVEL == "custom":
        return (float(SIGMA_TEMP_CUSTOM), float(SIGMA_SALT_CUSTOM))
    if SIGMA_LEVEL not in SIGMA_LEVELS_TEMP or SIGMA_LEVEL not in SIGMA_LEVELS_SALT:
        raise ValueError(
            f"SIGMA_LEVEL={SIGMA_LEVEL!r} not in allowed keys {sorted(SIGMA_LEVELS_TEMP.keys())} (or use 'custom')."
        )
    return (float(SIGMA_LEVELS_TEMP[SIGMA_LEVEL]), float(SIGMA_LEVELS_SALT[SIGMA_LEVEL]))


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
    return (sigT, sigS)


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


def compute_eofs_timecov(X: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate EOFs from the temporal covariance matrix of training anomalies."""
    n_snap, _n_space = X.shape
    if n_snap < 2:
        raise ValueError("Need at least 2 snapshots to compute EOFs.")
    C = X @ X.T / (n_snap - 1)
    evals, U = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    U = U[:, order]
    r = min(int(n_modes), n_snap)
    U_r = U[:, :r]
    evals_r = np.maximum(evals[:r], 0.0)
    s_r = np.sqrt(evals_r * (n_snap - 1))
    if s_r.size > 0:
        tol = 1e-08 * np.max(s_r)
        keep = s_r > tol
        U_r = U_r[:, keep]
        s_r = s_r[keep]
    EOFs = X.T @ U_r / (s_r[None, :] + 1e-12)
    return (EOFs, s_r)


def fit_pcs_from_sensors_ols(
    EOFs: np.ndarray,
    mean_space: np.ndarray,
    y_obs: np.ndarray,
    sensor_idx: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    if not np.isfinite(y_obs).all():
        pcs = np.empty((y_obs.shape[0], EOFs.shape[1]))
        masks, groups = np.unique(np.isfinite(y_obs), axis=0, return_inverse=True)
        for group, available in enumerate(masks):
            times = np.flatnonzero(groups == group)
            pcs[times] = fit_pcs_from_sensors_ols(
                EOFs, mean_space, y_obs[np.ix_(times, available)], sensor_idx[available], ridge
            )
        return pcs
    r = EOFs.shape[1]
    A = EOFs[sensor_idx, :]
    b = y_obs - mean_space[sensor_idx]
    AtA = A.T @ A
    if ridge > 0:
        AtA = AtA + float(ridge) * np.eye(r)
    rhs = A.T @ b.T
    PCs_hat = np.linalg.solve(AtA, rhs).T
    return PCs_hat


def fit_pcs_from_sensors_wls(
    EOFs: np.ndarray,
    mean_space: np.ndarray,
    y_obs: np.ndarray,
    sensor_idx: np.ndarray,
    sigma_sel: Optional[np.ndarray],
    ridge: float = 0.0,
) -> np.ndarray:
    """Fit coefficients using available observations and inverse-variance weights."""
    if sigma_sel is None:
        return fit_pcs_from_sensors_ols(EOFs, mean_space, y_obs, sensor_idx, ridge=ridge)
    if not np.isfinite(y_obs).all():
        pcs = np.empty((y_obs.shape[0], EOFs.shape[1]))
        masks, groups = np.unique(np.isfinite(y_obs), axis=0, return_inverse=True)
        for group, available in enumerate(masks):
            times = np.flatnonzero(groups == group)
            sigmas = np.broadcast_to(sigma_sel, (len(sensor_idx),))[available]
            pcs[times] = fit_pcs_from_sensors_wls(
                EOFs,
                mean_space,
                y_obs[np.ix_(times, available)],
                sensor_idx[available],
                sigmas,
                ridge,
            )
        return pcs
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
    return float(np.sqrt(np.nanmean((Xhat - X) ** 2)))


def anomaly_reference_scale(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    mean_space = np.nanmean(X, axis=0, keepdims=True)
    X_anom = X - mean_space
    scale = float(np.sqrt(np.nanmean(X_anom**2)))
    return max(scale, 1e-12)


def effective_density_coefficients(wt: float = 1.0, ws: float = 1.0) -> Tuple[float, float]:
    alpha_eff = float(wt) * float(ALPHA_THERMAL)
    beta_eff = float(ws) * float(BETA_HALINE)
    return (alpha_eff, beta_eff)


def density_proxy(
    temp: np.ndarray, salt: np.ndarray, *, wt: float = 1.0, ws: float = 1.0, rho0: float = 1.0
) -> np.ndarray:
    alpha_eff, beta_eff = effective_density_coefficients(wt=wt, ws=ws)
    return float(rho0) * (
        beta_eff * np.asarray(salt, dtype=np.float64)
        - alpha_eff * np.asarray(temp, dtype=np.float64)
    )


def density_proxy_error_rmse(
    T_hat: np.ndarray,
    S_hat: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    *,
    wt: float = 1.0,
    ws: float = 1.0,
    rho0: float = 1.0,
) -> float:
    e = density_proxy(T_hat, S_hat, wt=wt, ws=ws, rho0=rho0) - density_proxy(
        T, S, wt=wt, ws=ws, rho0=rho0
    )
    return float(np.sqrt(np.nanmean(e**2)))


def compute_reference_scales(
    temp_truth: np.ndarray, salt_truth: np.ndarray, *, wt: float = 1.0, ws: float = 1.0
) -> Dict[str, float]:
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


def effective_information_weights(
    wt: float, ws: float, ref_scales: Optional[Dict[str, float]] = None
) -> Tuple[float, float]:
    if not bool(USE_DENSITY_WEIGHTS_FOR_INFO_METHODS):
        return (float(wt), float(ws))
    return (float(wt), float(ws) * float(BETA_HALINE / ALPHA_THERMAL))


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
    objective = str(objective).lower()
    if objective == "density_proxy_nrmse":
        rmse_rho_proxy = density_proxy_error_rmse(
            T_hat, S_hat, temp_truth, salt_truth, wt=wt, ws=ws, rho0=1.0
        )
        return float(rmse_rho_proxy / max(ref_scales["std_density_proxy_anomaly"], 1e-12))
    if objective == "density_component_nrmse":
        alpha_eff = float(ref_scales["alpha_eff"])
        beta_eff = float(ref_scales["beta_eff"])
        denom = max(float(ref_scales["component_density_scale"]), 1e-12)
        return float((alpha_eff * float(rmse_t) + beta_eff * float(rmse_s)) / denom)
    if objective == "weighted_nrmse":
        denom = max(float(wt) + float(ws), 1e-12)
        return float(
            (
                float(wt) * float(rmse_t) / max(ref_scales["std_T_anomaly"], 1e-12)
                + float(ws) * float(rmse_s) / max(ref_scales["std_S_anomaly"], 1e-12)
            )
            / denom
        )
    if objective == "raw_sum":
        return float(float(wt) * float(rmse_t) + float(ws) * float(rmse_s))
    raise ValueError(f"Unknown COMBINED_OBJECTIVE={objective!r}")


def combined_uncertainty_score(
    stdT: np.ndarray,
    stdS: np.ndarray,
    ref_scales: Dict[str, float],
    *,
    wt: float = 1.0,
    ws: float = 1.0,
    objective: str = COMBINED_OBJECTIVE,
) -> float:
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
        return float(
            (alpha_eff * float(np.nanmean(stdT)) + beta_eff * float(np.nanmean(stdS))) / denom
        )
    if objective == "weighted_nrmse":
        denom = max(float(wt) + float(ws), 1e-12)
        return float(
            (
                float(wt) * float(np.nanmean(stdT)) / max(ref_scales["std_T_anomaly"], 1e-12)
                + float(ws) * float(np.nanmean(stdS)) / max(ref_scales["std_S_anomaly"], 1e-12)
            )
            / denom
        )
    if objective == "raw_sum":
        return float(float(wt) * float(np.nanmean(stdT)) + float(ws) * float(np.nanmean(stdS)))
    raise ValueError(f"Unknown COMBINED_OBJECTIVE={objective!r}")


def truncation_residual_std(X_anom: np.ndarray, EOFs: np.ndarray) -> np.ndarray:
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
    evT = sT**2 / np.sum(temp_anom**2)
    evS = sS**2 / np.sum(salt_anom**2)
    resT_std = truncation_residual_std(temp_anom, EOF_T)
    resS_std = truncation_residual_std(salt_anom, EOF_S)
    return (temp_mean, salt_mean, EOF_T, EOF_S, evT, evS, resT_std, resS_std)


def pc_covariance_wls(
    EOFs: np.ndarray, sensor_idx: np.ndarray, sigma_sel: Optional[np.ndarray], ridge: float
) -> np.ndarray:
    """Return the conditional Gaussian coefficient posterior covariance."""
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
        sigma = np.maximum(sigma, 1e-06)
        w = 1.0 / sigma**2
        AtA = A.T @ (w[:, None] * A)
    ridge = float(max(ridge, 1e-08))
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
    paired_noise: bool = False,
):
    """Reconstruct both fields with fixed models and score valid reference cells."""
    sel = np.asarray(selected, dtype=int)
    rng = np.random.RandomState(int(noise_seed))
    yT_true = temp_truth[:, sel]
    yS_true = salt_truth[:, sel]
    sigT_sel = None if sigmaT_space is None else np.asarray(sigmaT_space[sel], dtype=float)
    sigS_sel = None if sigmaS_space is None else np.asarray(sigmaS_space[sel], dtype=float)
    if add_obs_noise and paired_noise:
        observations = []
        for variable, truth, sigmas, bias, drift in (
            (0, yT_true, sigT_sel, BIAS_STD_FACTOR_TEMP, DRIFT_STEP_FACTOR_TEMP),
            (1, yS_true, sigS_sel, BIAS_STD_FACTOR_SALT, DRIFT_STEP_FACTOR_SALT),
        ):
            observations.append(
                np.column_stack(
                    [
                        add_sensor_noise(
                            truth[:, j : j + 1],
                            sigmas[j],
                            np.random.default_rng(
                                np.random.SeedSequence([noise_seed, int(cell), variable])
                            ),
                            add_bias=NOISE_ADD_BIAS,
                            bias_std_factor=bias,
                            add_drift=NOISE_ADD_DRIFT,
                            drift_step_factor=drift,
                        )[:, 0]
                        for j, cell in enumerate(sel)
                    ]
                )
            )
        yT_obs, yS_obs = observations
    elif add_obs_noise:
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
        PCsT_hat = fit_pcs_from_sensors_wls(
            EOF_T, temp_mean, yT_obs, sel, sigma_sel=sigT_sel, ridge=ridge
        )
        PCsS_hat = fit_pcs_from_sensors_wls(
            EOF_S, salt_mean, yS_obs, sel, sigma_sel=sigS_sel, ridge=ridge
        )
    else:
        PCsT_hat = fit_pcs_from_sensors_ols(EOF_T, temp_mean, yT_obs, sel, ridge=ridge)
        PCsS_hat = fit_pcs_from_sensors_ols(EOF_S, salt_mean, yS_obs, sel, ridge=ridge)
    T_hat = reconstruct_from_pcs(EOF_T, temp_mean, PCsT_hat)
    S_hat = reconstruct_from_pcs(EOF_S, salt_mean, PCsS_hat)
    T_hat = np.where(np.isfinite(temp_truth), T_hat, np.nan)
    S_hat = np.where(np.isfinite(salt_truth), S_hat, np.nan)
    rmse_t = rmse_global(T_hat, temp_truth)
    rmse_s = rmse_global(S_hat, salt_truth)
    stdT_space = None
    stdS_space = None
    if UNCERTAINTY_BANDS and use_wls and (sigmaT_space is not None) and (sigmaS_space is not None):
        Cov_p_T = pc_covariance_wls(EOF_T, sel, sigT_sel, ridge=ridge)
        Cov_p_S = pc_covariance_wls(EOF_S, sel, sigS_sel, ridge=ridge)
        stdT_space = field_std_from_pc_cov(EOF_T, Cov_p_T)
        stdS_space = field_std_from_pc_cov(EOF_S, Cov_p_S)
    if UNCERTAINTY_BANDS and use_wls:
        for obs, basis, sigmas, std in (
            (yT_obs, EOF_T, sigT_sel, stdT_space),
            (yS_obs, EOF_S, sigS_sel, stdS_space),
        ):
            if std is not None and (not np.isfinite(obs).all()):
                for available in np.unique(np.isfinite(obs), axis=0):
                    covariance = pc_covariance_wls(basis, sel[available], sigmas[available], ridge)
                    np.maximum(std, field_std_from_pc_cov(basis, covariance), out=std)
    if MODEL_ERROR_INCLUDE and resT_std is not None and (resS_std is not None):
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
    return (rmse_t, rmse_s, T_hat, S_hat, yT_obs, yS_obs, stdT_space, stdS_space)


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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    aa = a[m]
    bb = b[m]
    sa = np.std(aa)
    sb = np.std(bb)
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def compute_metrics_global(truth: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
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
    ev_capture = float("nan")
    if np.isfinite(var_t) and var_t > 0 and np.isfinite(var_e):
        ev_capture = float(1.0 - var_e / var_t)
    var_ratio = float("nan")
    if np.isfinite(var_t) and var_t > 0 and np.isfinite(var_r):
        var_ratio = float(var_r / var_t)
    nrmse = float("nan")
    if np.isfinite(std_t) and std_t > 0:
        nrmse = float(rmse / std_t)
    r = _safe_corr(truth, recon)
    r2 = float("nan") if not np.isfinite(r) else float(r**2)
    nse = float("nan")
    denom = float(np.nanmean((truth - np.nanmean(truth)) ** 2))
    if np.isfinite(denom) and denom > 0:
        nse = float(1.0 - mse / denom)
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


def _write_csv_rows(path: str, header: List[str], rows: List[List[Any]]) -> None:
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def select_sensors_qdeim(
    cand_idx,
    k_max,
    EOF_T,
    EOF_S,
    wt=1.0,
    ws=1.0,
    sigmaT_space: Optional[np.ndarray] = None,
    sigmaS_space: Optional[np.ndarray] = None,
    noise_aware: bool = False,
):
    cand_idx = np.asarray(cand_idx, dtype=int)
    PhiT = EOF_T[cand_idx, :]
    PhiS = EOF_S[cand_idx, :]
    if noise_aware and sigmaT_space is not None and (sigmaS_space is not None):
        sigT = np.asarray(sigmaT_space[cand_idx], dtype=float)
        sigS = np.asarray(sigmaS_space[cand_idx], dtype=float)
        PhiT = PhiT / sigT[:, None]
        PhiS = PhiS / sigS[:, None]
    fracT = float(wt) / max(float(wt) + float(ws), 1e-12)
    kT = max(1, int(round(int(k_max) * fracT)))
    kS = max(0, int(k_max) - kT)
    _Q, _R, pivT = scipy_qr(PhiT.T, mode="economic", pivoting=True)
    idxT_local = np.array(pivT[: min(kT, pivT.size)], dtype=int)
    if kS > 0:
        _Q, _R, pivS = scipy_qr(PhiS.T, mode="economic", pivoting=True)
        idxS_local = np.array(pivS[: min(kS, pivS.size)], dtype=int)
    else:
        idxS_local = np.array([], dtype=int)
    chosen = []
    chosen_set = set()
    for j in cand_idx[idxT_local]:
        jj = int(j)
        if len(chosen) < int(k_max) and jj not in chosen_set:
            chosen.append(jj)
            chosen_set.add(jj)
    for j in cand_idx[idxS_local]:
        jj = int(j)
        if len(chosen) >= int(k_max):
            break
        if jj not in chosen_set:
            chosen.append(jj)
            chosen_set.add(jj)
    if len(chosen) < int(k_max):
        for j in cand_idx:
            jj = int(j)
            if len(chosen) >= int(k_max):
                break
            if jj not in chosen_set:
                chosen.append(jj)
                chosen_set.add(jj)
    return np.array(chosen[: int(k_max)], dtype=int)


def select_sensors_dopt(
    cand_idx,
    k_max,
    EOF_T,
    EOF_S,
    wt=1.0,
    ws=1.0,
    ridge=1e-06,
    sigmaT_space: Optional[np.ndarray] = None,
    sigmaS_space: Optional[np.ndarray] = None,
    noise_aware: bool = False,
):
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
            if noise_aware and sigmaT_space is not None and (sigmaS_space is not None):
                vT = vT / float(sigmaT_space[c])
                vS = vS / float(sigmaS_space[c])
            ATc = AT + vT @ vT.T
            ASc = AS + vS @ vS.T
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
        if noise_aware and sigmaT_space is not None and (sigmaS_space is not None):
            vT = vT / float(sigmaT_space[best_c])
            vS = vS / float(sigmaS_space[best_c])
        AT = AT + vT @ vT.T
        AS = AS + vS @ vS.T
        selected.append(best_c)
        selected_set.add(best_c)
    return np.array(selected, dtype=int)


def select_sensors_aopt(
    cand_idx,
    k_max,
    EOF_T,
    EOF_S,
    wt=1.0,
    ws=1.0,
    ridge=1e-06,
    sigmaT_space: Optional[np.ndarray] = None,
    sigmaS_space: Optional[np.ndarray] = None,
    noise_aware: bool = False,
):
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
            if noise_aware and sigmaT_space is not None and (sigmaS_space is not None):
                vT = vT / float(sigmaT_space[c])
                vS = vS / float(sigmaS_space[c])
            denomT = 1.0 + float((vT.T @ AT_inv @ vT).item())
            denomS = 1.0 + float((vS.T @ AS_inv @ vS).item())
            vT_Ainv2_vT = float((vT.T @ (AT_inv @ AT_inv) @ vT).item())
            vS_Ainv2_vS = float((vS.T @ (AS_inv @ AS_inv) @ vS).item())
            ATc_inv_trace = tr_AT_inv - vT_Ainv2_vT / denomT
            ASc_inv_trace = tr_AS_inv - vS_Ainv2_vS / denomS
            s = float(wt) * ATc_inv_trace + float(ws) * ASc_inv_trace
            if s < best_score:
                best_score = s
                best_c = c
        if best_c is None:
            raise RuntimeError("A-opt: failed to select next sensor (no candidate left).")
        vT = EOF_T[best_c, :].reshape(-1, 1)
        vS = EOF_S[best_c, :].reshape(-1, 1)
        if noise_aware and sigmaT_space is not None and (sigmaS_space is not None):
            vT = vT / float(sigmaT_space[best_c])
            vS = vS / float(sigmaS_space[best_c])
        AT = AT + vT @ vT.T
        AS = AS + vS @ vS.T
        selected.append(best_c)
        selected_set.add(best_c)
    return np.array(selected, dtype=int)


def _eval_one(
    c,
    selected_list,
    temp_truth,
    salt_truth,
    EOF_T,
    temp_mean,
    EOF_S,
    salt_mean,
    ridge,
    sigmaT_space,
    sigmaS_space,
    resT_std,
    resS_std,
    add_obs_noise,
    use_wls,
    noise_seed,
    ref_scales,
    wt,
    ws,
    combined_objective,
):
    sel = np.array(selected_list + [int(c)], dtype=int)
    seed_raw = int(noise_seed) + 1000003 * int(sel.size) + 9176 * int(c)
    seed = int(seed_raw % 2**32)
    rmse_t, rmse_s, _T_hat, _S_hat, _yT_obs, _yS_obs, _stdT, _stdS = reconstruct_and_score(
        temp_truth,
        salt_truth,
        EOF_T,
        temp_mean,
        EOF_S,
        salt_mean,
        sel,
        ridge,
        sigmaT_space=sigmaT_space,
        sigmaS_space=sigmaS_space,
        resT_std=resT_std,
        resS_std=resS_std,
        add_obs_noise=add_obs_noise,
        use_wls=use_wls,
        noise_seed=seed,
    )
    score = combined_reconstruction_score(
        temp_truth,
        salt_truth,
        _T_hat,
        _S_hat,
        rmse_t,
        rmse_s,
        ref_scales,
        wt=wt,
        ws=ws,
        objective=combined_objective,
    )
    return (score, int(c), rmse_t, rmse_s)


def select_sensors_greedy_rmse(
    cand_idx,
    k_max,
    temp_truth,
    salt_truth,
    EOF_T,
    temp_mean,
    EOF_S,
    salt_mean,
    ridge=1e-06,
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
    combined_objective: str = COMBINED_OBJECTIVE,
):
    cand_idx = np.asarray(cand_idx, dtype=int)
    if ref_scales is None:
        ref_scales = compute_reference_scales(temp_truth, salt_truth, wt=wt, ws=ws)
    selected = resume_selection(cand_idx, k_max)
    selected_set = set(selected)
    for _ in range(len(selected), int(k_max)):
        remaining = np.array([c for c in cand_idx if int(c) not in selected_set], dtype=int)
        if remaining.size == 0:
            raise RuntimeError("Greedy: no candidates left to select.")
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
            (
                delayed(_eval_one)(
                    c,
                    selected,
                    temp_truth,
                    salt_truth,
                    EOF_T,
                    temp_mean,
                    EOF_S,
                    salt_mean,
                    ridge,
                    sigmaT_space,
                    sigmaS_space,
                    resT_std,
                    resS_std,
                    add_obs_noise,
                    use_wls,
                    noise_seed,
                    ref_scales,
                    wt,
                    ws,
                    combined_objective,
                )
                for c in remaining
            )
        )
        best = min(res, key=lambda x: x[0])
        best_c = int(best[1])
        selected.append(best_c)
        selected_set.add(best_c)
        save_selection(selected)
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
    if MODEL_ERROR_INCLUDE and resT_std is not None and (resS_std is not None):
        stdT = np.sqrt(
            np.maximum(stdT, 0.0) ** 2
            + (float(MODEL_ERROR_FACTOR_TEMP) * np.maximum(np.asarray(resT_std, dtype=float), 0.0))
            ** 2
            + float(MODEL_ERROR_FLOOR_TEMP) ** 2
        )
        stdS = np.sqrt(
            np.maximum(stdS, 0.0) ** 2
            + (float(MODEL_ERROR_FACTOR_SALT) * np.maximum(np.asarray(resS_std, dtype=float), 0.0))
            ** 2
            + float(MODEL_ERROR_FLOOR_SALT) ** 2
        )
    obj = combined_uncertainty_score(
        stdT, stdS, ref_scales, wt=wt, ws=ws, objective=combined_objective
    )
    return (obj, int(c))


def select_sensors_greedy_uncertainty(
    cand_idx,
    k_max,
    EOF_T: np.ndarray,
    EOF_S: np.ndarray,
    *,
    wt: float = 1.0,
    ws: float = 1.0,
    ridge: float = 1e-06,
    n_jobs: int = -1,
    sigmaT_space: Optional[np.ndarray] = None,
    sigmaS_space: Optional[np.ndarray] = None,
    resT_std: Optional[np.ndarray] = None,
    resS_std: Optional[np.ndarray] = None,
    ref_scales: Optional[Dict[str, float]] = None,
    combined_objective: str = COMBINED_OBJECTIVE,
) -> np.ndarray:
    cand_idx = np.asarray(cand_idx, dtype=int)
    if ref_scales is None:
        raise ValueError(
            "U-opt with a normalized/density objective requires ref_scales from the reference fields."
        )
    selected: List[int] = resume_selection(cand_idx, k_max)
    selected_set = set(selected)
    for _ in range(len(selected), int(k_max)):
        remaining = np.array([c for c in cand_idx if int(c) not in selected_set], dtype=int)
        if remaining.size == 0:
            raise RuntimeError("Greedy-UOPT: no candidates left to select.")
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
            (
                delayed(_eval_one_uncertainty)(
                    int(c),
                    selected,
                    EOF_T,
                    EOF_S,
                    ridge,
                    sigmaT_space,
                    sigmaS_space,
                    resT_std,
                    resS_std,
                    wt,
                    ws,
                    ref_scales,
                    combined_objective,
                )
                for c in remaining
            )
        )
        best = min(res, key=lambda x: x[0])
        best_c = int(best[1])
        selected.append(best_c)
        selected_set.add(best_c)
        save_selection(selected)
    return np.array(selected, dtype=int)


def select_sensors_hybrid(
    cand_idx,
    k_max,
    temp_truth,
    salt_truth,
    EOF_T,
    temp_mean,
    EOF_S,
    salt_mean,
    wt=1.0,
    ws=1.0,
    ridge=1e-06,
    preselect_frac=0.1,
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
        cand_idx,
        pre_n,
        EOF_T,
        EOF_S,
        wt=pre_wt,
        ws=pre_ws,
        ridge=ridge,
        sigmaT_space=sigmaT_space,
        sigmaS_space=sigmaS_space,
        noise_aware=noise_aware_preselect,
    )
    sel = select_sensors_greedy_rmse(
        pre_pool,
        int(k_max),
        temp_truth,
        salt_truth,
        EOF_T,
        temp_mean,
        EOF_S,
        salt_mean,
        ridge=ridge,
        n_jobs=n_jobs,
        sigmaT_space=sigmaT_space,
        sigmaS_space=sigmaS_space,
        resT_std=resT_std,
        resS_std=resS_std,
        add_obs_noise=add_obs_noise,
        use_wls=use_wls,
        noise_seed=noise_seed,
        ref_scales=ref_scales,
        wt=wt,
        ws=ws,
        combined_objective=combined_objective,
    )
    return np.asarray(sel, dtype=int)


def select_sensors_random(cand_idx, k_max, seed: int = 0) -> np.ndarray:
    cand_idx = np.asarray(cand_idx, dtype=int)
    if cand_idx.size < int(k_max):
        raise ValueError(f"Not enough candidates ({cand_idx.size}) for k_max={int(k_max)}.")
    rng = np.random.RandomState(int(seed))
    return rng.choice(cand_idx, size=int(k_max), replace=False).astype(int)


def run_one_method(
    method: str,
    candidate_mode: str,
    training_data=None,
    fixed_selected=None,
    candidate_indices=None,
    reference_scales=None,
) -> Dict[str, Any]:
    """Fit a training-derived layout and save the frozen reconstruction model."""
    np.random.seed(int(RANDOM_SEED))
    out_dir = OUT_ROOT
    os.makedirs(out_dir, exist_ok=True)
    csv_dir = out_dir
    os.makedirs(csv_dir, exist_ok=True)
    data = (
        training_data
        if training_data is not None
        else load_and_prepare(
            DATA_TOPO,
            DATA_3D,
            TIME_START,
            TIME_END,
            require_wet_all_time=REQUIRE_WET_ALL_TIME,
            require_finite_all_time=REQUIRE_FINITE_ALL_TIME,
            hn_min=HN_MIN,
            hv_min=HV_MIN,
            hv_var=HV_VAR,
        )
    )
    temp_truth = data["temp_snap"]
    salt_truth = data["salt_snap"]
    ref_scales = (
        reference_scales
        if reference_scales is not None
        else compute_reference_scales(temp_truth, salt_truth, wt=WT, ws=WS)
    )
    info_wt, info_ws = effective_information_weights(WT, WS, ref_scales)
    sigmaT_space, sigmaS_space = build_sigma_space(data)
    cand_idx = (
        np.asarray(candidate_indices)
        if candidate_indices is not None
        else make_candidates(
            data=data,
            seed=RANDOM_SEED,
            candidate_mode=candidate_mode,
            n_opt_points=N_OPT_POINTS,
            voxel_dx_m=VOXEL_DX_M,
            voxel_dy_m=VOXEL_DY_M,
            voxel_dz_m=VOXEL_DZ_M,
            max_candidates=MAX_CANDIDATES,
        )
    )
    noise_aware = bool(NOISE_AWARE_SELECTION)
    add_obs_noise = bool(NOISE_ADD_OBS)
    use_wls = bool(NOISE_USE_WLS)
    noise_seed = int(NOISE_SEED)
    validation = load_heldout(data, VALIDATION_DAY)
    validation_rows = []
    best_score = np.inf
    for modes_t, modes_s in VALIDATION_MODES:
        fitted = build_eofs(temp_truth, salt_truth, modes_t, modes_s)
        temp_mean, salt_mean, EOF_T, EOF_S, evT, evS, resT_std, resS_std = fitted
        for ridge in VALIDATION_RIDGES:
            if fixed_selected is not None:
                selected = np.asarray(fixed_selected, dtype=int).copy()
            elif method == "qdeim":
                selected = select_sensors_qdeim(
                    cand_idx,
                    MAX_SENSORS,
                    EOF_T,
                    EOF_S,
                    wt=info_wt,
                    ws=info_ws,
                    sigmaT_space=sigmaT_space,
                    sigmaS_space=sigmaS_space,
                    noise_aware=noise_aware,
                )
            elif method == "dopt":
                selected = select_sensors_dopt(
                    cand_idx,
                    MAX_SENSORS,
                    EOF_T,
                    EOF_S,
                    wt=info_wt,
                    ws=info_ws,
                    ridge=ridge,
                    sigmaT_space=sigmaT_space,
                    sigmaS_space=sigmaS_space,
                    noise_aware=noise_aware,
                )
            elif method == "aopt":
                selected = select_sensors_aopt(
                    cand_idx,
                    MAX_SENSORS,
                    EOF_T,
                    EOF_S,
                    wt=info_wt,
                    ws=info_ws,
                    ridge=ridge,
                    sigmaT_space=sigmaT_space,
                    sigmaS_space=sigmaS_space,
                    noise_aware=noise_aware,
                )
            elif method == "rmse":
                selected = select_sensors_greedy_rmse(
                    cand_idx,
                    MAX_SENSORS,
                    temp_truth,
                    salt_truth,
                    EOF_T,
                    temp_mean,
                    EOF_S,
                    salt_mean,
                    ridge=ridge,
                    n_jobs=N_JOBS,
                    sigmaT_space=sigmaT_space,
                    sigmaS_space=sigmaS_space,
                    resT_std=resT_std,
                    resS_std=resS_std,
                    add_obs_noise=add_obs_noise,
                    use_wls=use_wls,
                    noise_seed=noise_seed,
                    ref_scales=ref_scales,
                    wt=WT,
                    ws=WS,
                    combined_objective=COMBINED_OBJECTIVE,
                )
            elif method == "uopt":
                selected = select_sensors_greedy_uncertainty(
                    cand_idx,
                    MAX_SENSORS,
                    EOF_T,
                    EOF_S,
                    wt=WT,
                    ws=WS,
                    ridge=ridge,
                    n_jobs=N_JOBS,
                    sigmaT_space=sigmaT_space,
                    sigmaS_space=sigmaS_space,
                    resT_std=resT_std,
                    resS_std=resS_std,
                    ref_scales=ref_scales,
                    combined_objective=COMBINED_OBJECTIVE,
                )
            elif method == "hybrid":
                selected = select_sensors_hybrid(
                    cand_idx,
                    MAX_SENSORS,
                    temp_truth,
                    salt_truth,
                    EOF_T,
                    temp_mean,
                    EOF_S,
                    salt_mean,
                    wt=WT,
                    ws=WS,
                    ridge=ridge,
                    preselect_frac=PRESELECT_FRAC,
                    preselect_min=PRESELECT_MIN,
                    preselect_max=PRESELECT_MAX,
                    n_jobs=N_JOBS,
                    sigmaT_space=sigmaT_space,
                    sigmaS_space=sigmaS_space,
                    resT_std=resT_std,
                    resS_std=resS_std,
                    add_obs_noise=add_obs_noise,
                    use_wls=use_wls,
                    noise_seed=noise_seed,
                    noise_aware_preselect=noise_aware,
                    ref_scales=ref_scales,
                    combined_objective=COMBINED_OBJECTIVE,
                )
            elif method == "random":
                selected = select_sensors_random(cand_idx, MAX_SENSORS, seed=RANDOM_SEED)
            else:
                raise ValueError(f"Unknown method={method!r}")
            result = reconstruct_and_score(
                validation["temp_snap"],
                validation["salt_snap"],
                EOF_T,
                temp_mean,
                EOF_S,
                salt_mean,
                selected,
                ridge,
                sigmaT_space=sigmaT_space,
                sigmaS_space=sigmaS_space,
                resT_std=resT_std,
                resS_std=resS_std,
                add_obs_noise=add_obs_noise,
                use_wls=use_wls,
                noise_seed=noise_seed + 1,
            )
            rt, rs, Th, Sh, _, _, st, ss = result
            score = combined_reconstruction_score(
                validation["temp_snap"],
                validation["salt_snap"],
                Th,
                Sh,
                rt,
                rs,
                ref_scales,
                wt=WT,
                ws=WS,
                objective=COMBINED_OBJECTIVE,
            )
            validation_rows.append([modes_t, modes_s, ridge, score])
            if score < best_score:
                best_score = score
                best = (fitted, np.asarray(selected).copy(), ridge, result)
    (
        (temp_mean, salt_mean, EOF_T, EOF_S, evT, evS, resT_std, resS_std),
        selected,
        ridge,
        val_result,
    ) = best
    _write_csv_rows(
        output_path(csv_dir, "validation_settings.csv"),
        ["requested_modes_T", "requested_modes_S", "ridge", "validation_score"],
        validation_rows,
    )
    np.savez_compressed(
        output_path(out_dir, "frozen_model.npz"),
        **{key: data[key] for key in ("k_sp", "iy_sp", "ix_sp", "lon_sp", "lat_sp", "z_rep")},
        temp_mean=temp_mean,
        salt_mean=salt_mean,
        EOF_T=EOF_T,
        EOF_S=EOF_S,
        resT_std=resT_std,
        resS_std=resS_std,
        selected=selected,
        ridge=ridge,
        sigmaT_space=sigmaT_space,
        sigmaS_space=sigmaS_space,
        ref_keys=np.array(list(ref_scales)),
        ref_values=np.array(list(ref_scales.values())),
        mc_config=json.dumps({key: globals()[key] for key in MC_CONFIG_KEYS}),
        validation_data_file=DATA_3D,
        test_data_file=DATA_3D,
        training_start=TIME_START,
        training_end=TIME_END,
        validation_day=VALIDATION_DAY,
        test_day=TEST_DAY,
        wt=WT,
        ws=WS,
        objective=COMBINED_OBJECTIVE,
        sigma_base_T=_resolve_sigma_levels()[0],
        sigma_base_S=_resolve_sigma_levels()[1],
    )
    return dict(
        method=method,
        n_space=data["n_space"],
        n_candidates=len(cand_idx),
        validation_score=best_score,
    )


def output_path(folder, name):
    return os.path.join(folder, (OUTPUT_PREFIX + "__" if OUTPUT_PREFIX else "") + name)


def require_hours(times, start, count):
    expected = np.datetime64(start, "ns") + np.arange(count) * np.timedelta64(1, "h")
    if not np.array_equal(np.asarray(times, dtype="datetime64[ns]"), expected):
        raise ValueError(
            f"{start}: expected exactly {count} consecutive hourly records starting at 00:00."
        )


def validate_model_split(model):
    month = str(model["training_start"])[:7]
    if (str(model["training_start"]), str(model["training_end"]), str(model["validation_day"])) != (
        month + "-01",
        month + "-02",
        month + "-03",
    ) or not str(model["test_day"]).endswith("-04"):
        raise ValueError("Checkpoint does not use the corrected 48/24/24 split. Rebuild it.")


def resume_selection(candidates, budget):
    """Read and validate the completed greedy sensor-selection prefix."""
    path = Path(OUT_ROOT) / (OUTPUT_PREFIX + "__selection_checkpoint.json")
    selected = json.loads(path.read_text())["selected"] if path.exists() else []
    if (
        len(set(selected)) != len(selected)
        or len(selected) > budget
        or (not set(selected).issubset(set(candidates)))
    ):
        raise ValueError("Selection checkpoint does not match the candidate set.")
    return selected


def save_selection(selected):
    path = Path(OUT_ROOT) / (OUTPUT_PREFIX + "__selection_checkpoint.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(selected=selected)))
    temporary.replace(path)
    print(f"{OUTPUT_PREFIX}: selected {len(selected)} sensors", flush=True)
