import json

import numpy as np
import xarray as xr

import eof_core as core
import settings as cfg


def read_model(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def training():
    return core.load_and_prepare(
        core.DATA_TOPO,
        core.DATA_3D,
        core.TIME_START,
        core.TIME_END,
        core.REQUIRE_WET_ALL_TIME,
        core.REQUIRE_FINITE_ALL_TIME,
        core.HN_MIN,
        core.HV_MIN,
        core.HV_VAR,
    )


def candidates(data):
    return core.make_candidates(
        data,
        core.RANDOM_SEED,
        cfg.CANDIDATES,
        core.N_OPT_POINTS,
        core.VOXEL_DX_M,
        core.VOXEL_DY_M,
        core.VOXEL_DZ_M,
        core.MAX_CANDIDATES,
    )


def prepare_random():
    path = cfg.model("random", "reference")
    if path.exists():
        core.validate_model_split(read_model(path))
        return
    data = training()
    pool = candidates(data)
    temp_mean, salt_mean, eof_t, eof_s, _, _, residual_t, residual_s = core.build_eofs(
        data["temp_snap"], data["salt_snap"], cfg.RANK, cfg.RANK
    )
    if min(eof_t.shape[1], eof_s.shape[1]) != cfg.RANK:
        raise ValueError("The training data do not support the shared EOF count.")
    sigma_t, sigma_s = core.build_sigma_space(data)
    reference = core.compute_reference_scales(
        data["temp_snap"], data["salt_snap"], wt=core.WT, ws=core.WS
    )
    np.save(path.parent / "candidate_indices.npy", pool)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        **{key: data[key] for key in ("k_sp", "iy_sp", "ix_sp", "lon_sp", "lat_sp", "z_rep")},
        temp_mean=temp_mean,
        salt_mean=salt_mean,
        EOF_T=eof_t,
        EOF_S=eof_s,
        resT_std=residual_t,
        resS_std=residual_s,
        selected=np.array([], dtype=int),
        ridge=cfg.RIDGE,
        sigmaT_space=sigma_t,
        sigmaS_space=sigma_s,
        ref_keys=np.array(list(reference)),
        ref_values=np.array(list(reference.values())),
        mc_config=json.dumps({key: getattr(core, key) for key in core.MC_CONFIG_KEYS}),
        validation_data_file=core.DATA_3D,
        test_data_file=core.DATA_3D,
        training_start=core.TIME_START,
        training_end=core.TIME_END,
        validation_day=core.VALIDATION_DAY,
        test_day=core.TEST_DAY,
        wt=core.WT,
        ws=core.WS,
        objective=core.COMBINED_OBJECTIVE,
        sigma_base_T=core._resolve_sigma_levels()[0],
        sigma_base_S=core._resolve_sigma_levels()[1],
    )
    temporary.replace(path)


def grid_ids(data, shape):
    return np.ravel_multi_index(tuple((data[key] for key in ("k_sp", "iy_sp", "ix_sp"))), shape)


def prepare_adaptive():
    root = cfg.OUTPUT / "adaptive" / ".work"
    common, shape, coordinates = (None, None, None)
    for name, month in cfg.MONTHS.items():
        cfg.configure(core, "adaptive", name, month)
        with xr.open_dataset(core.DATA_3D) as ds:
            current_shape = ds.temp.shape[1:]
            current_coordinates = {
                dim: ds[dim].values for dim in ds.temp.dims[1:] if dim in ds.coords
            }
        if shape is None:
            shape, coordinates = (current_shape, current_coordinates)
        if (
            current_shape != shape
            or current_coordinates.keys() != coordinates.keys()
            or any(
                (
                    not np.array_equal(coordinates[key], current_coordinates[key])
                    for key in coordinates
                )
            )
        ):
            raise ValueError("Seasonal files must use the same model grid.")
        data = training()
        ids = grid_ids(data, shape)
        common = ids if common is None else np.intersect1d(common, ids)
        del data
    if len(common) < core.MAX_SENSORS:
        raise ValueError("Common training domain is smaller than the sensor budget.")
    np.savez(root / "common_grid.npz", ids=common, shape=shape)
    (root / "protocol.json").write_text(
        json.dumps(
            dict(
                domain="intersection of four training domains",
                interpretation="retrospective seasonal comparison",
                locations="fixed model layer/y/x indices",
                training="days 1-2",
                validation="day 3",
                test="day 4",
                reference_scales="March training",
            ),
            indent=2,
        )
    )


def adaptive_training():
    with np.load(cfg.OUTPUT / "adaptive/.work/common_grid.npz") as saved:
        common, shape = (saved["ids"], saved["shape"])
    data = training()
    indices = np.searchsorted(grid_ids(data, shape), common)
    for key in ("lon_sp", "lat_sp", "z_sp", "z_rep", "k_sp", "iy_sp", "ix_sp"):
        data[key] = data[key][indices]
    for key in ("temp_snap", "salt_snap"):
        data[key] = data[key][:, indices]
    data["n_space"] = len(common)
    return data


def fit_adaptive(name):
    cfg.configure(core, "adaptive", f"{name}_adaptive", cfg.MONTHS[name])
    data = adaptive_training()
    root = cfg.OUTPUT / "adaptive" / ".work"
    if name == "march":
        pool = candidates(data)
        reference = core.compute_reference_scales(
            data["temp_snap"], data["salt_snap"], wt=core.WT, ws=core.WS
        )
        np.save(root / "candidate_indices.npy", pool)
    else:
        pool = np.load(root / "candidate_indices.npy")
        march = read_model(cfg.model("adaptive", "march_adaptive"))
        reference = dict(zip(march["ref_keys"].tolist(), march["ref_values"].tolist()))
    core.run_one_method(
        "dopt",
        cfg.CANDIDATES,
        training_data=data,
        candidate_indices=pool,
        reference_scales=reference,
    )
    if name != "march":
        core.OUTPUT_PREFIX = f"{name}_updated_eofs"
        core.run_one_method(
            "dopt",
            cfg.CANDIDATES,
            training_data=data,
            fixed_selected=march["selected"],
            candidate_indices=pool,
            reference_scales=reference,
        )
