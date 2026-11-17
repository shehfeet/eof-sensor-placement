import hashlib
import json

import numpy as np
import xarray as xr

import eof_core as core

MODE_LEVELS = list(range(5, 16))
SALT_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]
NOISE_LEVELS = {
    "low": (0.01, 0.005),
    "medium": (0.02, 0.01),
    "high": (0.05, 0.02),
    "very_high": (0.1, 0.05),
    "extreme": (0.5, 0.1),
}
N_REALIZATIONS = 1


def read_model(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def save_csv(path, header, rows):
    core._write_csv_rows(str(path), header, rows)


def case_path(root, case, mode):
    return root / f"{case}__frozen_model.npz"


def fit_layout(model, candidates, multiplier=1.0):
    wt, ws = core.effective_information_weights(float(model["wt"]), float(model["ws"]))
    return core.select_sensors_dopt(
        candidates,
        len(model["selected"]),
        model["EOF_T"],
        model["EOF_S"],
        wt=wt,
        ws=ws * multiplier,
        ridge=float(model["ridge"]),
        sigmaT_space=model["sigmaT_space"],
        sigmaS_space=model["sigmaS_space"],
        noise_aware=core.NOISE_AWARE_SELECTION,
    )


def training_spectrum(data, root):
    rows = []
    for variable, key in [("T", "temp_snap"), ("S", "salt_snap")]:
        anomalies = data[key] - data[key].mean(axis=0)
        values = np.maximum(np.linalg.eigvalsh(anomalies @ anomalies.T)[::-1], 0)
        fractions = values / values.sum()
        rows.extend(
            (
                [variable, i + 1, fraction, cumulative]
                for i, (fraction, cumulative) in enumerate(zip(fractions, fractions.cumsum()))
            )
        )
    save_csv(
        root / "training_explained_variance.csv",
        ["variable", "mode", "variance_fraction", "cumulative_fraction"],
        rows,
    )


def prepare_cases(baseline_path, root, mode):
    baseline = read_model(baseline_path)
    core.validate_model_split(baseline)
    if json.loads(str(baseline["mc_config"])) != {
        key: getattr(core, key) for key in core.MC_CONFIG_KEYS
    }:
        raise ValueError(
            "Restore the methods-run reconstruction settings before running sensitivity."
        )
    core.DATA_3D = str(baseline.get("validation_data_file", core.DATA_3D))
    data = core.load_and_prepare(
        core.DATA_TOPO,
        core.DATA_3D,
        str(baseline["training_start"]),
        str(baseline["training_end"]),
        core.REQUIRE_WET_ALL_TIME,
        core.REQUIRE_FINITE_ALL_TIME,
        core.HN_MIN,
        core.HV_MIN,
        core.HV_VAR,
    )
    if any((not np.array_equal(data[key], baseline[key]) for key in ("k_sp", "iy_sp", "ix_sp"))):
        raise ValueError(
            "Training grid differs from the baseline checkpoint. Use the original methods inputs."
        )
    if not (
        np.allclose(data["temp_snap"].mean(0), baseline["temp_mean"])
        and np.allclose(data["salt_snap"].mean(0), baseline["salt_mean"])
    ):
        raise ValueError("Training means differ from the saved baseline.")
    candidates = core.make_candidates(
        data,
        core.RANDOM_SEED,
        mode,
        core.N_OPT_POINTS,
        core.VOXEL_DX_M,
        core.VOXEL_DY_M,
        core.VOXEL_DZ_M,
        core.MAX_CANDIDATES,
    )
    if not np.array_equal(fit_layout(baseline, candidates), baseline["selected"]):
        raise ValueError(
            "Baseline D-opt layout cannot be reproduced; check candidate, weighting and noise settings."
        )
    np.save(root / "candidate_indices.npy", candidates)
    training_spectrum(data, root)
    validation = core.load_heldout(data, str(baseline["validation_day"]))
    split_rows = []
    for period, path, start, end in (
        ("training", core.DATA_3D, str(baseline["training_start"]), str(baseline["training_end"])),
        (
            "validation",
            core.DATA_3D,
            str(baseline["validation_day"]),
            str(baseline["validation_day"]),
        ),
        (
            "test",
            str(baseline.get("test_data_file", core.DATA_3D)),
            str(baseline["test_day"]),
            str(baseline["test_day"]),
        ),
    ):
        with xr.open_dataset(path) as ds:
            times = ds.time.sel(time=slice(start, end)).values
        if not len(times):
            raise ValueError(f"No {period} snapshots in {path}.")
        split_rows.append([period, len(times), str(times[0]), str(times[-1]), path])
    save_csv(
        root / "data_split.csv",
        ["period", "n_snapshots", "first_time", "last_time", "source"],
        split_rows,
    )
    ref = dict(zip(baseline["ref_keys"].tolist(), baseline["ref_values"].tolist()))
    baseline_pair = (float(baseline["sigma_base_T"]), float(baseline["sigma_base_S"]))
    reference_noise = next(
        (name for name, pair in NOISE_LEVELS.items() if pair == baseline_pair), "baseline"
    )
    levels = dict(NOISE_LEVELS)
    levels[reference_noise] = baseline_pair
    manifest = {
        "cases": {},
        "experiments": [],
        "mode": mode,
        "noise_levels": levels,
        "reference_noise": reference_noise,
        "n_realizations": N_REALIZATIONS,
        "baseline_path": str(baseline_path.resolve()),
        "noise_seed": core.NOISE_SEED,
    }
    digest = hashlib.sha256()
    with baseline_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest["baseline_sha256"] = digest.hexdigest()
    validation_rows, location_rows, overlap_rows = ([], [], [])

    def register(case, model, multiplier):
        path = case_path(root, case, mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **model)
        wt, ws = core.effective_information_weights(float(model["wt"]), float(model["ws"]))
        manifest["cases"][case] = {
            "levels": [],
            "modes_T": model["EOF_T"].shape[1],
            "modes_S": model["EOF_S"].shape[1],
            "ridge": float(model["ridge"]),
            "salinity_multiplier": multiplier,
            "selection_weight_T": wt,
            "selection_weight_S": ws * multiplier,
        }
        for order, cell in enumerate(model["selected"], 1):
            location_rows.append(
                [
                    case,
                    order,
                    cell,
                    data["lon_sp"][cell],
                    data["lat_sp"][cell],
                    data["z_rep"][cell],
                    data["k_sp"][cell],
                    data["iy_sp"][cell],
                    data["ix_sp"][cell],
                ]
            )
            common = len(set(model["selected"][:order]) & set(baseline["selected"][:order]))
            overlap_rows.append(
                [case, order, common, common / order, common / (2 * order - common)]
            )

    def experiment(axis, value, case, level):
        manifest["experiments"].append(
            {"axis": axis, "value": value, "case": case, "noise_level": level}
        )
        if level not in manifest["cases"][case]["levels"]:
            manifest["cases"][case]["levels"].append(level)

    register("baseline", baseline, 1.0)
    rank_rows = []
    for rank in MODE_LEVELS:
        if (rank, rank) == (baseline["EOF_T"].shape[1], baseline["EOF_S"].shape[1]):
            case = "baseline"
            actual_t = actual_s = rank
        else:
            fitted = core.build_eofs(data["temp_snap"], data["salt_snap"], rank, rank)
            actual_t, actual_s = (fitted[2].shape[1], fitted[3].shape[1])
            if (actual_t, actual_s) != (rank, rank):
                rank_rows.append(
                    [rank, actual_t, actual_s, "skipped: exceeds numerical training rank"]
                )
                continue
            model = dict(baseline)
            for key, value in zip(
                ("temp_mean", "salt_mean", "EOF_T", "EOF_S", "resT_std", "resS_std"),
                (fitted[0], fitted[1], fitted[2], fitted[3], fitted[6], fitted[7]),
            ):
                model[key] = value
            model["selected"] = fit_layout(model, candidates)
            case = f"modes_{rank:02d}"
            register(case, model, 1.0)
        rank_rows.append([rank, actual_t, actual_s, "evaluated"])
        experiment("modes", rank, case, reference_noise)
    for multiplier in SALT_MULTIPLIERS:
        case = "baseline" if multiplier == 1 else "weight_" + str(multiplier).replace(".", "p")
        if case != "baseline":
            model = dict(baseline)
            model["selected"] = fit_layout(model, candidates, multiplier)
            register(case, model, multiplier)
        experiment("weight", multiplier, case, reference_noise)
    for level in NOISE_LEVELS:
        experiment("noise", level, "baseline", level)
    for case, info in manifest["cases"].items():
        model = read_model(case_path(root, case, mode))
        for level in info["levels"]:
            sigma_t = model["sigmaT_space"] * levels[level][0] / float(model["sigma_base_T"])
            sigma_s = model["sigmaS_space"] * levels[level][1] / float(model["sigma_base_S"])
            result = core.reconstruct_and_score(
                validation["temp_snap"],
                validation["salt_snap"],
                model["EOF_T"],
                model["temp_mean"],
                model["EOF_S"],
                model["salt_mean"],
                model["selected"],
                float(model["ridge"]),
                sigmaT_space=sigma_t,
                sigmaS_space=sigma_s,
                resT_std=model["resT_std"],
                resS_std=model["resS_std"],
                add_obs_noise=True,
                use_wls=core.NOISE_USE_WLS,
                noise_seed=core.NOISE_SEED + 1000000,
                paired_noise=True,
            )
            score = core.combined_reconstruction_score(
                validation["temp_snap"],
                validation["salt_snap"],
                result[2],
                result[3],
                result[0],
                result[1],
                ref,
                wt=float(model["wt"]),
                ws=float(model["ws"]),
                objective=str(model["objective"]),
            )
            validation_rows.append(
                [case, level, len(model["selected"]), result[0], result[1], score]
            )
    save_csv(
        root / "validation_scores.csv",
        ["case", "noise_level", "k", "rmse_T", "rmse_S", "common_objective_score"],
        validation_rows,
    )
    save_csv(
        root / "rank_availability.csv",
        ["requested_rank", "actual_rank_T", "actual_rank_S", "status"],
        rank_rows,
    )
    save_csv(
        root / "selected_sensors.csv",
        ["case", "order", "space_idx", "lon", "lat", "z_rep", "k", "iy", "ix"],
        location_rows,
    )
    save_csv(
        root / "layout_overlap.csv",
        ["case", "sensor_count", "shared_locations", "fraction_shared", "jaccard"],
        overlap_rows,
    )
    (root / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
