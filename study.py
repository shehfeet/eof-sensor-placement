import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import settings as cfg


def initialize(threads):
    global np, pd, core, flow, limits
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = str(threads)
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    import eof_core as core
    import workflows as flow

    limits = threadpool_limits(limits=threads)


def csv(path, rows):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def stamp(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def fit(group, label, method="dopt", **kwargs):
    path = cfg.model(group, label)
    if path.exists():
        core.validate_model_split(flow.read_model(path))
        return
    started = time.monotonic()
    core.run_one_method(method, cfg.CANDIDATES, **kwargs)
    model = flow.read_model(path)
    if model["EOF_T"].shape[1] != cfg.RANK or model["EOF_S"].shape[1] != cfg.RANK:
        path.unlink()
        raise ValueError("The training numerical rank does not support the shared EOF count.")
    stamp(
        Path(core.OUT_ROOT) / f"{label}__selection_time.json",
        dict(seconds=time.monotonic() - started),
    )


def evaluate(
    group,
    label,
    method,
    selected=None,
    levels=None,
    ridge=None,
    baseline_calibration=None,
    model_label=None,
):
    work = cfg.OUTPUT / group / ".work"
    model = flow.read_model(cfg.model(group, model_label or label))
    core.validate_model_split(model)
    if selected is not None:
        model["selected"] = selected
    if ridge is not None:
        model["ridge"] = ridge
    grid = {key: model[key] for key in ("k_sp", "iy_sp", "ix_sp")}
    validation = core.load_heldout(
        grid, str(model["validation_day"]), str(model["validation_data_file"])
    )
    test = core.load_heldout(grid, str(model["test_day"]), str(model["test_data_file"]))
    reference = dict(zip(model["ref_keys"].tolist(), model["ref_values"].tolist()))
    levels = levels or core.MC_NOISE_LEVELS
    for level in levels:
        sigma_t = (
            model["sigmaT_space"] * core.SIGMA_LEVELS_TEMP[level] / float(model["sigma_base_T"])
        )
        sigma_s = (
            model["sigmaS_space"] * core.SIGMA_LEVELS_SALT[level] / float(model["sigma_base_S"])
        )
        kwargs = dict(
            sigmaT_space=sigma_t,
            sigmaS_space=sigma_s,
            resT_std=model["resT_std"],
            resS_std=model["resS_std"],
            add_obs_noise=True,
            use_wls=core.NOISE_USE_WLS,
            paired_noise=True,
        )
        for k in range(1, len(model["selected"]) + 1):
            path = work / f"{label}__{level}_k{k:02d}.csv"
            if path.exists():
                continue

            def reconstruct(data, seed):
                return core.reconstruct_and_score(
                    data["temp_snap"],
                    data["salt_snap"],
                    model["EOF_T"],
                    model["temp_mean"],
                    model["EOF_S"],
                    model["salt_mean"],
                    model["selected"][:k],
                    float(model["ridge"]),
                    noise_seed=seed,
                    **kwargs,
                )

            if method == "random":
                factors = [float("nan"), float("nan")]
            elif baseline_calibration is None:
                vr = reconstruct(validation, core.NOISE_SEED + 1000000)
                factors = [
                    float(
                        np.nanquantile(
                            np.abs(validation[key] - vr[2 + j]) / np.maximum(vr[6 + j], 1e-12), 0.95
                        )
                    )
                    for j, key in enumerate(("temp_snap", "salt_snap"))
                ]
                del vr
            else:
                calibration = pd.read_csv(
                    work / f"{baseline_calibration}__{level}_k{k:02d}.csv"
                ).iloc[0]
                factors = [calibration["q_T"], calibration["q_S"]]
            result = reconstruct(test, core.NOISE_SEED + 2000000)
            rt, rs, th, sh, _, _, st, ss = result
            row = dict(
                case=label,
                method=method,
                noise=level,
                k=k,
                rank_T=model["EOF_T"].shape[1],
                rank_S=model["EOF_S"].shape[1],
                ridge=float(model["ridge"]),
                rmse_T=float(rt),
                rmse_S=float(rs),
                score=float(
                    core.combined_reconstruction_score(
                        test["temp_snap"],
                        test["salt_snap"],
                        th,
                        sh,
                        rt,
                        rs,
                        reference,
                        wt=float(model["wt"]),
                        ws=float(model["ws"]),
                        objective=str(model["objective"]),
                    )
                ),
                q_T=factors[0],
                q_S=factors[1],
            )
            for j, (variable, key, estimate, std) in enumerate(
                zip(("T", "S"), ("temp_snap", "salt_snap"), (th, sh), (st, ss))
            ):
                truth = test[key]
                valid = np.isfinite(truth)
                band = factors[j] * std
                metrics = core.compute_metrics_global(truth, estimate)
                row.update(
                    {
                        name + "_" + variable: float(metrics[name])
                        for name in ("corr_r", "var_ratio", "nrmse", "mae", "bias")
                    }
                )
                row["coverage_" + variable] = (
                    float("nan")
                    if method == "random"
                    else float(np.mean((np.abs(truth - estimate) <= band)[valid]))
                )
                row["width_" + variable] = float(2 * np.sum(np.where(valid, band, 0)) / valid.sum())
            if method != "random" and level == "low" and (k == cfg.BUDGET):
                traces = []
                for j, cell in enumerate(model["selected"][:k]):
                    for t, date in enumerate(test["time"]):
                        traces.append(
                            dict(
                                case=label,
                                order=j + 1,
                                time=str(date),
                                reference_T=float(test["temp_snap"][t, cell]),
                                reference_S=float(test["salt_snap"][t, cell]),
                                reconstruction_T=float(th[t, cell]),
                                reconstruction_S=float(sh[t, cell]),
                                observation_T=float(result[4][t, j]),
                                observation_S=float(result[5][t, j]),
                                halfwidth_T=float(factors[0] * st[cell]),
                                halfwidth_S=float(factors[1] * ss[cell]),
                            )
                        )
                csv(cfg.OUTPUT / group / f"{label}_sensor_timeseries.csv", traces)
                if group == "methods" and method in ("dopt", "hybrid"):
                    spatial = {
                        key: model[key]
                        for key in ("k_sp", "iy_sp", "ix_sp", "lon_sp", "lat_sp", "z_rep")
                    }
                    for variable, key, estimate, std, factor in zip(
                        ("T", "S"), ("temp_snap", "salt_snap"), (th, sh), (st, ss), factors
                    ):
                        spatial["rmse_" + variable] = np.sqrt(
                            np.nanmean((test[key] - estimate) ** 2, axis=0)
                        )
                        spatial["halfwidth_" + variable] = factor * std
                    np.savez_compressed(
                        cfg.OUTPUT / group / f"{label}_spatial_summary.npz", **spatial
                    )
            csv(path, [row])
    import xarray as xr

    with xr.open_dataset(core.DATA_TOPO) as topo:
        lon, lat = (topo.lonc.values, topo.latc.values)
    selected = np.asarray(model["selected"])
    with xr.open_dataset(str(model["validation_data_file"])) as ds:
        dims = ds.hn.dims
        y = xr.DataArray(model["iy_sp"][selected], dims="points")
        x = xr.DataArray(model["ix_sp"][selected], dims="points")
        subset = ds.sel(time=slice(str(model["training_start"]), str(model["training_end"])))
        thickness = (
            subset.hn.isel({dims[2]: y, dims[3]: x}).transpose(dims[0], dims[1], "points").values
        )
        bath = subset.bathymetry.isel({dims[2]: y, dims[3]: x}).values
        centres = np.cumsum(thickness, axis=1) - 0.5 * thickness - bath
        elevations = np.nanmedian(
            centres[:, model["k_sp"][selected], np.arange(len(selected))], axis=0
        )
    coordinates = []
    for order, cell in enumerate(model["selected"], 1):
        coordinates.append(
            dict(
                case=label,
                order=order,
                selected_budget=order <= cfg.BUDGET,
                layer=int(model["k_sp"][cell]),
                y=int(model["iy_sp"][cell]),
                x=int(model["ix_sp"][cell]),
                longitude=float(lon[model["iy_sp"][cell], model["ix_sp"][cell]]),
                latitude=float(lat[model["iy_sp"][cell], model["ix_sp"][cell]]),
                training_median_elevation=float(elevations[order - 1]),
            )
        )
    csv(work / f"{label}__grid_locations.csv", coordinates)


def task(job):
    group, label, action, threads = job
    cfg.configure(core, group, label)
    core.N_JOBS = min(4, threads)
    started = time.monotonic()
    if action == "method":
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1 if label in ("rmse", "hybrid", "uopt") else threads):
            fit(group, label, label)
        evaluate(group, label, label)
    elif action == "random":
        candidates = np.load(cfg.OUTPUT / group / ".work/candidate_indices.npy")
        selected = core.select_sensors_random(
            candidates, cfg.SENSORS, core.RANDOM_SEED + 10000 + int(label.split("_")[-1])
        )
        evaluate(group, label, "random", selected=selected, model_label="reference")
    elif action == "adaptive":
        cfg.configure(core, group, label, cfg.MONTHS[label])
        flow.fit_adaptive(label)
        evaluate(group, label + "_adaptive", "dopt")
        if label != "march":
            evaluate(group, label + "_updated_eofs", "dopt")
    elif action == "sensitivity":
        import sensitivity_cases as sensitivity

        for level, (t, s) in sensitivity.NOISE_LEVELS.items():
            core.SIGMA_LEVELS_TEMP[level], core.SIGMA_LEVELS_SALT[level] = (t, s)
        manifest = json.loads((cfg.OUTPUT / group / ".work/experiment.json").read_text())
        if label.startswith("ridge_"):
            evaluate(
                group, label, "dopt", levels=["low"], ridge=float(label[6:]), model_label="baseline"
            )
        else:
            evaluate(group, label, "dopt", levels=manifest["cases"][label]["levels"])
    stamp(
        cfg.OUTPUT / group / ".work" / f"{label}__complete.json",
        dict(seconds=time.monotonic() - started),
    )
    return group + ": " + label + " complete"


def parallel(jobs, args):
    pending = [
        (*job, args.cores // args.workers)
        for job in jobs
        if not (cfg.OUTPUT / job[0] / ".work" / f"{job[1]}__complete.json").exists()
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(args.cores // args.workers,),
    ) as pool:
        for future in as_completed([pool.submit(task, job) for job in pending]):
            print(future.result(), flush=True)


def collect(group):
    root = cfg.OUTPUT / group
    paths = list((root / ".work").glob("*__*_k[0-9][0-9].csv"))
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    csv(root / "reconstruction_scores.csv", frame)
    if group == "random":
        random = frame[frame.method == "random"]
        csv(
            root / "random_summary.csv",
            random.groupby(["noise", "k"])[["rmse_T", "rmse_S", "score"]]
            .agg(["mean", "std"])
            .reset_index()
            .pipe(lambda d: d.set_axis(["_".join(c).rstrip("_") for c in d.columns], axis=1)),
        )
    locations = []
    for path in sorted((root / ".work").glob("*__grid_locations.csv")):
        locations.extend(pd.read_csv(path).to_dict("records"))
    csv(root / "sensor_grid_locations.csv", locations)
    if group == "sensitivity":
        manifest = json.loads((root / ".work/experiment.json").read_text())
        rows = []
        for experiment in manifest["experiments"]:
            subset = frame[
                (frame.case == experiment["case"]) & (frame.noise == experiment["noise_level"])
            ]
            info = manifest["cases"][experiment["case"]]
            rows.extend(
                (
                    dict(
                        row,
                        axis=experiment["axis"],
                        value=experiment["value"],
                        selection_weight_T=info["selection_weight_T"],
                        selection_weight_S=info["selection_weight_S"],
                    )
                    for row in subset.to_dict("records")
                )
            )
        rows.extend(
            (
                dict(row, axis="ridge", value=row["ridge"])
                for row in frame[frame.case.str.startswith("ridge_")].to_dict("records")
            )
        )
        csv(root / "sensitivity_scores.csv", rows)
        for name in (
            "layout_overlap.csv",
            "training_explained_variance.csv",
            "validation_scores.csv",
            "rank_availability.csv",
        ):
            csv(root / name, pd.read_csv(root / ".work" / name))


def main(group):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", cfg.CORES))
    )
    parser.add_argument("--workers", type=int, default=cfg.WORKERS)
    if group == "methods":
        parser.add_argument("--method", choices=cfg.METHODS)
    args = parser.parse_args()
    if not 1 <= args.workers <= args.cores or args.cores > int(
        os.environ.get("SLURM_CPUS_PER_TASK", args.cores)
    ):
        parser.error("Check the CPU allocation and worker count.")
    initialize(args.cores // args.workers)
    root = cfg.OUTPUT / group
    (root / ".work").mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    dependencies = ["eof_core.py", "settings.py", "study.py", "workflows.py"]
    if group == "sensitivity":
        dependencies.append("sensitivity_cases.py")
    for name in dependencies:
        digest.update((Path(__file__).parent / name).read_bytes())
    for path in [cfg.DATA / "topo.nc"] + [
        cfg.DATA / f"3D_mean_{m}.nc"
        for m in (cfg.MONTHS.values() if group == "adaptive" else ["2016-03"])
    ]:
        digest.update(str((path.name, path.stat().st_size, path.stat().st_mtime_ns)).encode())
    fingerprint = digest.hexdigest()
    protocol = root / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Inputs or code changed. Use a new experiment directory.")
    stamp(
        protocol,
        dict(
            fingerprint=fingerprint,
            rank=cfg.RANK,
            ridge=cfg.RIDGE,
            budget=cfg.BUDGET,
            training="days 1-2",
            validation="day 3",
            test="day 4",
            noise_realizations=1,
            random_layouts=cfg.RANDOM_LAYOUTS if group == "random" else 0,
        ),
    )
    cfg.configure(core, group, "baseline")
    if group == "methods":
        methods = [args.method] if args.method else cfg.METHODS
        parallel([(group, method, "method") for method in methods], args)
    elif group == "random":
        flow.prepare_random()
        parallel(
            [(group, f"random_{i:03d}", "random") for i in range(1, cfg.RANDOM_LAYOUTS + 1)], args
        )
    elif group == "adaptive":
        if not (root / ".work/common_grid.npz").exists():
            flow.prepare_adaptive()
        parallel([(group, "march", "adaptive")], args)
        parallel([(group, name, "adaptive") for name in cfg.MONTHS if name != "march"], args)
    else:
        import sensitivity_cases as sensitivity

        fit(group, "baseline")
        if not (root / ".work/experiment.json").exists():
            sensitivity.prepare_cases(cfg.model(group, "baseline"), root / ".work", cfg.CANDIDATES)
        manifest = json.loads((root / ".work/experiment.json").read_text())
        parallel([(group, label, "sensitivity") for label in manifest["cases"]], args)
        parallel(
            [(group, f"ridge_{ridge:g}", "sensitivity") for ridge in (0.0001, 0.001, 0.01)], args
        )
    collect(group)
    print(f"Results: {root}", flush=True)
