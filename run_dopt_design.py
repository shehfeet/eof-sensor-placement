import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "dopt_design"
WORK = OUT / ".work"
CACHE = WORK
RANKS = list(range(5, 16))
RIDGES = [1e-4, 1e-3, 1e-2]
MAX_SENSORS = 21
REALIZATIONS = 100
BATCH = 10
TOLERANCE = 0.05
CANDIDATES = "voxel"
REGION_DEGREES = 0.02


def initialize(threads):
    global np, pd, core, limits
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(threads)
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    import eof_core as core

    limits = threadpool_limits(limits=threads)
    core.DATA_TOPO = str(ROOT / "Data" / "topo.nc")
    core.DATA_3D = str(ROOT / "Data" / "3D_mean_2016-03.nc")
    core.TIME_START, core.TIME_END = "2016-03-01", "2016-03-02"
    core.VALIDATION_DAY, core.TEST_DAY = "2016-03-03", "2016-03-04"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def array(name):
    return np.load(CACHE / (name + ".npy"), mmap_mode="r")


def metadata():
    return json.loads((CACHE / "metadata.json").read_text())


def prepare():
    if (CACHE / "metadata.json").exists():
        return
    data = core.load_and_prepare(
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
    candidates = core.make_candidates(
        data,
        core.RANDOM_SEED,
        CANDIDATES,
        core.N_OPT_POINTS,
        core.VOXEL_DX_M,
        core.VOXEL_DY_M,
        core.VOXEL_DZ_M,
        core.MAX_CANDIDATES,
    )
    if len(candidates) < MAX_SENSORS:
        raise ValueError("The candidate set is smaller than the requested sensor budget.")
    values = {k: data[k] for k in ("k_sp", "iy_sp", "ix_sp", "lon_sp", "lat_sp", "z_rep")}
    values["candidates"] = candidates
    values["sigma_T"], values["sigma_S"] = core.build_sigma_space(data)
    reference = core.compute_reference_scales(
        data["temp_snap"], data["salt_snap"], wt=core.WT, ws=core.WS
    )
    spectra = []
    available = []
    for variable, key in [("T", "temp_snap"), ("S", "salt_snap")]:
        mean = data[key].mean(0)
        anomalies = data[key] - mean
        basis, singular = core.compute_eofs_timecov(anomalies, max(RANKS))
        available.append(basis.shape[1])
        values["mean_" + variable] = mean
        values["basis_" + variable] = basis
        fractions = singular**2 / np.sum(anomalies**2)
        spectra.extend(
            dict(variable=variable, mode=i + 1, fraction=float(f), cumulative=float(c))
            for i, (f, c) in enumerate(zip(fractions, np.cumsum(fractions)))
        )
        for rank in RANKS:
            if rank <= basis.shape[1]:
                values[f"residual_{variable}_{rank}"] = core.truncation_residual_std(
                    anomalies, basis[:, :rank]
                )
    for key, value in values.items():
        np.save(CACHE / (key + ".npy"), value)
    validation = core.load_heldout(data, core.VALIDATION_DAY)
    for variable, key in [("T", "temp_snap"), ("S", "salt_snap")]:
        np.save(CACHE / ("validation_" + variable + ".npy"), validation[key])
    write_csv(OUT / "training_variance.csv", spectra)
    write_csv(
        OUT / "rank_eligibility.csv", [dict(rank=r, eligible=r <= min(available)) for r in RANKS]
    )
    if not any(r <= min(available) for r in RANKS):
        raise ValueError("No requested EOF count is supported by the training data.")
    write_json(
        CACHE / "metadata.json",
        dict(
            reference=reference,
            ranks=[r for r in RANKS if r <= min(available)],
            training_hours=len(data["time"]),
            validation_hours=len(validation["time"]),
            training_start=core.TIME_START,
            training_end=core.TIME_END,
            validation_day=core.VALIDATION_DAY,
            test_day=core.TEST_DAY,
        ),
    )


def reconstruct(period, rank, ridge, selected, level, seed):
    base_t, base_s = core._resolve_sigma_levels()
    return core.reconstruct_and_score(
        array(period + "_T"),
        array(period + "_S"),
        array("basis_T")[:, :rank],
        array("mean_T"),
        array("basis_S")[:, :rank],
        array("mean_S"),
        selected,
        ridge,
        sigmaT_space=array("sigma_T") * core.SIGMA_LEVELS_TEMP[level] / base_t,
        sigmaS_space=array("sigma_S") * core.SIGMA_LEVELS_SALT[level] / base_s,
        resT_std=array(f"residual_T_{rank}"),
        resS_std=array(f"residual_S_{rank}"),
        add_obs_noise=True,
        use_wls=core.NOISE_USE_WLS,
        noise_seed=seed,
        paired_noise=True,
    )


def score(period, result):
    ref = metadata()["reference"]
    return float(
        core.combined_reconstruction_score(
            array(period + "_T"),
            array(period + "_S"),
            result[2],
            result[3],
            result[0],
            result[1],
            ref,
            wt=core.WT,
            ws=core.WS,
            objective=core.COMBINED_OBJECTIVE,
        )
    )


def search_case(rank, ridge):
    path = WORK / f"search_r{rank:02d}_lambda{ridge:g}.json"
    if path.exists():
        return path.name + " reused"
    started = time.monotonic()
    wt, ws = core.effective_information_weights(core.WT, core.WS, metadata()["reference"])
    selected = core.select_sensors_dopt(
        array("candidates"),
        MAX_SENSORS,
        array("basis_T")[:, :rank],
        array("basis_S")[:, :rank],
        wt=wt,
        ws=ws,
        ridge=ridge,
        sigmaT_space=array("sigma_T"),
        sigmaS_space=array("sigma_S"),
        noise_aware=core.NOISE_AWARE_SELECTION,
    )
    result = reconstruct(
        "validation", rank, ridge, selected, core.SIGMA_LEVEL, core.NOISE_SEED + 1000000
    )
    write_json(
        path,
        dict(
            rank=rank,
            ridge=ridge,
            validation_score=score("validation", result),
            selected=selected.tolist(),
            seconds=time.monotonic() - started,
        ),
    )
    return path.name + " completed"


def choose():
    rows = [
        json.loads((WORK / f"search_r{r:02d}_lambda{v:g}.json").read_text())
        for r in metadata()["ranks"]
        for v in RIDGES
    ]
    best = min(rows, key=lambda row: (row["validation_score"], row["rank"], row["ridge"]))
    write_csv(
        OUT / "configuration_search.csv",
        [{k: v for k, v in row.items() if k != "selected"} for row in rows],
    )
    curves = []
    for count in range(1, MAX_SENSORS + 1):
        path = WORK / f"budget_{count:02d}.json"
        if not path.exists():
            result = reconstruct(
                "validation",
                best["rank"],
                best["ridge"],
                best["selected"][:count],
                core.SIGMA_LEVEL,
                core.NOISE_SEED + 1000000,
            )
            write_json(
                path,
                dict(
                    sensors=count,
                    validation_score=score("validation", result),
                    rmse_T=float(result[0]),
                    rmse_S=float(result[1]),
                ),
            )
        curves.append(json.loads(path.read_text()))
    minimum = min(row["validation_score"] for row in curves)
    tolerances = [
        dict(
            tolerance=t,
            sensors=min(
                row["sensors"] for row in curves if row["validation_score"] <= (1 + t) * minimum
            ),
        )
        for t in (0.02, TOLERANCE, 0.10)
    ]
    best["sensor_count"] = next(
        row["sensors"] for row in tolerances if row["tolerance"] == TOLERANCE
    )
    best["tolerance"] = TOLERANCE
    write_json(OUT / "selected_design.json", best)
    write_csv(OUT / "sensor_budget.csv", curves)
    write_csv(OUT / "budget_tolerances.csv", tolerances)
    write_csv(
        OUT / "sensor_locations.csv",
        [
            dict(
                order=i + 1,
                selected_budget=i < best["sensor_count"],
                space_index=int(cell),
                longitude=float(array("lon_sp")[cell]),
                latitude=float(array("lat_sp")[cell]),
                elevation=float(array("z_rep")[cell]),
                layer=int(array("k_sp")[cell]),
                y=int(array("iy_sp")[cell]),
                x=int(array("ix_sp")[cell]),
            )
            for i, cell in enumerate(best["selected"])
        ],
    )


def prepare_test():
    if not (CACHE / "test_ready.json").exists():
        test = core.load_heldout({k: array(k) for k in ("k_sp", "iy_sp", "ix_sp")}, core.TEST_DAY)
        for variable, key in [("T", "temp_snap"), ("S", "salt_snap")]:
            np.save(CACHE / ("test_" + variable + ".npy"), test[key])
        write_json(
            CACHE / "test_ready.json",
            dict(
                hours=len(test["time"]),
                first=str(test["time"][0]),
                last=str(test["time"][-1]),
                times=[str(t) for t in test["time"]],
            ),
        )
    best = json.loads((OUT / "selected_design.json").read_text())
    ridge_rows = []
    for ridge in RIDGES:
        path = WORK / f"ridge_{ridge:g}.json"
        if not path.exists():
            result = reconstruct(
                "test",
                best["rank"],
                ridge,
                best["selected"][: best["sensor_count"]],
                core.SIGMA_LEVEL,
                core.NOISE_SEED + 2000000,
            )
            write_json(
                path,
                dict(
                    ridge=ridge,
                    sensors=best["sensor_count"],
                    noise=core.SIGMA_LEVEL,
                    score=score("test", result),
                    rmse_T=float(result[0]),
                    rmse_S=float(result[1]),
                ),
            )
        ridge_rows.append(json.loads(path.read_text()))
    write_csv(OUT / "ridge_sensitivity.csv", ridge_rows)
    calibrations = []
    for count in sorted(set([best["sensor_count"], MAX_SENSORS])):
        for level in core.MC_NOISE_LEVELS:
            path = WORK / f"calibration_k{count:02d}_{level}.json"
            if not path.exists():
                result = reconstruct(
                    "validation",
                    best["rank"],
                    best["ridge"],
                    best["selected"][:count],
                    level,
                    core.NOISE_SEED + 1000000,
                )
                factors = [
                    float(
                        np.nanquantile(
                            np.abs(array("validation_" + v) - result[2 + j])
                            / np.maximum(result[6 + j], 1e-12),
                            0.95,
                        )
                    )
                    for j, v in enumerate(("T", "S"))
                ]
                checks = []
                for j, variable in enumerate(("T", "S")):
                    truth = array("validation_" + variable)
                    valid = np.isfinite(truth)
                    band = factors[j] * result[6 + j]
                    if not np.isfinite(result[2 + j][valid]).all() or not np.isfinite(band).all():
                        raise ValueError(
                            "Non-finite reconstruction or interval on valid validation cells."
                        )
                    checks.append(
                        dict(
                            sensors=count,
                            noise=level,
                            variable=variable,
                            multiplier=factors[j],
                            validation_coverage=float(
                                np.mean((np.abs(truth - result[2 + j]) <= band)[valid])
                            ),
                            mean_width=float(2 * np.sum(np.where(valid, band, 0)) / valid.sum()),
                        )
                    )
                write_json(path, dict(factors=factors, checks=checks))
            calibrations.extend(json.loads(path.read_text())["checks"])
    write_csv(OUT / "interval_calibration.csv", calibrations)
    prepare_groups()


def prepare_groups():
    if (WORK / "diagnostic_groups.csv").exists():
        return
    coordinates = np.column_stack(
        (np.floor(array("lon_sp") / REGION_DEGREES), np.floor(array("lat_sp") / REGION_DEGREES))
    ).astype(np.int64)
    regions, region_ids = np.unique(coordinates, axis=0, return_inverse=True)
    elevation_ids = np.searchsorted([-20, -10, -5, 0], array("z_rep"), side="left")
    np.save(WORK / "region_ids.npy", region_ids.astype(np.int32))
    np.save(WORK / "elevation_ids.npy", elevation_ids.astype(np.int32))
    groups = [
        dict(kind="elevation", group=i, label=label)
        for i, label in enumerate(
            ["z <= -20 m", "-20 < z <= -10 m", "-10 < z <= -5 m", "-5 < z <= 0 m", "z > 0 m"]
        )
    ]
    groups += [
        dict(
            kind="region",
            group=i,
            label=f"{i+1}",
            lon_min=float(x * REGION_DEGREES),
            lon_max=float((x + 1) * REGION_DEGREES),
            lat_min=float(y * REGION_DEGREES),
            lat_max=float((y + 1) * REGION_DEGREES),
        )
        for i, (x, y) in enumerate(regions)
    ]
    times = json.loads((CACHE / "test_ready.json").read_text())["times"]
    groups += [dict(kind="hour", group=i, label=t) for i, t in enumerate(times)]
    write_csv(WORK / "diagnostic_groups.csv", groups)


def diagnostic_sums(truth, estimate, band, region_ids, elevation_ids, nregions):
    totals = {
        "hour": np.zeros((truth.shape[0], 8)),
        "elevation": np.zeros((5, 8)),
        "region": np.zeros((nregions, 8)),
        "cell": np.zeros((truth.shape[1], 2), dtype=np.uint32),
    }
    for hour in range(truth.shape[0]):
        valid = np.isfinite(truth[hour])
        error = estimate[hour, valid] - truth[hour, valid]
        width = band[valid]
        if not np.isfinite(error).all() or not np.isfinite(width).all() or np.any(width < 0):
            raise ValueError("Invalid test reconstruction or uncertainty interval.")
        values = (
            np.ones(len(error)),
            (np.abs(error) <= width).astype(float),
            error,
            np.abs(error),
            error**2,
            2 * width,
            (error < -width).astype(float),
            (error > width).astype(float),
        )
        totals["cell"][valid, 0] += 1
        totals["cell"][valid, 1] += (np.abs(error) <= width).astype(np.uint32)
        totals["hour"][hour] = [v.sum() for v in values]
        for kind, ids in [("region", region_ids), ("elevation", elevation_ids)]:
            selected_ids = ids[valid]
            for column, value in enumerate(values):
                totals[kind][:, column] += np.bincount(
                    selected_ids, weights=value, minlength=len(totals[kind])
                )
    return totals


def monte_carlo(count, level, start):
    path = WORK / f"mc_k{count:02d}_{level}_{start:03d}.csv"
    if path.with_suffix(".json").exists():
        return path.name + " reused"
    best = json.loads((OUT / "selected_design.json").read_text())
    factors = json.loads((WORK / f"calibration_k{count:02d}_{level}.json").read_text())["factors"]
    rows = []
    diagnose = count == best["sensor_count"] and level == core.SIGMA_LEVEL
    diagnostic = {}
    if diagnose:
        region_ids, elevation_ids = array("region_ids"), array("elevation_ids")
        nregions = int(region_ids.max()) + 1
    started = time.monotonic()
    for realization in range(start, min(start + BATCH, REALIZATIONS)):
        seed = core.NOISE_SEED + 2000000 + realization
        result = reconstruct(
            "test", best["rank"], best["ridge"], best["selected"][:count], level, seed
        )
        row = dict(
            sensors=count,
            noise=level,
            realization=realization + 1,
            seed=seed,
            score=score("test", result),
            rmse_T=float(result[0]),
            rmse_S=float(result[1]),
        )
        for j, variable in enumerate(("T", "S")):
            truth = array("test_" + variable)
            valid = np.isfinite(truth)
            band = factors[j] * result[6 + j]
            row["coverage_" + variable] = float(
                np.mean((np.abs(truth - result[2 + j]) <= band)[valid])
            )
            row["width_" + variable] = float(2 * np.sum(np.where(valid, band, 0)) / valid.sum())
            if diagnose:
                sums = diagnostic_sums(
                    truth, result[2 + j], band, region_ids, elevation_ids, nregions
                )
                for kind, values in sums.items():
                    key = variable + "_" + kind
                    if key not in diagnostic:
                        diagnostic[key] = values
                    else:
                        diagnostic[key] += values
        rows.append(row)
    if diagnose:
        np.savez_compressed(WORK / f"diagnostics_{start:03d}.npz", **diagnostic)
    write_csv(path, rows)
    write_json(
        path.with_suffix(".json"), dict(seconds=time.monotonic() - started, realizations=len(rows))
    )
    return path.name + " completed"


def dispatch(task):
    return search_case(*task[1:]) if task[0] == "search" else monte_carlo(*task[1:])


def parallel(tasks, args):
    tasks = tasks[args.shard_index :: args.shards]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(args.cores // args.workers,),
    ) as pool:
        for future in as_completed([pool.submit(dispatch, task) for task in tasks]):
            print(future.result(), flush=True)


def cleanup():
    if WORK.resolve().parent != OUT.resolve() or WORK.name != ".work":
        raise ValueError("Unexpected temporary directory.")
    if WORK.exists():
        shutil.rmtree(WORK)
    for name in ("rank_eligibility.csv", "budget_tolerances.csv", "mc_convergence.csv"):
        (OUT / name).unlink(missing_ok=True)


def report():
    best = json.loads((OUT / "selected_design.json").read_text())
    paths = [
        WORK / f"mc_k{k:02d}_{level}_{start:03d}.csv"
        for k in sorted(set([best["sensor_count"], MAX_SENSORS]))
        for level in core.MC_NOISE_LEVELS
        for start in range(0, REALIZATIONS, BATCH)
    ]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    write_csv(OUT / "mc_realizations.csv", frame)
    metrics = ["score", "rmse_T", "rmse_S", "coverage_T", "coverage_S", "width_T", "width_S"]
    rows = []
    for (count, level), group in frame.groupby(["sensors", "noise"]):
        row = dict(sensors=int(count), noise=level, realizations=len(group))
        for metric in metrics:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std()
            row[metric + "_mcse"] = group[metric].std() / np.sqrt(len(group))
        rows.append(row)
    summary = pd.DataFrame(rows)
    write_csv(OUT / "mc_summary.csv", summary)
    convergence = []
    for (count, level), group in frame.groupby(["sensors", "noise"]):
        group = group.sort_values("realization")
        for n in sorted(set([min(v, len(group)) for v in (10, 20, 50, 100)])):
            convergence.append(
                dict(
                    sensors=count,
                    noise=level,
                    repetitions=n,
                    mean_score=group.score.iloc[:n].mean(),
                    mcse=group.score.iloc[:n].std() / np.sqrt(n),
                )
            )
    write_csv(OUT / "mc_convergence.csv", convergence)
    report_diagnostics(best, summary)
    details = metadata()
    protocol = json.loads((OUT / "protocol.json").read_text())
    protocol.update(
        reference_scales=details["reference"],
        eligible_ranks=details["ranks"],
        training_hours=details["training_hours"],
        validation_hours=details["validation_hours"],
        rank_eligibility=pd.read_csv(OUT / "rank_eligibility.csv").to_dict("records"),
        budget_tolerances=pd.read_csv(OUT / "budget_tolerances.csv").to_dict("records"),
        mc_convergence=pd.read_csv(OUT / "mc_convergence.csv").to_dict("records"),
        uncertainty_settings={key: getattr(core, key) for key in core.MC_CONFIG_KEYS},
        temperature_priority=core.WT,
        salinity_priority=core.WS,
        noise_temperature=core.SIGMA_LEVELS_TEMP,
        noise_salinity=core.SIGMA_LEVELS_SALT,
        test=json.loads((WORK / "test_ready.json").read_text()),
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "pandas",
                "xarray",
                "netCDF4",
                "matplotlib",
                "threadpoolctl",
                "geopandas",
            )
        },
    )
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "complete.json",
        dict(complete=True, message="Reference-design results and diagnostic totals saved."),
    )
    cleanup()


def report_diagnostics(best, summary):
    totals = {}
    for start in range(0, REALIZATIONS, BATCH):
        with np.load(WORK / f"diagnostics_{start:03d}.npz") as batch:
            for key in batch.files:
                if key not in totals:
                    totals[key] = batch[key].copy()
                else:
                    totals[key] += batch[key]
    groups = pd.read_csv(WORK / "diagnostic_groups.csv")
    records = []
    selected = best["selected"][: best["sensor_count"]]
    baseline = summary[
        (summary.sensors == best["sensor_count"]) & (summary.noise == core.SIGMA_LEVEL)
    ].iloc[0]
    for variable in ("T", "S"):
        overall = totals[variable + "_hour"].sum(0)
        if not np.isclose(
            overall[1] / overall[0],
            baseline["coverage_" + variable + "_mean"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Diagnostic coverage differs from the Monte Carlo summary.")
        for kind in ("hour", "elevation", "region"):
            values = totals[variable + "_" + kind]
            if not np.allclose(values.sum(0), overall, rtol=1e-10, atol=1e-8):
                raise ValueError("Diagnostic groups do not reproduce the overall totals.")
            for item in groups[groups.kind == kind].to_dict("records"):
                index = int(item["group"])
                n, covered, signed, absolute, squared, width, above, below = values[index]
                if n == 0:
                    continue
                item = {key: value for key, value in item.items() if pd.notna(value)}
                item.update(
                    variable=variable,
                    sensors=best["sensor_count"],
                    noise=core.SIGMA_LEVEL,
                    realizations=REALIZATIONS,
                    valid_pairs_per_realization=int(round(n / REALIZATIONS)),
                    coverage=covered / n,
                    reference_above_interval=above / n,
                    reference_below_interval=below / n,
                    rmse=float(np.sqrt(squared / n)),
                    mean_absolute_error=absolute / n,
                    reconstruction_bias=signed / n,
                    mean_interval_width=width / n,
                )
                if kind == "hour":
                    item["available_sensors"] = int(
                        np.isfinite(array("test_" + variable)[index, selected]).sum()
                    )
                records.append(item)
    native = {
        key: np.asarray(array(key))
        for key in ("k_sp", "iy_sp", "ix_sp", "lon_sp", "lat_sp", "z_rep")
    }
    for variable in ("T", "S"):
        values = totals[variable + "_cell"]
        overall = totals[variable + "_hour"].sum(0)
        if not np.array_equal(values.sum(0, dtype=np.uint64), overall[:2].astype(np.uint64)):
            raise ValueError("Native-cell counts differ from the pooled coverage totals.")
        if np.any(values[:, 1] > values[:, 0]):
            raise ValueError("Covered counts exceed available reference counts.")
        native["valid_" + variable] = values[:, 0]
        native["covered_" + variable] = values[:, 1]
    np.savez_compressed(OUT / "coverage_3d.npz", **native)
    table = pd.DataFrame(records)
    write_csv(OUT / "coverage_diagnostics.csv", table)
    lines = [
        f"Selected design: {best['rank']} modes per variable, ridge {best['ridge']:g}, {best['sensor_count']} sensors.",
        f"Coverage diagnostics use {REALIZATIONS} noise realizations at the {core.SIGMA_LEVEL} noise level on 4 March.",
        "Intervals use validation-derived multipliers. No test errors change the design or interval widths.",
        "Elevation groups use training-median model-cell elevations, not instantaneous depth.",
        f"Regions are {REGION_DEGREES:g}-degree longitude/latitude boxes, pooled over model layers.",
        "Locations and hours below describe observed undercoverage; they do not establish its cause.",
        "",
    ]
    for variable in ("T", "S"):
        data = table[table.variable == variable]
        hours = data[data.kind == "hour"]
        worst_hour = hours.loc[hours.coverage.idxmin()]
        elevations = data[data.kind == "elevation"]
        worst_elevation = elevations.loc[elevations.coverage.idxmin()]
        regions = data[data.kind == "region"]
        minimum_pairs = 0.005 * hours.valid_pairs_per_realization.sum()
        eligible = regions[regions.valid_pairs_per_realization >= minimum_pairs]
        if eligible.empty:
            eligible = regions.nlargest(20, "valid_pairs_per_realization")
            qualifier = "among the 20 most-sampled regions"
        else:
            qualifier = "among regions containing at least 0.5% of valid time-cell pairs"
        region = eligible.loc[eligible.coverage.idxmin()]
        lines += [
            f"{variable}: overall coverage {100*baseline['coverage_'+variable+'_mean']:.2f}% (nominal target 95%).",
            f"Lowest hourly coverage: {worst_hour.label}, {100*worst_hour.coverage:.2f}%; available sensors {int(worst_hour.available_sensors)}.",
            f"Lowest elevation-band coverage: {worst_elevation.label}, {100*worst_elevation.coverage:.2f}%.",
            f"Lowest regional coverage {qualifier}: longitude {region.lon_min:.3f}–{region.lon_max:.3f}, latitude {region.lat_min:.3f}–{region.lat_max:.3f}, {100*region.coverage:.2f}%.",
            f"Regional RMSE {region.rmse:.4g}; mean full interval width {region.mean_interval_width:.4g} ({'degrees C' if variable == 'T' else 'g/kg'}).",
            "",
        ]
    lines += [
        "Checks passed: native-cell counts and grouped coverage match the Monte Carlo summary; all groups reproduce pooled totals.",
        "Coverage is a fraction of valid time-cell pairs, averaged over noise, not a fraction of successful runs.",
        "These spatially and temporally dependent pairs are not independent trials. No binomial confidence interval is inferred.",
        "Near-95% validation coverage checks calibration on that day, not independent reliability.",
    ]
    lines += [
        "Native-cell coverage is saved for every training-grid cell, with valid and covered counts.",
        "Figure 8 maps show the uppermost model layer. Hour and elevation panels summarize all eligible layers.",
    ]
    (OUT / "results_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["all", "prepare", "search", "select", "mc", "report"], default="all"
    )
    parser.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 60)))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= args.cores or not 0 <= args.shard_index < args.shards:
        parser.error("Check workers, cores and shard index.")
    if args.cores > int(os.environ.get("SLURM_CPUS_PER_TASK", args.cores)):
        parser.error("Requested cores exceed the Slurm allocation.")
    if args.shards > 1 and args.stage not in ("search", "mc"):
        parser.error("Sharding is supported for search and mc only; prepare first.")
    initialize(args.cores // args.workers)
    CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in [Path(__file__), ROOT / "eof_core.py"]:
        digest.update(path.read_bytes())
    for path in [Path(core.DATA_TOPO), Path(core.DATA_3D)]:
        digest.update(str((path.name, path.stat().st_size, path.stat().st_mtime_ns)).encode())
    fingerprint = digest.hexdigest()
    protocol = OUT / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text())["fingerprint"] != fingerprint:
        raise ValueError(
            "Code or inputs changed. Use a separate output directory for a new experiment."
        )
    if (OUT / "complete.json").exists():
        cleanup()
        print(
            "D-opt reference run is complete.",
            flush=True,
        )
        return
    if not protocol.exists():
        if args.stage not in ("all", "prepare"):
            raise ValueError("Run --stage prepare first.")
        write_json(
            protocol,
            dict(
                fingerprint=fingerprint,
                ranks=RANKS,
                ridges=RIDGES,
                maximum_sensors=MAX_SENSORS,
                realizations=REALIZATIONS,
                tolerance=TOLERANCE,
                candidates=CANDIDATES,
                noise_levels=list(core.MC_NOISE_LEVELS),
                baseline_noise=core.SIGMA_LEVEL,
                cores=args.cores,
                workers=args.workers,
                threads_per_worker=args.cores // args.workers,
                diagnostic_region_degrees=REGION_DEGREES,
                diagnostic_network="validation-selected budget",
                validation_seed=core.NOISE_SEED + 1000000,
                test_seed_start=core.NOISE_SEED + 2000000,
                training="1–2 March 2016",
                validation="3 March 2016",
                test="4 March 2016",
            ),
        )
    stages = (
        ["prepare", "search", "select", "mc", "report"] if args.stage == "all" else [args.stage]
    )
    for stage in stages:
        print(stage + " started", flush=True)
        if stage == "prepare":
            prepare()
        elif stage == "search":
            parallel([("search", r, v) for r in metadata()["ranks"] for v in RIDGES], args)
        elif stage == "select":
            choose()
            prepare_test()
        elif stage == "mc":
            best = json.loads((OUT / "selected_design.json").read_text())
            parallel(
                [
                    ("mc", k, level, start)
                    for k in sorted(set([best["sensor_count"], MAX_SENSORS]))
                    for level in core.MC_NOISE_LEVELS
                    for start in range(0, REALIZATIONS, BATCH)
                ],
                args,
            )
        else:
            report()
        print(stage + " finished", flush=True)


if __name__ == "__main__":
    main()
