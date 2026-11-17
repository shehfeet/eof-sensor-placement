import argparse
import hashlib
import json
import os
import time
from pathlib import Path

METHODS = ("qdeim", "aopt", "dopt", "uopt", "hybrid", "rmse")
NOISE = {"low": (0.01, 0.005), "medium": (0.02, 0.01), "high": (0.05, 0.02)}
SEED = 2001234


def locate():
    root = Path(__file__).resolve().parent
    methods = root / "outputs/methods"
    if not (methods / ".work/dopt__frozen_model.npz").exists():
        raise FileNotFoundError("Run run_methods.py before the noise-weight diagnostic.")
    return root, methods


def prepare(paths, datafile, locations, output, signature):
    import numpy as np
    import pandas as pd
    import xarray as xr

    cache = output / "test_summaries.npz"
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved["signature"]) != signature:
                raise ValueError(
                    "Inputs changed. Keep the existing diagnostic and use a new output folder."
                )
            return {key: saved[key] for key in saved.files}

    models = {}
    for method, path in paths.items():
        with np.load(path) as saved:
            keys = ["k_sp", "iy_sp", "ix_sp", "selected", "ridge", "test_day", "mc_config"]
            keys += [
                "EOF_T",
                "EOF_S",
                "temp_mean",
                "salt_mean",
                "sigmaT_space",
                "sigmaS_space",
                "sigma_base_T",
                "sigma_base_S",
            ]
            models[method] = {key: saved[key] for key in keys}
        config = json.loads(str(models[method]["mc_config"]))
        assert (
            config["NOISE_USE_WLS"]
            and not config["NOISE_ADD_BIAS"]
            and not config["NOISE_ADD_DRIFT"]
        )
    first = models["dopt"]
    for model in models.values():
        assert str(model["test_day"]) == str(first["test_day"])
        assert (
            json.loads(str(model["mc_config"]))["HN_MIN"]
            == json.loads(str(first["mc_config"]))["HN_MIN"]
        )
        for key in ["k_sp", "iy_sp", "ix_sp"]:
            assert np.array_equal(model[key], first[key])
    indices = tuple(first[key] for key in ["k_sp", "iy_sp", "ix_sp"])
    summaries = {"signature": signature}
    for method, model in models.items():
        selected = model["selected"]
        assert len(np.unique(selected)) == len(selected)
        assert method == "reference" or len(selected) == 21
        summaries[method + "_selected"] = selected
        summaries[method + "_ridge"] = model["ridge"]
        for variable, mean, sigma in [
            ("T", "temp_mean", "sigmaT_space"),
            ("S", "salt_mean", "sigmaS_space"),
        ]:
            basis = model["EOF_" + variable].astype("float64")
            model["EOF_" + variable] = basis
            prefix = method + "_" + variable + "_"
            rank = basis.shape[1]
            summaries[prefix + "gram"] = np.empty((24, rank, rank))
            summaries[prefix + "cross"] = np.empty((24, rank))
            summaries[prefix + "energy"] = np.empty(24)
            summaries[prefix + "observations"] = np.empty((24, len(selected)))
            summaries[prefix + "basis"] = basis[selected]
            summaries[prefix + "mean"] = model[mean][selected]
            summaries[prefix + "sigma"] = model[sigma][selected]
            summaries[prefix + "sigma_base"] = model["sigma_base_" + variable]
            model["gram_" + variable] = basis.T @ basis
    random_model = models.get("reference", first)
    random_locations = pd.read_csv(locations)
    random_locations = random_locations[random_locations.case.str.startswith("random_")]
    shape = tuple(int(first[key].max()) + 1 for key in ["k_sp", "iy_sp", "ix_sp"])
    grid_ids = np.ravel_multi_index(indices, shape)
    assert np.all(np.diff(grid_ids) > 0)
    random_selections = {}
    for case, values in random_locations.groupby("case"):
        values = values.sort_values("order")
        requested = np.ravel_multi_index(
            tuple(values[key].to_numpy() for key in ["layer", "y", "x"]), shape
        )
        selected = np.searchsorted(grid_ids, requested)
        assert len(selected) == 21 and np.array_equal(grid_ids[selected], requested)
        assert len(np.unique(selected)) == 21
        random_selections[case] = selected
        summaries[case + "_selected"] = selected
        summaries[case + "_ridge"] = random_model["ridge"]
        for variable, mean, sigma in [
            ("T", "temp_mean", "sigmaT_space"),
            ("S", "salt_mean", "sigmaS_space"),
        ]:
            prefix = case + "_" + variable + "_"
            summaries[prefix + "observations"] = np.empty((24, len(selected)))
            summaries[prefix + "basis"] = random_model["EOF_" + variable][selected]
            summaries[prefix + "mean"] = random_model[mean][selected]
            summaries[prefix + "sigma"] = random_model[sigma][selected]
            summaries[prefix + "sigma_base"] = random_model["sigma_base_" + variable]
    assert len(random_selections) == 100
    day = str(first["test_day"])
    minimum = json.loads(str(first["mc_config"]))["HN_MIN"]
    with xr.open_dataset(datafile) as dataset:
        test = dataset.sel(time=slice(day, day))
        dates = test.time.values
        expected = np.datetime64(day, "h") + np.arange(24).astype("timedelta64[h]")
        assert np.array_equal(dates.astype("datetime64[h]"), expected)
        summaries["times"] = dates.astype("datetime64[ns]").astype("int64")
        summaries["counts"] = np.empty(24, dtype="int64")
        for t in range(24):
            fields = [
                np.asarray(test[name].isel(time=t).values[indices], dtype="float64")
                for name in ["temp", "salt", "hn"]
            ]
            valid = (
                np.isfinite(fields[0])
                & np.isfinite(fields[1])
                & np.isfinite(fields[2])
                & (fields[2] > minimum)
            )
            summaries["counts"][t] = valid.sum()
            assert valid.any()
            for method, model in models.items():
                for variable, truth, mean in [
                    ("T", fields[0], "temp_mean"),
                    ("S", fields[1], "salt_mean"),
                ]:
                    prefix = method + "_" + variable + "_"
                    basis = model["EOF_" + variable]
                    anomaly = np.where(valid, truth - model[mean], 0)
                    excluded = basis[~valid]
                    summaries[prefix + "gram"][t] = (
                        model["gram_" + variable] - excluded.T @ excluded
                    )
                    summaries[prefix + "cross"][t] = basis.T @ anomaly
                    summaries[prefix + "energy"][t] = anomaly @ anomaly
                    summaries[prefix + "observations"][t] = np.where(valid, truth, np.nan)[
                        model["selected"]
                    ]
            for case, selected in random_selections.items():
                for variable, truth in [("T", fields[0]), ("S", fields[1])]:
                    summaries[case + "_" + variable + "_observations"][t] = np.where(
                        valid[selected], truth[selected], np.nan
                    )
            if (t + 1) % 6 == 0:
                print(f"Test summaries: {t + 1}/24 hours", flush=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **summaries)
    temporary.replace(cache)
    return summaries


def draws(selected, variable, repetitions, hours):
    import numpy as np

    return np.stack(
        [
            np.column_stack(
                [
                    np.random.default_rng(
                        np.random.SeedSequence([SEED + b, int(cell), variable])
                    ).normal(size=hours)
                    for cell in selected
                ]
            )
            for b in range(repetitions)
        ]
    )


def evaluate(summary, method, variable, noise, strategy, k, standard):
    import numpy as np

    prefix = method + "_" + variable + "_"
    scoring = (
        (
            ("reference" if "reference_T_gram" in summary else "dopt")
            if method.startswith("random_")
            else method
        )
        + "_"
        + variable
        + "_"
    )
    baseline = summary[prefix + "sigma"][:k]
    sigma = (
        baseline * NOISE[noise][0 if variable == "T" else 1] / float(summary[prefix + "sigma_base"])
    )
    fitting = baseline if strategy == "fixed_weights" else sigma
    observations = summary[prefix + "observations"][:, :k]
    means = summary[prefix + "mean"][:k]
    basis = summary[prefix + "basis"][:k]
    ridge = float(summary[method + "_ridge"])
    squared = np.zeros(standard.shape[0])
    clean_total, variance_total = 0.0, 0.0
    for t in range(len(observations)):
        available = np.isfinite(observations[t])
        A = basis[available]
        weights = 1 / fitting[available] ** 2
        mapping = np.linalg.solve(
            A.T @ (A * weights[:, None]) + ridge * np.eye(A.shape[1]), A.T * weights
        )
        clean = mapping @ (observations[t, available] - means[available])
        perturbation = standard[:, t, :k][:, available] * sigma[available]
        coefficients = clean + perturbation @ mapping.T
        gram = summary[scoring + "gram"][t]
        cross = summary[scoring + "cross"][t]
        energy = summary[scoring + "energy"][t]
        squared += (
            np.einsum("bi,ij,bj->b", coefficients, gram, coefficients)
            - 2 * coefficients @ cross
            + energy
        )
        clean_total += clean @ gram @ clean - 2 * clean @ cross + energy
        propagation = mapping * sigma[available]
        variance_total += np.sum(propagation * (gram @ propagation))
    count = summary["counts"].sum()
    mse = squared / count
    assert mse.min() > -1e-8
    rmse = np.sqrt(np.maximum(mse, 0))
    return rmse, max(clean_total / count, 0), max(variance_total / count, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Noise-only diagnostic using all seven completed placement methods."
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--cores",
        type=int,
        default=min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))),
    )
    parser.add_argument("--realizations", type=int, default=100)
    args = parser.parse_args()
    if args.cores < 1 or args.realizations < 2:
        parser.error("Use at least one core and two realizations.")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"]:
        os.environ[key] = str(args.cores)
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    root, methods = locate()
    paths = {method: methods / ".work" / (method + "__frozen_model.npz") for method in METHODS}
    options = [root / "Data/3D_mean_2016-03.nc"]
    datafile = args.data or next((path for path in options if path.exists()), options[0])
    random = methods.parent / "random"
    paths["reference"] = random / ".work/reference__frozen_model.npz"
    locations = random / "sensor_grid_locations.csv"
    sources = [*paths.values(), datafile, locations, Path(__file__)]
    metadata = [
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in sources
    ]
    signature = hashlib.sha256(json.dumps(metadata).encode()).hexdigest()
    output = root / "outputs/noise_weight_check"
    output.mkdir(exist_ok=True)
    with threadpool_limits(limits=args.cores):
        summary = prepare(paths, datafile, locations, output, signature)
        cases = [
            *METHODS,
            *sorted(
                key[:-9]
                for key in summary
                if key.startswith("random_") and key.endswith("_selected")
            ),
        ]
        rows, realizations, random_totals = [], [], {}
        for case_number, case in enumerate(cases):
            method = "random" if case.startswith("random_") else case
            for variable, variable_id in [("T", 0), ("S", 1)]:
                standard = draws(summary[case + "_selected"], variable_id, args.realizations, 24)
                for k in range(1, 22):
                    for strategy in ["noise_dependent_weights", "fixed_weights"]:
                        for noise in NOISE:
                            values, bias, variance = evaluate(
                                summary, case, variable, noise, strategy, k, standard
                            )
                            rows.append(
                                dict(
                                    case=case,
                                    method=method,
                                    variable=variable,
                                    k=k,
                                    strategy=strategy,
                                    noise=noise,
                                    mean_rmse=values.mean(),
                                    std_rmse=values.std(ddof=1),
                                    noise_free_rmse=np.sqrt(bias),
                                    noise_mse=variance,
                                    sqrt_expected_mse=np.sqrt(bias + variance),
                                    first_rmse=values[0],
                                )
                            )
                            if method == "random":
                                key = (variable, k, strategy, noise)
                                random_totals[key] = (
                                    random_totals.get(key, np.zeros(args.realizations)) + values
                                )
                            else:
                                realizations.extend(
                                    dict(
                                        case=case,
                                        method=method,
                                        variable=variable,
                                        k=k,
                                        strategy=strategy,
                                        noise=noise,
                                        realization=b,
                                        rmse=value,
                                    )
                                    for b, value in enumerate(values)
                                )
            if method != "random" or (case_number - len(METHODS) + 1) % 10 == 0:
                print(f"{case}: diagnostic complete", flush=True)
    cases_frame = pd.DataFrame(rows)
    original = pd.concat(
        [pd.read_csv(folder / "reconstruction_scores.csv") for folder in (methods, random)],
        ignore_index=True,
    ).set_index(["case", "noise", "k"])
    assert original.index.is_unique
    checks = []
    for row in cases_frame[cases_frame.strategy == "noise_dependent_weights"].itertuples():
        expected = float(original.at[(row.case, row.noise, row.k), "rmse_" + row.variable])
        checks.append(
            dict(
                case=row.case,
                method=row.method,
                variable=row.variable,
                noise=row.noise,
                k=row.k,
                saved_rmse=expected,
                diagnostic_rmse=row.first_rmse,
                absolute_difference=abs(expected - row.first_rmse),
            )
        )
    checks = pd.DataFrame(checks)
    checks.to_csv(output / "reproduction_check.csv", index=False)
    if checks.absolute_difference.max() > 5e-5:
        raise ValueError(
            "Saved-result reproduction failed. Inspect reproduction_check.csv before interpreting this diagnostic."
        )
    fixed = cases_frame[cases_frame.strategy == "fixed_weights"]
    for key, group in fixed.groupby(["case", "variable", "k"]):
        group = group.set_index("noise").loc[list(NOISE)]
        assert np.all(np.diff(group.sqrt_expected_mse) >= -1e-10), key
        assert np.ptp(group.noise_free_rmse) < 1e-10, key
    low = cases_frame[cases_frame.noise == "low"].pivot(
        index=["case", "variable", "k"], columns="strategy", values="mean_rmse"
    )
    assert np.allclose(low.fixed_weights, low.noise_dependent_weights, rtol=0, atol=1e-12)
    frame = cases_frame[cases_frame.method != "random"].copy()
    frame["noise_sd"] = frame.std_rmse
    frame["layout_sd"] = 0.0
    frame["ensemble_noise_sd"] = frame.std_rmse
    random_rows = []
    for key, group in cases_frame[cases_frame.method == "random"].groupby(
        ["variable", "k", "strategy", "noise"]
    ):
        variable, k, strategy, noise = key
        assert len(group) == 100
        ensemble = random_totals[key] / len(group)
        pooled_variance = (
            (args.realizations - 1) * (group.std_rmse**2).sum()
            + args.realizations * ((group.mean_rmse - group.mean_rmse.mean()) ** 2).sum()
        ) / (len(group) * args.realizations - 1)
        random_rows.append(
            dict(
                case="random_ensemble",
                method="random",
                variable=variable,
                k=k,
                strategy=strategy,
                noise=noise,
                mean_rmse=group.mean_rmse.mean(),
                std_rmse=np.sqrt(pooled_variance),
                noise_sd=np.sqrt(np.mean(group.std_rmse**2)),
                layout_sd=group.mean_rmse.std(ddof=1),
                ensemble_noise_sd=ensemble.std(ddof=1),
                noise_free_rmse=np.sqrt(np.mean(group.noise_free_rmse**2)),
                noise_mse=group.noise_mse.mean(),
                sqrt_expected_mse=np.sqrt(np.mean(group.sqrt_expected_mse**2)),
                first_rmse=group.first_rmse.mean(),
            )
        )
        realizations.extend(
            dict(
                case="random_ensemble",
                method="random",
                variable=variable,
                k=k,
                strategy=strategy,
                noise=noise,
                realization=b,
                rmse=value,
            )
            for b, value in enumerate(ensemble)
        )
    frame = pd.concat([frame, pd.DataFrame(random_rows)], ignore_index=True)
    frame.to_csv(output / "noise_summary.csv", index=False)
    cases_frame.to_csv(output / "case_summary.csv", index=False)
    pd.DataFrame(realizations).to_csv(output / "noise_realizations.csv", index=False)
    protocol = dict(
        methods=[*METHODS, "random"],
        realizations=args.realizations,
        random_layouts=100,
        seed=SEED,
        noise=NOISE,
        test_period="4 March 2016, 24 hourly records",
        sensor_counts=list(range(1, 22)),
        layouts="frozen",
        EOFs="frozen",
        ridge="saved value, unchanged",
        fixed_weights="saved baseline inverse observation variances",
        pairing="same standardized draws for matching cells, variables, realizations and times",
        scoring="full valid 3D domain, exact quadratic error summaries",
        sqrt_expected_mse="square root of analytic expected MSE, not mean RMSE",
        random_noise_sd="root mean within-layout noise variance",
        random_layout_sd="standard deviation of the 100 layout mean RMSEs",
        random_std_rmse="pooled standard deviation of all layout-realization RMSEs",
        random_realization_rows="mean RMSE over 100 layouts for each matched noise realization",
        maximum_reproduction_difference=float(checks.absolute_difference.max()),
        uncertainty_intervals="not evaluated",
        sources=metadata,
        elapsed_seconds=time.perf_counter() - started,
        cores=args.cores,
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
