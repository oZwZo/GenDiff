"""
gendiff_dev — four-layer GenDiff model-development package (see MODEL_DEV_ARCHITECTURE.md).

    data/        dataset registry      (name -> spec)
    targets/     ΔX target registry    (name -> versioned ndarray + manifest; never mutated in place)
    models/      predictor registry    (name -> predictor; fit/predict_delta[/predict_population])
    eval/        unified metric panel  (L1/L2/L3 + composite + pt_drift + rev_collin)
    experiments/ one runner            ((dataset, target, model) -> results)
    results/     one long-format store (dataset, target, model, metric, value, stratum)

A new direction enters as a registry entry + a config, not an edit to anyone else's code.
"""
import datetime as _dt


def _now():
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
