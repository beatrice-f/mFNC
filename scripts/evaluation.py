import json
from copy import deepcopy
from glob import glob

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from FairEval import compare_spans, precision, recall

metrics = []

# as suggested
weigths = {
    "TP": {"TP": 1},
    "FP": {"FP": 1},
    "FN": {"FN": 1},
    "LE": {"TP": 0, "FP": 0.5, "FN": 0.5},
    "BE": {"TP": 0.5, "FP": 0.25, "FN": 0.25},
    "LBE": {"TP": 0, "FP": 0.5, "FN": 0.5},
}


def evaluate(path):
    ds_preds = [json.loads(l.strip()) for l in open(path).readlines()]
    ds_metrics = {
        "model": path,
    }

    frame_golds, frame_preds = [], []
    role_golds, role_preds = [], []
    for sample in ds_preds:
        # frame
        for l, k in zip([frame_golds, frame_preds], ["frames", "predictions"]):
            l.extend(
                [
                    [
                        f["name"],
                        f["idxs"][0],
                        f["idxs"][1],
                        set(range(f["idxs"][0], f["idxs"][1] + 1)),
                    ]
                    for f in sample[k]
                ]
            )

        # role
        for l, k in zip([role_golds, role_preds], ["frames", "predictions"]):
            l.extend(
                [
                    [
                        f"{f['name']}.{r['name']}",
                        r["idxs"][0],
                        r["idxs"][1],
                        set(range(r["idxs"][0], r["idxs"][1] + 1)),
                    ]
                    for f in sample[k]
                    for r in f["roles"]
                ]
            )

    # traditional metrics
    for k in ["traditional", "fair"]:
        comparison = compare_spans(deepcopy(frame_golds), deepcopy(frame_preds))
        ds_metrics[f"{k}_frame_p"] = precision(comparison["overall"][k], k)
        ds_metrics[f"{k}_frame_r"] = recall(comparison["overall"][k], k)
        ds_metrics[f"{k}_frame_f1"] = (
            2 * ds_metrics[f"{k}_frame_p"] * ds_metrics[f"{k}_frame_r"]
        ) / (ds_metrics[f"{k}_frame_p"] + ds_metrics[f"{k}_frame_r"])
        
        comparison = compare_spans(deepcopy(role_golds), deepcopy(role_preds))
        ds_metrics[f"{k}_role_p"] = precision(comparison["overall"][k], k)
        ds_metrics[f"{k}_role_r"] = recall(comparison["overall"][k], k)
        ds_metrics[f"{k}_role_f1"] = (
            2 * ds_metrics[f"{k}_role_p"] * ds_metrics[f"{k}_role_r"]
        ) / (ds_metrics[f"{k}_role_p"] + ds_metrics[f"{k}_role_r"])
        

    return ds_metrics


if __name__ == "__main__":
    try:
        df = pd.read_csv("baselines/ood/metrics.csv")
        paths = [
            p for p in glob("baselines/ood/**/**") if p not in df.model.values
        ]
        print(paths)
    except:
        df = None
        paths = glob("baselines/ood/**/**")

    metrics = list(map(evaluate, tqdm(paths)))

    if df is None:
        pd.DataFrame.from_dict(metrics).to_csv("baselines/ood/metrics.csv")
    else:
        pd.concat([pd.DataFrame.from_dict(metrics), df]).to_csv(
            "baselines/ood/metrics.csv"
        )
