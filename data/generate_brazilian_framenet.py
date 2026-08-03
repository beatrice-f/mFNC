import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sacremoses import MosesDetokenizer
from skmultilearn.model_selection import IterativeStratification

argparser = ArgumentParser()
argparser.add_argument(
    "-i",
    "--input",
    required=True,
    help="The PT.jsonl file from frame-squared/data available at https://github.com/FrameNetBrasil/frame-squared",
)
argparser.add_argument(
    "-o",
    "--output",
    required=True,
    help="The output directory where the split are being written.",
)
argparser.add_argument(
    "--filter-language-specifics",
    required=False,
    help="Remove the language specific frames.",
    default=False,
    action="store_true"
)

if __name__ == "__main__":
    args = argparser.parse_args()

    allowed_frames = [
        l.strip() for l in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    source_annotations = [json.loads(line) for line in open(args.input).readlines()]

    detok = MosesDetokenizer(lang="pt")

    annotations = []
    for s_ann in source_annotations:
        tokens = s_ann["tokens"]
        ann = {
            "sentence": s_ann["sentence"],
            "tokens": tokens,
            "language": "pt",
            "metadata": {
                "episode": s_ann["episode"],
                "sentenceId": s_ann["sentenceId"],
                "sentenceTimespan": s_ann["sentenceTimespan"],
            },
            "frames": [],
        }

        for f in s_ann["frames"]:
            f_i, f_j = f["span"][0], f["span"][1]

            if args.filter_language_specifics and f["id"] not in allowed_frames:
                continue

            activation = detok.detokenize(tokens[f_i : f_j + 1]).strip()
            if not activation:
                continue

            roles = []
            seen_role_names = set()
            for r in f["frameElements"]:
                if r["id"] in seen_role_names:
                    continue
                r_i, r_j = r["span"][0], r["span"][1]
                filler = detok.detokenize(tokens[r_i : r_j + 1]).strip()
                if not filler:
                    continue
                seen_role_names.add(r["id"])
                roles.append(
                    {
                        "name": r["id"],
                        "idxs": (r_i, r_j),
                        "filler": filler,
                    }
                )

            if not roles:
                continue

            ann["frames"].append(
                {
                    "name": f["id"],
                    "idxs": (f_i, f_j),
                    "activation": activation,
                    "roles": roles,
                }
            )

        if len(ann["frames"]) > 0:
            annotations.append(ann)

    unique_annotations = {}
    for ann in annotations:
        sent = ann["sentence"]
        if sent not in unique_annotations:
            unique_annotations[sent] = ann
        else:
            unique_annotations[sent]["frames"].extend(ann["frames"])
    annotations = list(unique_annotations.values())

    # deduplicate frames by activation within each sentence
    for ann in annotations:
        seen_activations = set()
        deduped_frames = []
        for fr in ann["frames"]:
            if fr["activation"] not in seen_activations:
                seen_activations.add(fr["activation"])
                deduped_frames.append(fr)
        ann["frames"] = deduped_frames

    all_frames = sorted({fr["name"] for ann in annotations for fr in ann["frames"]})
    frame_to_idx = {f: i for i, f in enumerate(all_frames)}
    X = np.arange(len(annotations)).reshape(-1, 1)
    y = np.zeros((len(annotations), len(all_frames)), dtype=int)
    for i, ann in enumerate(annotations):
        for fr in ann["frames"]:
            y[i, frame_to_idx[fr["name"]]] = 1

    stratifier = IterativeStratification(
        n_splits=3,
        order=2,
        sample_distribution_per_fold=[0.7, 0.1, 0.2],
        random_state=42,
    )
    splits = list(stratifier.split(X, y))
    ann_train = [annotations[i] for i in splits[0][1]]
    ann_valid = [annotations[i] for i in splits[1][1]]
    ann_test = [annotations[i] for i in splits[2][1]]
    for anns, anns_label in zip(
        [ann_train, ann_valid, ann_test], ["train", "valid", "test"]
    ):
        with open(f"{args.output}.{anns_label}.jsonl", "w") as f:
            for a in anns:
                if len(a["frames"]) > 0:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
