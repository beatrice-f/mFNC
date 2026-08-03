import json
from argparse import ArgumentParser
from pathlib import Path

from sacremoses import MosesDetokenizer
import numpy as np
from skmultilearn.model_selection import IterativeStratification

argparser = ArgumentParser()
argparser.add_argument("-i", "--input", required=True, type=Path)
argparser.add_argument("-o", "--output", required=True)
argparser.add_argument(
    "--filter-language-specifics",
    required=False,
    help="Remove the language specific frames.",
    default=False,
    action="store_true"
)

if __name__ == "__main__":
    args = argparser.parse_args()

    detok = MosesDetokenizer("ko")

    allowed_frames = [
        l.strip() for l in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    annotations = []
    for file in args.input.glob("*.tsv"):
        docs = open(file).read().strip().split("\n\n")

        for doc in docs:
            doc_id = doc.split("\n")[0].split(":")[-1]
            lines = doc.split("\n")[2:]
            tokens = [l.split("\t")[1] for l in lines]

            ann = {
                "sentence": detok.detokenize(tokens),
                "tokens": tokens,
                "language": "ko",
                "metadata": {"id": doc_id},
                "frames": [],
            }

            prev_frame_el = "O"
            seen_role_names = set()
            frame_ann = {"activation": [], "idxs": [], "roles": []}
            for line in lines:
                i, token, verb_lemma, frame_name, frame_el = line.split("\t")
                i = int(i)
                frame_el = frame_el.split("-")[-1]

                if verb_lemma != "_":
                    frame_ann["name"] = frame_name
                    frame_ann["activation"].append(tokens[i])
                    frame_ann["idxs"].append(i)

                if frame_el != "O":
                    if frame_el != prev_frame_el:
                        if frame_el not in seen_role_names:
                            seen_role_names.add(frame_el)
                            frame_ann["roles"].append(
                                {
                                    "name": frame_el,
                                    "filler": [token],
                                    "idxs": [i],
                                }
                            )
                            prev_frame_el = frame_el
                    else:
                        frame_ann["roles"][-1]["filler"].append(token)
                        frame_ann["roles"][-1]["idxs"].append(i)

            if "name" not in frame_ann or not frame_ann["idxs"]:
                continue

            activation = detok.detokenize(frame_ann["activation"]).strip()
            if not activation:
                continue

            frame_ann["activation"] = activation
            frame_ann["idxs"] = (min(frame_ann["idxs"]), max(frame_ann["idxs"]))

            roles = []
            for r in frame_ann["roles"]:
                filler = detok.detokenize(r["filler"]).strip()
                if not filler:
                    continue
                roles.append({
                    "name": r["name"],
                    "filler": filler,
                    "idxs": (min(r["idxs"]), max(r["idxs"])),
                })
            frame_ann["roles"] = roles

            if not frame_ann["roles"]:
                continue

            if args.filter_language_specifics and frame_ann["name"] not in allowed_frames:
                continue

            ann["frames"].append(frame_ann)
            annotations.append(ann)

    unique_annotations = {}
    for ann in annotations:
        sent = ann["sentence"]
        if sent not in unique_annotations:
            unique_annotations[sent] = ann
        else:
            unique_annotations[sent]["frames"].extend(ann["frames"])
    annotations = list(unique_annotations.values())

    for ann in annotations:
        seen_activations = set()
        deduped = []
        for fr in ann["frames"]:
            if fr["activation"] not in seen_activations:
                seen_activations.add(fr["activation"])
                deduped.append(fr)
        ann["frames"] = deduped

    all_frames = sorted({fr["name"] for ann in annotations for fr in ann["frames"]})
    frame_to_idx = {f: i for i, f in enumerate(all_frames)}
    X = np.arange(len(annotations)).reshape(-1, 1)
    y = np.zeros((len(annotations), len(all_frames)), dtype=int)
    for i, ann in enumerate(annotations):
        for fr in ann["frames"]:
            y[i, frame_to_idx[fr["name"]]] = 1

    stratifier = IterativeStratification(
        n_splits=3, order=2, sample_distribution_per_fold=[0.7, 0.1, 0.2],
        random_state=42,
    )
    splits = list(stratifier.split(X, y))
    ann_train = [annotations[i] for i in splits[0][1]]
    ann_valid = [annotations[i] for i in splits[1][1]]
    ann_test  = [annotations[i] for i in splits[2][1]]

    for anns, anns_label in zip(
        [ann_train, ann_valid, ann_test], ["train", "valid", "test"]
    ):
        with open(f"{args.output}.{anns_label}.jsonl", "w") as f:
            for a in anns:
                if len(a["frames"]) > 0:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
