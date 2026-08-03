import json
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

from sacremoses import MosesDetokenizer
import numpy as np
from skmultilearn.model_selection import IterativeStratification

argparser = ArgumentParser()
argparser.add_argument("-i", "--input", required=True)
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

    # detokenizer to convert tokens into sentence
    detok = MosesDetokenizer(lang="sv")

    allowed_frames = [
        l.strip() for l in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    annotations = []
    root = ET.fromstring(open(args.input).read())
    for a_el in root.findall(".//text"):
        frame_name = a_el.get("BFNID")
        if args.filter_language_specifics and frame_name not in allowed_frames:
            continue

        for s_el in a_el.findall(".//sentence"):
            tokens_els = s_el.findall(".//token")
            tokens = [t.text for t in tokens_els]

            sentence_ann = {
                "sentence": detok.detokenize(tokens),
                "tokens": tokens,
                "language": "sv",
                "metadata": {},
                "frames": [],
            }

            activation_els = s_el.findall(f".//element[@name='LU']//token")
            activation_idxs = [int(el.get("ref")) - 1 for el in activation_els]
            if len(activation_idxs) == 0:
                continue

            activation = detok.detokenize([t.text for t in activation_els]).strip()
            if not activation:
                continue

            roles = []
            seen_role_names = set()
            for el in s_el.findall(".//element"):
                if el.get("name") != "LU":
                    role_name = el.get("name")
                    if not role_name:
                        continue
                    if role_name in seen_role_names:
                        continue
                    el_tokens_els = sorted(
                        el.findall("token"), key=lambda e: int(e.get("ref"))
                    )
                    el_tokens = [t.text for t in el_tokens_els]
                    el_tokens_idxs = [int(t.get("ref")) - 1 for t in el_tokens_els]
                    if len(el_tokens_idxs) == 0:
                        continue
                    filler = detok.detokenize(el_tokens).strip()
                    if not filler:
                        continue
                    seen_role_names.add(role_name)
                    roles.append(
                        {
                            "name": role_name,
                            "filler": filler,
                            "idxs": (min(el_tokens_idxs), max(el_tokens_idxs)),
                        }
                    )

            if not roles:
                continue

            frame_ann = {
                "name": frame_name,
                "activation": activation,
                "idxs": (min(activation_idxs), max(activation_idxs)),
                "roles": roles,
            }

            sentence_ann["frames"].append(frame_ann)
            annotations.append(sentence_ann)

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
