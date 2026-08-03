import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from skmultilearn.model_selection import IterativeStratification

argparser = ArgumentParser()
argparser.add_argument(
    "-i",
    "--input",
    required=True,
    help="The CFN directory",
    type=Path,
)
argparser.add_argument(
    "-o",
    "--output",
    required=True,
    help="The output directory where the split are being written.",
    type=Path,
)
argparser.add_argument(
    "--filter-language-specifics",
    required=False,
    help="Remove the language specific frames.",
    default=False,
    action="store_true"
)


def merge_and_dedup(annotations):
    unique = {}
    for ann in annotations:
        sent = ann["sentence"]
        if sent not in unique:
            unique[sent] = ann
        else:
            unique[sent]["frames"].extend(ann["frames"])
    merged = list(unique.values())
    for ann in merged:
        seen_activations = set()
        deduped = []
        for fr in ann["frames"]:
            if fr["activation"] not in seen_activations:
                seen_activations.add(fr["activation"])
                deduped.append(fr)
        ann["frames"] = deduped
    return merged


if __name__ == "__main__":
    args = argparser.parse_args()

    # load mappings
    mapping = json.load((args.input / "frame_info.json").open())
    cf2ef = {m["frame_name"]: m["frame_ename"] for m in mapping}
    cr2er = {
        m["frame_ename"]: {fe["fe_name"]: fe["fe_ename"] for fe in m["fes"]}
        for m in mapping
    }

    allowed_frames = [
        l.strip() for l in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    documents = []
    for fname in ["cfn-train.json", "cfn-dev.json"]:
        for doc in json.load((args.input / fname).open()):
            text = doc["text"]
            frame = cf2ef.get(doc["frame"], "")
            if frame == "" or (args.filter_language_specifics and frame not in allowed_frames):
                continue

            roles_map = cr2er[frame]
            words = doc["word"]
            tokens = [text[w["start"] : w["end"] + 1] for w in words]

            def char_to_tok(char_start, char_end):
                return [
                    i for i, w in enumerate(words)
                    if w["start"] >= char_start and w["end"] <= char_end
                ]

            target_tok = char_to_tok(doc["target"]["start"], doc["target"]["end"])
            if not target_tok:
                continue

            roles = []
            for span in doc["cfn_spans"]:
                if span["fe_name"] not in roles_map:
                    continue
                role_name = roles_map[span["fe_name"]]
                if not role_name:
                    continue
                role_tok = char_to_tok(span["start"], span["end"])
                if not role_tok:
                    continue
                roles.append({
                    "name": role_name,
                    "idxs": (min(role_tok), max(role_tok)),
                    "filler": text[span["start"] : span["end"] + 1],
                })

            if not roles:
                continue

            documents.append({
                "sentence": text,
                "tokens": tokens,
                "language": "zh",
                "metadata": {"sentence_id": doc["sentence_id"]},
                "frames": [{
                    "name": frame,
                    "idxs": (min(target_tok), max(target_tok)),
                    "activation": text[
                        doc["target"]["start"] : doc["target"]["end"] + 1
                    ],
                    "roles": roles,
                }],
            })

    annotations = merge_and_dedup(documents)

    # After merging, records from different tokenizations of the same sentence
    # can have frame/role idxs that are valid for the source word list but
    # out-of-bounds for the canonical token list kept by merge_and_dedup.
    # Drop any such frames/roles rather than propagating bad indices.
    def idxs_valid(idxs, n):
        return (
            isinstance(idxs, (list, tuple))
            and len(idxs) == 2
            and idxs[0] <= idxs[1]
            and idxs[0] >= 0
            and idxs[1] < n
        )

    clean = []
    for ann in annotations:
        n = len(ann["tokens"])
        valid_frames = []
        for fr in ann["frames"]:
            if not idxs_valid(fr.get("idxs"), n):
                continue
            valid_roles = [
                r for r in fr["roles"] if idxs_valid(r.get("idxs"), n)
            ]
            if not valid_roles:
                continue
            fr["roles"] = valid_roles
            valid_frames.append(fr)
        if valid_frames:
            ann["frames"] = valid_frames
            clean.append(ann)
    annotations = clean

    all_frames = sorted({fr["name"] for ann in annotations for fr in ann["frames"]})
    frame_to_idx = {fr: i for i, fr in enumerate(all_frames)}
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
