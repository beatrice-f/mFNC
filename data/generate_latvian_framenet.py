import json
from argparse import ArgumentParser
from pathlib import Path

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

def parse_conll(filepath, allowed_frames):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    annotations = []
    for sent_block in content.strip().split("\n\n"):
        lines = sent_block.split("\n")

        sent_id = None
        sentence = None
        for line in lines:
            if line.startswith("# sent_id = "):
                sent_id = line[len("# sent_id = "):].strip()
            elif line.startswith("# text = "):
                sentence = line[len("# text = "):].strip()

        if sentence is None:
            continue

        token_lines = [l for l in lines if l.strip() and not l.startswith("#")]
        if not token_lines:
            continue

        tokens = []
        predicate_idx = None
        frame_name = None
        role_tokens = {}  # role_name -> [(0-based idx, form)]

        for tline in token_lines:
            parts = tline.split("\t")
            if len(parts) != 12:
                continue
            idx = int(parts[0]) - 1  # 0-based
            form = parts[1]
            tokens.append(form)

            if parts[9] == "Y":
                predicate_idx = idx
                frame_name = parts[10]

            role = parts[11]
            if role != "_":
                role_tokens.setdefault(role, []).append((idx, form))

        if predicate_idx is None or not frame_name or frame_name == "_":
            continue
        if args.filter_language_specifics and frame_name not in allowed_frames:
            continue
        if not role_tokens:
            continue

        roles = []
        for role_name, toks in role_tokens.items():
            toks_sorted = sorted(toks, key=lambda x: x[0])
            filler = " ".join(t for _, t in toks_sorted)
            roles.append({
                "name": role_name,
                "filler": filler,
                "idxs": [toks_sorted[0][0], toks_sorted[-1][0]],
            })

        annotations.append({
            "sentence": sentence,
            "tokens": tokens,
            "language": "lv",
            "metadata": {"id": sent_id},
            "frames": [{
                "name": frame_name,
                "activation": tokens[predicate_idx],
                "idxs": [predicate_idx, predicate_idx],
                "roles": roles,
            }],
        })

    return annotations


if __name__ == "__main__":
    args = argparser.parse_args()

    allowed_frames = [
        l.strip() for l in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    annotations = (
        parse_conll(args.input / "train.conll2009", allowed_frames) +
        parse_conll(args.input / "test.conll2009", allowed_frames)
    )

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

    for anns, label in zip([ann_train, ann_valid, ann_test], ["train", "valid", "test"]):
        with open(f"{args.output}.{label}.jsonl", "w") as f:
            for a in anns:
                if a["frames"]:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
