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
    detok = MosesDetokenizer(lang="it")

    allowed_frames = [
        l.strip() for l in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    annotations = []
    root = ET.fromstring(open(args.input).read())
    for sentence in root.findall(".//s"):
        sentence_graph = sentence.find("graph")

        node_to_text = dict()
        annotation = dict()

        # parse terminals
        node_to_span_idxs = {}
        for t in sentence_graph.findall("./terminals//t"):
            node_to_text[t.get("id")] = t.get("word")

            i = int(t.get("id").split("_")[-1]) - 1
            node_to_span_idxs[t.get("id")] = (i, i)

        tokens = list(node_to_text.values())

        annotation = {
            "sentence": detok.detokenize(tokens),
            "tokens": tokens,
            "language": "it",
            "metadata": {"corpus": "ISST", "id": sentence.get("id")},
            "frames": [],
        }

        # parse non-terminals — TIGER format lists them bottom-up so children
        # are always processed before their parents
        for nt in sentence_graph.findall("./nonterminals//nt"):
            child_spans = []
            nt_words = []
            for edge in nt.findall(".//edge"):
                child_id = edge.get("idref")
                if child_id not in node_to_span_idxs:
                    continue
                c_start, c_end = node_to_span_idxs[child_id]
                child_spans.append((c_start, c_end))
                nt_words.append((c_start, node_to_text[child_id]))

            if not child_spans:
                continue

            node_to_span_idxs[nt.get("id")] = (
                min(s for s, _ in child_spans),
                max(e for _, e in child_spans),
            )
            node_to_text[nt.get("id")] = " ".join(
                w for _, w in sorted(nt_words)
            )

        # parse the semantic annotation
        for frame in sentence.findall("sem//frames/frame"):
            if args.filter_language_specifics and frame.get("name") not in allowed_frames:
                continue

            fenode = frame.find("target/fenode")
            if fenode is None:
                continue
            activation_id = fenode.get("idref")

            # TODO: Some sentences annotated on splitwords might be lost in this way!
            if activation_id not in node_to_span_idxs:
                continue

            activation = node_to_text[activation_id].strip()
            if not activation:
                continue

            roles = []
            seen_role_names = set()
            for fe in frame.findall("fe"):
                fe_node = fe.find("fenode")
                if fe_node is None:
                    continue
                fe_id = fe_node.get("idref")
                if fe_id not in node_to_text:
                    continue
                role_name = fe.get("name")
                if role_name in seen_role_names:
                    continue
                filler = node_to_text[fe_id].strip()
                if not filler:
                    continue
                seen_role_names.add(role_name)
                roles.append(
                    {
                        "name": role_name,
                        "filler": filler,
                        "idxs": node_to_span_idxs[fe_id],
                    }
                )

            if not roles:
                continue

            annotation["frames"].append({
                "name": frame.get("name"),
                "idxs": node_to_span_idxs[activation_id],
                "activation": activation,
                "roles": roles,
            })

        if annotation["frames"]:
            annotations.append(annotation)

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
