import json
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

from sacremoses import MosesDetokenizer, MosesTokenizer
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

    NS = {"fn": "http://framenet.icsi.berkeley.edu"}
    annotations = []

    allowed_frames = [
        l.strip() for l in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    detok = MosesDetokenizer(lang="fr")
    tok = MosesTokenizer(lang="fr")

    for file in args.input.glob("*.xml"):
        root = ET.fromstring(open(file).read())
        frame_name = root.get("frame")

        if args.filter_language_specifics and frame_name not in allowed_frames:
            continue

        for s_el in root.findall(".//fn:sentence", NS):
            sentence = s_el.find("fn:text", NS).text

            # extract tokens and save which char spans a token covers
            # spans are saved as [inclusive:inclusive]
            tokens = tok.tokenize(sentence)
            token_spans = []
            pointer = 0
            for token in tokens:
                surface = detok.detokenize([token])
                try:
                    start = sentence.index(surface, pointer)
                except ValueError:
                    # surface not found from pointer (e.g. entity-encoded token
                    # like &apos; whose pointer advanced past the actual char);
                    # fall back to first occurrence in the full sentence
                    try:
                        start = sentence.index(surface, 0)
                    except ValueError:
                        token_spans.append((pointer, pointer))
                        continue
                end = start + len(surface) - 1  # use surface length, not token length
                token_spans.append((start, end))
                pointer = end + 1

            # annotations are stored in annotation sets, each AS
            # contains 2 layers, one for the target and one for the FEs
            annotation = {
                "sentence": detok.detokenize(tokens),
                "tokens": tokens,
                "language": "fr",
                "metadata": {"sentence_id": s_el.get("ID")},
                "frames": [],
            }

            for ann_el in s_el.findall("./fn:annotationSet", NS):
                frame_ann = {
                    "name": frame_name,
                    "roles": [],
                }
                seen_role_names = set()

                for layer in ann_el.findall("./fn:layer", NS):
                    if layer.get("name") == "Target":
                        # extract frame trigger
                        trigger_el = layer.find("./fn:label", NS)
                        if trigger_el is None:
                            continue
                        char_start, char_end = (
                            int(trigger_el.get("start")),
                            int(trigger_el.get("end")),
                        )
                        trigger_token_idxs = [
                            i
                            for i, (ts, te) in enumerate(token_spans)
                            if ts >= char_start and te <= char_end
                        ]

                        if len(trigger_token_idxs) == 0:
                            continue

                        i, j = min(trigger_token_idxs), max(trigger_token_idxs)
                        activation = detok.detokenize(tokens[i : j + 1]).strip()
                        if not activation:
                            continue
                        frame_ann["idxs"] = (i, j)
                        frame_ann["activation"] = activation
                    elif layer.get("name") == "FE":
                        # extract the FEs
                        for fe_el in layer.findall("./fn:label", NS):
                            if "start" in fe_el.keys() and "end" in fe_el.keys():
                                role_name = fe_el.get("name")
                                if role_name in seen_role_names:
                                    continue
                                f_start_char_i, f_end_char_i = (
                                    int(fe_el.get("start")),
                                    int(fe_el.get("end")),
                                )
                                fe_token_idxs = [
                                    i
                                    for i, (ts, te) in enumerate(token_spans)
                                    if ts >= f_start_char_i and te <= f_end_char_i
                                ]

                                if len(fe_token_idxs) == 0:
                                    continue

                                i, j = min(fe_token_idxs), max(fe_token_idxs)
                                filler = detok.detokenize(tokens[i : j + 1]).strip()
                                if not filler:
                                    continue
                                seen_role_names.add(role_name)
                                frame_ann["roles"].append(
                                    {
                                        "name": role_name,
                                        "filler": filler,
                                        "idxs": (i, j),
                                    }
                                )

                if "activation" not in frame_ann or not frame_ann["roles"]:
                    continue
                annotation["frames"].append(frame_ann)

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
