import json
import xml.etree.ElementTree as ET
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

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def capitalize_first(s):
    return s[0].upper() + s[1:] if s else s


def build_maps(root):
    """Return wf_info, sent_wfs, term_to_wfs from a NAF root element."""
    text_layer = root.find("text")

    wf_info = {}   # wf_id -> (sent_id, pos_in_sent, text)
    sent_wfs = {}  # sent_id -> list of wf_ids in sentence order
    sent_pos = {}  # sent_id -> next position counter

    for wf in text_layer.findall("wf"):
        wid = wf.get("id")
        sent = wf.get("sent")
        word = wf.text.strip() if wf.text else ""
        pos = sent_pos.get(sent, 0)
        sent_pos[sent] = pos + 1
        wf_info[wid] = (sent, pos, word)
        sent_wfs.setdefault(sent, []).append(wid)

    def resolve(wid):
        # sub-token ids like "w16.sub1" -> parent "w16"
        return wid.split(".sub")[0] if ".sub" in wid else wid

    terms_layer = root.find("terms")
    term_to_wfs = {}  # term/component id -> list of parent wf_ids
    for term in terms_layer.findall("term"):
        tid = term.get("id")
        span = term.find("span")
        if span is not None:
            term_to_wfs[tid] = [resolve(t.get("id")) for t in span.findall("target")]
        for comp in term.findall("component"):
            cid = comp.get("id")
            cspan = comp.find("span")
            if cspan is not None:
                term_to_wfs[cid] = [resolve(t.get("id")) for t in cspan.findall("target")]

    return wf_info, sent_wfs, term_to_wfs


def resolve_span(span_el, term_to_wfs, wf_info):
    """Map a <span> element's targets to a list of (sent_id, pos, text) tuples."""
    if span_el is None:
        return []
    results = []
    for target in span_el.findall("target"):
        tid = target.get("id")
        wf_ids = term_to_wfs.get(tid)
        if wf_ids is None:
            # fall back: treat tid as a direct wf_id (resolve sub-tokens)
            wf_ids = [tid.split(".sub")[0] if ".sub" in tid else tid]
        for wid in wf_ids:
            info = wf_info.get(wid)
            if info:
                results.append(info)
    return results


if __name__ == "__main__":
    args = argparser.parse_args()

    detok = MosesDetokenizer(lang="nl")

    allowed_frames = [
        line.strip()
        for line in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    all_annotations = {}  # sentence text -> annotation dict

    for naf_file in sorted(args.input.glob("*.naf")):
        try:
            tree = ET.parse(naf_file)
        except ET.ParseError:
            continue

        root = tree.getroot()

        # only process Dutch-language documents
        if root.get(XML_LANG) != "nl":
            continue

        srl = root.find("srl")
        if srl is None:
            continue

        wf_info, sent_wfs, term_to_wfs = build_maps(root)

        # pre-build sentence tokens and text
        sent_tokens = {}
        sent_text = {}
        for sent_id, wf_ids in sent_wfs.items():
            tokens = [wf_info[wid][2] for wid in wf_ids]
            sent_tokens[sent_id] = tokens
            sent_text[sent_id] = detok.detokenize(tokens)

        for pred in srl.findall("predicate"):
            if pred.get("status") == "deprecated":
                continue

            evoke_ref = pred.find('.//externalRef[@reftype="evoke"]')
            if evoke_ref is None:
                continue
            raw_ref = evoke_ref.get("reference", "")
            if "fn17-" not in raw_ref:
                continue
            frame_name = capitalize_first(raw_ref.split("fn17-")[-1])
            if args.filter_language_specifics and frame_name not in allowed_frames:
                continue

            pred_infos = resolve_span(pred.find("span"), term_to_wfs, wf_info)
            if not pred_infos:
                continue

            pred_sents = {s for s, _, _ in pred_infos}
            if len(pred_sents) != 1:
                continue
            sent_id = next(iter(pred_sents))

            pred_positions = [pos for _, pos, _ in pred_infos]
            activation_tokens = [t for _, _, t in sorted(pred_infos, key=lambda x: x[1])]
            activation = detok.detokenize(activation_tokens).strip()
            if not activation:
                continue

            roles = []
            seen_role_names = set()
            for role in pred.findall("role"):
                if role.get("status") == "deprecated":
                    continue

                role_ref = role.find(".//externalRef")
                if role_ref is None:
                    continue
                raw_role = role_ref.get("reference", "")
                if "@" not in raw_role:
                    continue
                role_name = capitalize_first(raw_role.split("@")[-1])
                if role_name in seen_role_names:
                    continue

                role_infos = resolve_span(role.find("span"), term_to_wfs, wf_info)
                # keep only tokens in the predicate's sentence
                role_infos = [r for r in role_infos if r[0] == sent_id]
                if not role_infos:
                    continue

                role_positions = [pos for _, pos, _ in role_infos]
                filler_tokens = [t for _, _, t in sorted(role_infos, key=lambda x: x[1])]
                filler = detok.detokenize(filler_tokens).strip()
                if not filler:
                    continue

                seen_role_names.add(role_name)
                roles.append({
                    "name": role_name,
                    "filler": filler,
                    "idxs": [min(role_positions), max(role_positions)],
                })

            if not roles:
                continue

            frame_ann = {
                "name": frame_name,
                "activation": activation,
                "idxs": [min(pred_positions), max(pred_positions)],
                "roles": roles,
            }

            sentence = sent_text.get(sent_id, "")
            if not sentence:
                continue

            if sentence not in all_annotations:
                all_annotations[sentence] = {
                    "sentence": sentence,
                    "tokens": sent_tokens.get(sent_id, []),
                    "language": "nl",
                    "metadata": {"file": naf_file.name},
                    "frames": [],
                }
            all_annotations[sentence]["frames"].append(frame_ann)

    annotations = list(all_annotations.values())

    for ann in annotations:
        seen_activations = set()
        deduped = []
        for fr in ann["frames"]:
            if fr["activation"] not in seen_activations:
                seen_activations.add(fr["activation"])
                deduped.append(fr)
        ann["frames"] = deduped

    annotations = [a for a in annotations if a["frames"]]

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
