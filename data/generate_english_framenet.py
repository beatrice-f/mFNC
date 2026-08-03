# code adapted from https://github.com/chanind/frame-semantic-transformer
import json
from argparse import ArgumentParser
from pathlib import Path

import nltk
import numpy as np

nltk.download("framenet_v17")
from nltk.corpus import framenet as fn
from sacremoses import MosesDetokenizer
from skmultilearn.model_selection import IterativeStratification

# from https://github.com/swabhs/open-sesame/blob/master/sesame/globalconfig.py#L43-L78
TEST_FILES = [
    "ANC__110CYL067.xml",
    "ANC__110CYL069.xml",
    "ANC__112C-L013.xml",
    "ANC__IntroHongKong.xml",
    "ANC__StephanopoulosCrimes.xml",
    "ANC__WhereToHongKong.xml",
    "KBEval__atm.xml",
    "KBEval__Brandeis.xml",
    "KBEval__cycorp.xml",
    "KBEval__parc.xml",
    "KBEval__Stanford.xml",
    "KBEval__utd-icsi.xml",
    "LUCorpus-v0.3__20000410_nyt-NEW.xml",
    "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
    "LUCorpus-v0.3__enron-thread-159550.xml",
    "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
    "LUCorpus-v0.3__SNO-525.xml",
    "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
    "Miscellaneous__Hound-Ch14.xml",
    "Miscellaneous__SadatAssassination.xml",
    "NTI__NorthKorea_Introduction.xml",
    "NTI__Syria_NuclearOverview.xml",
    "PropBank__AetnaLifeAndCasualty.xml",
]

DEV_FILES = [
    "ANC__110CYL072.xml",
    "KBEval__MIT.xml",
    "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
    "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
    "Miscellaneous__Hijack.xml",
    "NTI__NorthKorea_NuclearOverview.xml",
    "NTI__WMDNews_062606.xml",
    "PropBank__TicketSplitting.xml",
]

argparser = ArgumentParser()
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

    detok = MosesDetokenizer(lang="en")

    allowed_frames = [
        l.strip() for l in (Path(__file__).parent / "frames.txt").open().readlines()
    ]

    annotations = []
    total_frames = []
    for doc in fn.docs():
        for sentence in doc["sentence"]:
            sentence_text = sentence["text"]

            # extract tokens first
            tokens = []
            token_spans = []
            for aset in sentence["annotationSet"]:
                for layer in aset["layer"]:
                    if layer["name"] == "PENN":
                        for label in sorted(layer["label"], key=lambda l: l["start"]):
                            i, j = label["start"], label["end"] + 1
                            token = sentence_text[i:j]
                            tokens.append(token)
                            token_spans.append((i, j))

            ann = {
                "sentence": detok.detokenize(tokens),
                "tokens": tokens,
                "language": "en",
                "metadata": {
                    "doc": doc.filename,
                    "corpus": doc.corpname,
                    "description": doc.description,
                },
                "frames": [],
            }

            for annotation in sentence["annotationSet"]:
                if (
                    "FE" in annotation
                    and "Target" in annotation
                    and "frame" in annotation
                ):
                    total_frames.append(annotation["frame"]["name"])
                    char_start, char_end = [loc for loc in annotation["Target"]][0]
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

                    roles = []
                    seen_role_names = set()
                    for fe_layer in annotation["FE"]:
                        if not isinstance(fe_layer, list):
                            # FE[1] is a dict of null-instantiation markers {name: itype}
                            continue
                        for start_i, end_i, role in fe_layer:
                            if role in seen_role_names:
                                continue
                            filler_token_idxs = [
                                k
                                for k, (ts, te) in enumerate(token_spans)
                                if ts >= start_i and te <= end_i
                            ]

                            if len(filler_token_idxs) == 0:
                                continue

                            f_i, f_j = min(filler_token_idxs), max(filler_token_idxs)
                            filler = detok.detokenize(tokens[f_i : f_j + 1]).strip()
                            if not filler:
                                continue
                            seen_role_names.add(role)
                            roles.append(
                                {
                                    "name": role,
                                    "filler": filler,
                                    "idxs": (f_i, f_j),
                                }
                            )

                    if not roles:
                        continue

                    ann["frames"].append({
                        "idxs": (i, j),
                        "activation": activation,
                        "name": annotation["frame"]["name"],
                        "roles": roles,
                    })

            if len(ann["frames"]) > 0:
                annotations.append(ann)

    print(len(set(total_frames)))

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

    annotations = merge_and_dedup(annotations)

    ann_train, ann_valid, ann_test = [], [], []
    for ann in annotations:
        source_doc = ann["metadata"]["doc"]
        if source_doc in TEST_FILES:
            ann_test.append(ann)
        elif source_doc in DEV_FILES:
            ann_valid.append(ann)
        else:
            ann_train.append(ann)

    for anns, anns_label in zip(
        [ann_train, ann_valid, ann_test],
        ["train", "valid", "test"],
    ):
        # export as jsonl file
        with open(f"{args.output}.{anns_label}.jsonl", "w") as f:
            for a in anns:
                if len(a["frames"]) > 0:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
