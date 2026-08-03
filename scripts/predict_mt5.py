import argparse
import json
import re
from itertools import batched
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

# Any tag: captures the tag name (or "extra_id_N" for sentinel tokens)
_ANY_TAG_RE = re.compile(r"<([^>]+)>")


def parse_linearized(linearized: str, tokens: list[str]) -> list[dict]:
    """
    Parse a model-generated linearized string into prediction dicts.

    Expected format:
        <FrameName><extra_id_i>[<extra_id_j>](<RoleName><extra_id_k>[<extra_id_l>])*</>...

    Single-token spans use one <extra_id_N>; multi-token spans use two (start, end).
    Frames are delimited by </>. Malformed chunks are silently skipped.
    """
    predictions = []

    for chunk in linearized.split("</>"):
        tags = _ANY_TAG_RE.findall(chunk.strip())
        if not tags:
            continue

        frame_name = tags[0]
        i = 1

        # Activation span: consecutive extra_id_N tags right after the frame name
        act_indices = []
        while i < len(tags) and tags[i].startswith("extra_id_"):
            act_indices.append(int(tags[i].split("_")[-1]))
            i += 1

        if not act_indices:
            continue  # no activation — skip malformed frame

        act_start, act_end = act_indices[0], act_indices[-1]
        activation = " ".join(tokens[act_start : act_end + 1])

        # Roles: <RoleName> followed by one or two extra_id tags
        roles = []
        while i < len(tags):
            role_name = tags[i]
            i += 1
            role_indices = []
            while i < len(tags) and tags[i].startswith("extra_id_"):
                role_indices.append(int(tags[i].split("_")[-1]))
                i += 1
            if not role_indices:
                continue
            r_start, r_end = role_indices[0], role_indices[-1]
            roles.append(
                {
                    "name": role_name,
                    "filler": " ".join(tokens[r_start : r_end + 1]),
                    "idxs": [r_start, r_end],
                }
            )

        predictions.append(
            {
                "name": frame_name,
                "activation": activation,
                "idxs": [act_start, act_end],
                "roles": roles,
            }
        )

    return predictions


argparser = argparse.ArgumentParser()
argparser.add_argument("--model", required=True)
argparser.add_argument("--input", required=True)
argparser.add_argument("--output", required=True)
argparser.add_argument("--batch-size", type=int, default=16)
argparser.add_argument("--max-new-tokens", type=int, default=512)
argparser.add_argument("--max-len", type=int, default=256)
argparser.add_argument("--device", type=str, default="cpu")

# conda activate mfn; python predict.py --input data/datasets/iframe.test.jsonl --model models/mt5-small/checkpoint-168055/ --output /tmp/iframe.jsonl --device cuda:0

if __name__ == "__main__":
    args = argparser.parse_args()

    print(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)

    documents = [json.loads(l) for l in open(args.input).readlines()]
    print(f"Loaded {len(documents)} documents")

    batches = list(batched(documents, n=args.batch_size))
    for batch in tqdm(batches):
        batch = list(batch)
        token_lists = [d["tokens"] for d in batch]

        inputs = tokenizer(
            token_lists,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
        ).to(args.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                early_stopping=True,
            )

        # skip_special_tokens=False to preserve <extra_id_N> sentinel tokens;
        # strip pad and eos manually
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        pad, eos = tokenizer.pad_token or "", tokenizer.eos_token or ""
        decoded = [s.replace(pad, "").replace(eos, "").strip() for s in decoded]

        for doc, dec in zip(batch, decoded):
            doc["predictions"] = parse_linearized(dec, doc["tokens"])

    with open(args.output, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
