import argparse
import json
import multiprocessing
import os

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

multiprocessing.set_start_method("fork")


def load_pairs(files):
    pairs = []
    for fname in sorted(files):
        documents = [json.loads(line) for line in open(fname).readlines()]

        for d in documents:
            if len(d["tokens"]) > 100:
                continue

            frames = sorted(
                [fr for fr in d["frames"] if "activation" in fr],
                key=lambda fr: fr["idxs"][0],
            )
            if not frames:
                continue

            linearized = ""
            for fr in frames:
                linearized += f"<{fr['name']}>"
                fr_i, fr_j = fr["idxs"]

                if fr_i != fr_j:
                    linearized += f"<extra_id_{fr_i}><extra_id_{fr_j}>"
                else:
                    linearized += f"<extra_id_{fr_i}>"

                for rl in fr["roles"]:
                    linearized += f"<{rl['name']}>"
                    rl_i, rl_j = rl["idxs"]

                    if rl_i != rl_j:
                        linearized += f"<extra_id_{rl_i}><extra_id_{rl_j}>"
                    else:
                        linearized += f"<extra_id_{rl_i}>"

                linearized += "</>"
            pairs.append({"tokens": d["tokens"], "target": linearized})
    return pairs


parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--train", required=True, nargs="+")
parser.add_argument("--valid", required=True, nargs="+")
parser.add_argument("--epochs", required=True, type=int)
parser.add_argument("--micro-batch-size", type=int, default=8)
parser.add_argument("--max-len", type=int, default=256)
parser.add_argument(
    "--hf-org",
    default=None,
    help="HuggingFace organization to upload the model to (skips upload if not set)",
)
parser.add_argument("--wandb-project", default="mframeparsing")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    datasets_dir = "data/datasets"
    train_pairs = load_pairs(args.train)
    valid_pairs = load_pairs(args.valid)
    print(f"Loaded {len(train_pairs)} train / {len(valid_pairs)} valid examples")

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["tokens"],
            max_length=args.max_len,
            truncation=True,
            is_split_into_words=True,
        )
        labels = tokenizer(
            text_target=batch["target"], max_length=args.max_len, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = Dataset.from_list(train_pairs).map(
        tokenize, batched=True, remove_columns=["tokens", "target"]
    )

    valid_dataset = Dataset.from_list(valid_pairs).map(
        tokenize, batched=True, remove_columns=["tokens", "target"]
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        run_name=args.model,
        # Batching & accumulation
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=128 // args.micro_batch_size,
        gradient_checkpointing=True,
        # bf16=True,
        # Optimizer
        learning_rate=0.001,
        weight_decay=0.01,
        #dataloader_num_workers=16,
        #dataloader_persistent_workers=True,
        #dataloader_prefetch_factor=4,
        train_sampling_strategy="group_by_length",
        # Training loop — num_train_epochs is a ceiling; early stopping kicks in sooner
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        # save_total_limit=2 keeps at most 2 checkpoints; with load_best_model_at_end=True
        # HF always protects the best from deletion → we get best + latest at any time
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_strategy="steps",
        logging_steps=20,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    if args.hf_org:
        repo_id = f"{args.hf_org}/{os.path.basename(args.output)}"
        print(f"Uploading to {repo_id} ...")
        trainer.model.push_to_hub(repo_id, private=True)
        tokenizer.push_to_hub(repo_id, private=True)
        print("Upload complete.")
