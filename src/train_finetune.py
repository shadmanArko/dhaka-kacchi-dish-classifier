import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
LINEAR_PROBE_DIR = "checkpoints/linear_probe"

# Load the same processor we used for the linear probe (its config never changes,
# it's tied to the original checkpoint, not to our trained weights)
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

ds = load_from_disk("data/dish_classifier_splits")
class_names = ds["train"].features["label"].names


def transform_batch(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    processed = processor(images=images, return_tensors="pt")
    batch["pixel_values"] = processed["pixel_values"]
    batch["labels"] = batch["label"]
    return batch


ds = ds.with_transform(transform_batch)

# Find the verified linear-probe checkpoint and load FROM there, not from
# the original pretrained checkpoint — we keep the already-trained head.
subdirs = [d for d in os.listdir(LINEAR_PROBE_DIR) if d.startswith("checkpoint")]
latest_probe_checkpoint = os.path.join(LINEAR_PROBE_DIR, sorted(subdirs)[-1])
print("Starting fine-tuning from:", latest_probe_checkpoint)

model = AutoModelForImageClassification.from_pretrained(latest_probe_checkpoint)

# Unfreeze EVERYTHING this time — the whole point of fine-tuning
for param in model.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters this run: {trainable}")


def collate_fn(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.tensor([f["labels"] for f in features])
    return {"pixel_values": pixel_values, "labels": labels}


accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="checkpoints/finetuned",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=20,
    learning_rate=2e-5,          # much smaller than the linear probe's 1e-3
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

if __name__ == "__main__":
    trainer.train()