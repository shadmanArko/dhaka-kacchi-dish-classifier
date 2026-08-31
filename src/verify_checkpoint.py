import numpy as np
from datasets import load_from_disk
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
import evaluate
import torch
import os

CHECKPOINT_DIR = "checkpoints/finetuned"
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

# Find the actual checkpoint subfolder (e.g. checkpoints/linear_probe/checkpoint-XXX)
subdirs = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint")]
latest_checkpoint = os.path.join(CHECKPOINT_DIR, sorted(subdirs)[-1])
print("Loading from:", latest_checkpoint)

ds = load_from_disk("data/dish_classifier_splits")
class_names = ds["train"].features["label"].names
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

def transform_batch(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    processed = processor(images=images, return_tensors="pt")
    batch["pixel_values"] = processed["pixel_values"]
    batch["labels"] = batch["label"]
    return batch

ds = ds.with_transform(transform_batch)

model = AutoModelForImageClassification.from_pretrained(latest_checkpoint)

def collate_fn(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.tensor([f["labels"] for f in features])
    return {"pixel_values": pixel_values, "labels": labels}

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="tmp_eval", per_device_eval_batch_size=32, remove_unused_columns=False),
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

results = trainer.evaluate()
print("\nReloaded model validation accuracy:", results["eval_accuracy"])