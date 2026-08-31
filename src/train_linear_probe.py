import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

import torch


def collate_fn(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.tensor([f["labels"] for f in features])
    return {"pixel_values": pixel_values, "labels": labels}

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

ds = load_from_disk("data/dish_classifier_splits")
class_names = ds["train"].features["label"].names
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)


def transform_batch(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    processed = processor(images=images, return_tensors="pt")
    batch["pixel_values"] = processed["pixel_values"]
    batch["labels"] = batch["label"]
    return batch


ds = ds.with_transform(transform_batch)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(class_names),
    id2label=id2label,
    label2id=label2id,
)

# Freeze everything except the classifier head (our verified linear-probe setup)
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="checkpoints/linear_probe",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=1e-3,
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