import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import evaluate

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
FINETUNED_DIR = "checkpoints/finetuned"

os.makedirs("data", exist_ok=True)

subdirs = [d for d in os.listdir(FINETUNED_DIR) if d.startswith("checkpoint")]
latest_checkpoint = os.path.join(FINETUNED_DIR, sorted(subdirs)[-1])
print("Evaluating checkpoint:", latest_checkpoint)

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


trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="tmp_eval", per_device_eval_batch_size=32, remove_unused_columns=False),
    data_collator=collate_fn,
)

predictions = trainer.predict(ds["test"])
logits = predictions.predictions
true_labels = predictions.label_ids
pred_labels = np.argmax(logits, axis=1)

print("\n=== TEST SET RESULTS (final, honest number) ===")
print(classification_report(true_labels, pred_labels, target_names=class_names, digits=3))

cm = confusion_matrix(true_labels, pred_labels)
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Test Set")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=120)
print("\nSaved confusion matrix to data/confusion_matrix.png")

biryani_idx = class_names.index("Biryani")
biryani_row = cm[biryani_idx]
print(f"\nBiryani row (true label = Biryani, what it got predicted as):")
for i, count in enumerate(biryani_row):
    if count > 0:
        print(f"  Predicted as {class_names[i]:20s}: {count}")


# Find the biggest off-diagonal confusions across the whole matrix
print("\nTop confusions (true label -> predicted label, count):")
confusions = []
for true_idx in range(len(class_names)):
    for pred_idx in range(len(class_names)):
        if true_idx != pred_idx and cm[true_idx][pred_idx] > 0:
            confusions.append((cm[true_idx][pred_idx], class_names[true_idx], class_names[pred_idx]))

confusions.sort(reverse=True)
for count, true_name, pred_name in confusions[:10]:
    print(f"  {true_name:20s} -> predicted as {pred_name:20s}: {count} times")