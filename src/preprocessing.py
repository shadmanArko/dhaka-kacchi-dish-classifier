from datasets import load_from_disk
from transformers import AutoImageProcessor

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

ds = load_from_disk("data/dish_classifier_splits")


def transform_batch(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    processed = processor(images=images, return_tensors="pt")
    batch["pixel_values"] = processed["pixel_values"]
    return batch


ds = ds.with_transform(transform_batch)

# Sanity check: pull a few examples through and confirm shape
sample = ds["train"][:4]
print("Batch pixel_values shape:", sample["pixel_values"].shape)
print("Batch labels:", sample["label"])