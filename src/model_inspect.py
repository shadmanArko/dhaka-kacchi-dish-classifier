from transformers import AutoModelForImageClassification
from datasets import load_from_disk

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"

ds = load_from_disk("data/dish_classifier_splits")
class_names = ds["train"].features["label"].names

id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(class_names),
    id2label=id2label,
    label2id=label2id,
)

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

# Print just the names of the top-level modules, not the full architecture dump
for name, _ in model.named_children():
    print("Module:", name)

print("\nClassifier head:", model.classifier)

# Freeze every parameter first
for param in model.parameters():
    param.requires_grad = False

# Then unfreeze only the classifier head
for param in model.classifier.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print("\nTrainable parameters:", trainable)
print("Frozen parameters:", frozen)
print(f"Trainable %: {100 * trainable / (trainable + frozen):.4f}%")