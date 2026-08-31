import os
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
FINETUNED_DIR = "checkpoints/finetuned"
EXPORT_DIR = "model_export"

subdirs = [d for d in os.listdir(FINETUNED_DIR) if d.startswith("checkpoint")]
latest_checkpoint = os.path.join(FINETUNED_DIR, sorted(subdirs)[-1])
print("Exporting from:", latest_checkpoint)

model = AutoModelForImageClassification.from_pretrained(latest_checkpoint)
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

os.makedirs(EXPORT_DIR, exist_ok=True)
model.save_pretrained(EXPORT_DIR)
processor.save_pretrained(EXPORT_DIR)

print(f"\nExported clean model + processor to {EXPORT_DIR}/")
print("Contents:", os.listdir(EXPORT_DIR))