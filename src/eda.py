import os
os.makedirs("data", exist_ok=True)

from collections import Counter
from datasets import load_dataset
import matplotlib.pyplot as plt

DATASET_ID = "SKSKCulinaryAI/Indian_Food_Images_Dataset"

ds = load_dataset(DATASET_ID)
train = ds["train"]
class_names = train.features["label"].names

# 1. Check image size distribution across a sample (checking all 6,678 would be slow)
sample = train.select(range(500))
sizes = Counter(img.size for img in sample["image"])
print("Distinct image sizes found in first 500 train images:")
for size, count in sizes.most_common(10):
    print(f"  {size}: {count} images")

# 2. Visual sample grid — one random image per class, first 12 classes
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
seen = {}
for img, label in zip(train["image"], train["label"]):
    if label not in seen and len(seen) < 12:
        seen[label] = img
    if len(seen) == 12:
        break

for ax, (label, img) in zip(axes.flat, sorted(seen.items())):
    ax.imshow(img)
    ax.set_title(class_names[label], fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("data/sample_grid.png", dpi=120)
print("\nSaved sample grid to data/sample_grid.png")

# 3. Full class distribution across the whole train split
from collections import Counter as Counter2
full_counts = Counter2(train["label"])
print("\nFull class distribution (train split):")
for label_id, count in sorted(full_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {class_names[label_id]:20s} {count}")

smallest = min(full_counts.values())
largest = max(full_counts.values())
print(f"\nSmallest class: {smallest} | Largest class: {largest} | Imbalance ratio: {largest/smallest:.1f}x")