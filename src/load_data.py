from datasets import load_dataset

DATASET_ID = "SKSKCulinaryAI/Indian_Food_Images_Dataset"

ds = load_dataset(DATASET_ID)

print(ds)
print("\nFeatures:", ds["train"].features)
print("\nFirst example (raw):", ds["train"][0])