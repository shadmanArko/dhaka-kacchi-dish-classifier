import os
from datasets import load_dataset, DatasetDict

os.makedirs("data", exist_ok=True)

DATASET_ID = "SKSKCulinaryAI/Indian_Food_Images_Dataset"
SEED = 42

ds = load_dataset(DATASET_ID)

# Split the existing 'validation' pot in half: one half becomes our real
# validation set, the other half becomes our held-out test set.
val_test_split = ds["validation"].train_test_split(
    test_size=0.2,
    stratify_by_column="label",  # keeps class proportions equal in both halves
    seed=SEED,
)

final = DatasetDict(
    train=ds["train"],
    validation=val_test_split["train"],
    test=val_test_split["test"],
)

print(final)

final.save_to_disk("data/dish_classifier_splits")
print("\nSaved final splits to data/dish_classifier_splits")