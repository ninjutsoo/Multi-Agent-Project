import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

# Updated file path
input_file = "/home/hq6375/Desktop/Code/Multi-Agent-Project/batch_extractions_with_tailored.csv"
output_dir = "/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json_unbalanced"
os.makedirs(output_dir, exist_ok=True)

# Load and clean dataset
df = pd.read_csv(input_file)
df = df.dropna(subset=["Extracted", "Label"])
df["Label"] = df["Label"].astype(int)
df["text"] = df["Extracted"].astype(str)

# Step 1: Create main 128 training samples (stratified)
train_128, remaining = train_test_split(
    df,
    train_size=128,
    stratify=df["Label"],
    random_state=42
)

# Step 2: Split rest into validation and test (50/50 stratified)
val_set, test_set = train_test_split(
    remaining,
    test_size=0.5,
    stratify=remaining["Label"],
    random_state=42
)

# Save to JSONL helper
def save_jsonl(dataframe, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            json.dump({"text": row["text"], "label": row["Label"]}, f)
            f.write("\n")

# Print label distribution
def print_distribution(name, df):
    pos = (df["Label"] == 1).sum()
    neg = (df["Label"] == 0).sum()
    print(f"{name}: Total={len(df)}, Positive={pos}, Negative={neg}, Pos Ratio={pos/len(df):.2f}")

# Save validation and test sets
save_jsonl(val_set, os.path.join(output_dir, "validation.json"))
save_jsonl(test_set, os.path.join(output_dir, "test.json"))

# Report distributions
print_distribution("Train 128", train_128)
print_distribution("Validation", val_set)
print_distribution("Test", test_set)

# Step 3: Create training splits with label balancing
subset_sizes = [2, 4, 8, 16, 32, 64, 128]
total_128 = len(train_128)
pos_ratio = (train_128["Label"] == 1).sum() / total_128
neg_ratio = 1 - pos_ratio

for size in subset_sizes:
    if size == 2:
        pos_sample = train_128[train_128["Label"] == 1].sample(1, random_state=42)
        neg_sample = train_128[train_128["Label"] == 0].sample(1, random_state=42)
        subset = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)
    else:
        num_pos = round(size * pos_ratio)
        num_neg = size - num_pos

        pos_samples = train_128[train_128["Label"] == 1].sample(num_pos, random_state=42)
        neg_samples = train_128[train_128["Label"] == 0].sample(num_neg, random_state=42)
        subset = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)

    save_jsonl(subset, os.path.join(output_dir, f"train_{size}.json"))
    print_distribution(f"train_{size}", subset)

print("✅ All JSONL training/validation/test sets created in:", output_dir)


################################################################################################

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os
# import json

# # Paths
# input_csv = "/home/hq6375/Desktop/Code/Multi-Agent-Project/batch_extractions_only.csv"  # <- UPDATE path if needed
# output_dir = "/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json"
# os.makedirs(output_dir, exist_ok=True)

# # Load data
# df = pd.read_csv(input_csv)
# df = df.dropna(subset=["Extracted", "Label"])
# df["Label"] = df["Label"].astype(int)
# df["text"] = df["Extracted"].astype(str)

# # ➤ Step 1: Balanced train_128 (64 pos, 64 neg)
# pos_128 = df[df["Label"] == 1].sample(64, random_state=42)
# neg_128 = df[df["Label"] == 0].sample(64, random_state=42)
# train_128 = pd.concat([pos_128, neg_128]).sample(frac=1, random_state=42)
# remaining = df.drop(train_128.index)

# # ➤ Step 2: Stratified validation/test split
# val_set, test_set = train_test_split(
#     remaining,
#     test_size=0.5,
#     stratify=remaining["Label"],
#     random_state=42
# )

# # ➤ Save JSON format compatible with your training code
# def save_json(df_subset, path):
#     data = [
#         {"text": row["text"], "label": "yes" if row["Label"] == 1 else "no"}
#         for _, row in df_subset.iterrows()
#     ]
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)

# # Save val/test sets
# save_json(val_set, os.path.join(output_dir, "validation.json"))
# save_json(test_set, os.path.join(output_dir, "test.json"))

# # ➤ Step 3: Create balanced training splits
# subset_sizes = [2, 4, 8, 16, 32, 64, 128]
# for size in subset_sizes:
#     if size == 2:
#         pos = train_128[train_128["Label"] == 1].sample(1, random_state=42)
#         neg = train_128[train_128["Label"] == 0].sample(1, random_state=42)
#         split = pd.concat([pos, neg]).sample(frac=1, random_state=42)
#     else:
#         half = size // 2
#         pos = train_128[train_128["Label"] == 1].sample(half, random_state=42)
#         neg = train_128[train_128["Label"] == 0].sample(half, random_state=42)
#         split = pd.concat([pos, neg]).sample(frac=1, random_state=42)
#     save_json(split, os.path.join(output_dir, f"train_{size}.json"))

# print("✅ All splits created in:", output_dir)

################################################################################################


# import pandas as pd
# import os
# import json

# # Updated file path
# input_file = "/home/hq6375/Desktop/Code/Multi-Agent-Project/batch_extractions_with_tailored.csv"
# output_path = "/home/hq6375/Desktop/Code/Multi-Agent-Project/all_400.json"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)

# # Load and clean dataset
# df = pd.read_csv(input_file)
# df = df.dropna(subset=["Extracted", "Label"])
# df["Label"] = df["Label"].astype(int)
# df["text"] = df["Extracted"].astype(str)

# # Save all 400 samples to one JSONL file
# with open(output_path, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         json.dump({"text": row["text"], "label": row["Label"]}, f)
#         f.write("\n")

# print(f"✅ All 400 samples saved to: {output_path}")


################################################################################################
#Random seed data split

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

# Updated file path
input_file = "/home/hq6375/Desktop/Code/Multi-Agent-Project/batch_extractions_with_tailored.csv"
output_dir = "/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json_unbalanced"
os.makedirs(output_dir, exist_ok=True)

# Seeds to use
seeds = [0, 21, 42, 1337, 1024]

# Load and clean dataset
df = pd.read_csv(input_file)
df = df.dropna(subset=["Extracted", "Label"])
df["Label"] = df["Label"].astype(int)
df["text"] = df["Extracted"].astype(str)

# Save to JSONL helper
def save_jsonl(dataframe, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            json.dump({"text": row["text"], "label": row["Label"]}, f)
            f.write("\n")

# Print label distribution
def print_distribution(name, df):
    pos = (df["Label"] == 1).sum()
    neg = (df["Label"] == 0).sum()
    print(f"{name}: Total={len(df)}, Positive={pos}, Negative={neg}, Pos Ratio={pos/len(df):.2f}")

# Step 1: Create validation and test splits (shared across seeds)
train_full, remaining = train_test_split(
    df,
    train_size=128,
    stratify=df["Label"],
    random_state=42  # fixed for consistent val/test
)
val_set, test_set = train_test_split(
    remaining,
    test_size=0.5,
    stratify=remaining["Label"],
    random_state=42
)
save_jsonl(val_set, os.path.join(output_dir, "validation.json"))
save_jsonl(test_set, os.path.join(output_dir, "test.json"))
print_distribution("Validation", val_set)
print_distribution("Test", test_set)

# Step 2: Generate all seed-based training splits
subset_sizes = [2, 4, 8, 16, 32, 64, 128]

for seed in seeds:
    train_seed = train_full.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    for size in subset_sizes:
        if size == 2:
            pos_sample = train_seed[train_seed["Label"] == 1].sample(1, random_state=seed)
            neg_sample = train_seed[train_seed["Label"] == 0].sample(1, random_state=seed)
            subset = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed)
        else:
            pos_ratio = (train_seed["Label"] == 1).sum() / len(train_seed)
            num_pos = round(size * pos_ratio)
            num_neg = size - num_pos

            pos_samples = train_seed[train_seed["Label"] == 1].sample(num_pos, random_state=seed)
            neg_samples = train_seed[train_seed["Label"] == 0].sample(num_neg, random_state=seed)
            subset = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=seed)

        filename = f"train_{size}_seed{seed}.json"
        save_jsonl(subset, os.path.join(output_dir, filename))
        print_distribution(filename, subset)

print("✅ All seed-based shot splits saved in:", output_dir)

