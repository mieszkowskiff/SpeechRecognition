import os
import random
import shutil
from math import ceil

data_root = "../speech_commands_v0.02"
output_root = "./"
total_samples = 8000
split_ratios = {
    "train": 0.7,
    "validation": 0.2,
    "test": 0.1,
}

unknown_list = [
    "backward", "bed", "bird", "cat", "dog", "eight", "five", "follow", "forward",
    "four", "happy", "house", "learn", "marvin", "nine", "off", "on", "one",
    "seven", "sheila", "six", "three", "tree", "two", "visual", "wow", "zero"
]

samples_per_class = ceil(total_samples / len(unknown_list))
collected_samples = []

print(f"Collecting ~{samples_per_class} from each of {len(unknown_list)} classes...")

for cls in unknown_list:
    cls_path = os.path.join(data_root, cls)
    if not os.path.isdir(cls_path):
        print(f"Skipping missing class folder: {cls_path}")
        continue

    all_files = [f for f in os.listdir(cls_path) if f.endswith(".wav")]
    random.shuffle(all_files)
    selected = all_files[:samples_per_class]

    for file in selected:
        full_path = os.path.join(cls_path, file)
        collected_samples.append(full_path)

random.shuffle(collected_samples)
n = len(collected_samples)
train_end = int(split_ratios["train"] * n)
val_end = train_end + int(split_ratios["validation"] * n)

splits = {
    "train": collected_samples[:train_end],
    "validation": collected_samples[train_end:val_end],
    "test": collected_samples[val_end:]
}

for split, files in splits.items():
    out_dir = os.path.join(output_root, split)
    os.makedirs(out_dir, exist_ok=True)

    for idx, file_path in enumerate(files):
        new_filename = f"unknown_{idx}.wav"
        out_path = os.path.join(out_dir, new_filename)
        shutil.copy2(file_path, out_path)

    print(f"{split.capitalize()}: {len(files)} files saved to {out_dir}")

print("\n Unknown class creation complete!")
