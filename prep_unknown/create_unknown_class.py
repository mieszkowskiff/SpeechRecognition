import os
import random
import shutil
import tqdm
from math import ceil

data_root = "../kaggle/tensorflow-speech-recognition-challenge/train/train/audio"
output_root = "../dataset"
total_samples = 8000
split_ratios = {
    "train": 0.7,
    "validation": 0.2,
    "test": 0.1,
}

unknown_list = [
    "bed", "bird", "cat", "dog", "eight", "five",
    "four", "happy", "house", "marvin", "nine", "one",
    "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"
]

samples_per_class = ceil(total_samples / len(unknown_list))
collected_samples = []

print(f"Collecting ~{samples_per_class} from each of {len(unknown_list)} classes...")
print()

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
    print(f"Sampling from \"{cls}\" finished.")
    print()

random.shuffle(collected_samples)
n = len(collected_samples)
train_end = int(split_ratios["train"] * n)
val_end = train_end + int(split_ratios["validation"] * n)

out_dir = os.path.join(output_root, "unknown")
print("Writting to \"unknown\".")
try:
    os.makedirs(out_dir, exist_ok=False)

    for idx, file_path in tqdm.tqdm(enumerate(collected_samples)):
        # this approach disables analyzing if there is any subclass of "unknown" e.g.
        # "happy" that is missclassified particularly often since ale file names are unified
        # to solve it, model might be fed (not trained) the instances from only one class 
        # to observe the missclassification distribution
        new_filename = f"unknown_{idx}.wav"
        out_path = os.path.join(out_dir, new_filename)
        shutil.copy2(file_path, out_path)
    
    print("Folder \"unknown\" created successfully.")
except:
    print(f"Error during witting to folder.")
    print("Abort.")