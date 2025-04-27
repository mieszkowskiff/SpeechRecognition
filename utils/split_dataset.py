import os
import shutil
import random
from pathlib import Path

def split_dataset(input_dir, output_dir, split_ratios=(0.7, 0.2, 0.1), class_list=None, seed=42):
    random.seed(seed)
    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "valid"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    except:
        print(f"Error during creating the folders.")
        print("Abort.")
        return 0

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    selected_classes = []
    class_counts = {}

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_list and (class_name not in class_list):
            continue

        print(class_dir)
        samples = list(class_dir.glob("*"))
        class_counts[class_name] = len(samples)
        selected_classes.append(class_name)

    equal_counts = (len(set(class_counts.values())) == 1)
    if not equal_counts:
        print("Classes have different counts of files. The dataset will be balanced.")
        for cls in selected_classes:
            print(f"  - {cls}: {class_counts[cls]} files")

    # leave it as it is
    #min_count = min(class_counts.values())

    for class_name in selected_classes:
        class_dir = input_dir / class_name
        samples = list(class_dir.glob("*"))
        random.shuffle(samples)

        #ignore imbalanced
        '''
        if not equal_counts:
            samples = samples[:min_count]
        '''

        total = len(samples)
        train_end = int(split_ratios[0] * total)
        val_end = train_end + int(split_ratios[1] * total)

        splits = {
            "train": samples[:train_end],
            "valid": samples[train_end:val_end],
            "test": samples[val_end:]
        }

        for split_name, split_files in splits.items():
            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for file_path in split_files:
                shutil.copy(file_path, split_class_dir / file_path.name)


    print(f"Dataset successfully split. Saved under: {output_dir}")

class_list = ["down", "up", "go", "stop", "right", "left", "no", "yes", "on", "off", "unknown", "background"]

if __name__ == "__main__":
    split_dataset(
        input_dir = "../dataset_grouped",
        output_dir = "../dataset",
        split_ratios = (0.7, 0.2, 0.1),
        class_list = class_list,
        seed = 42
    )