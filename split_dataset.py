import os
import shutil
import random
from pathlib import Path

def split_dataset(input_dir, output_dir, split_ratios=(0.7, 0.2, 0.1), class_list=None, seed=42):
    random.seed(seed)
    assert sum(split_ratios) == 1.0, "Sum of split ratios must be 1.0"
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    selected_classes = []
    class_counts = {}

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_list and class_name not in class_list:
            continue

        images = list(class_dir.glob("*"))
        class_counts[class_name] = len(images)
        selected_classes.append(class_name)

    equal_counts = len(set(class_counts.values())) == 1
    if not equal_counts:
        print("Classes have different counts of files. The dataset will be balanced.")
        for cls in selected_classes:
            print(f"  - {cls}: {class_counts[cls]} files")

    min_count = min(class_counts.values())

    for class_name in selected_classes:
        class_dir = input_dir / class_name
        images = list(class_dir.glob("*"))
        random.shuffle(images)

        if not equal_counts:
            images = images[:min_count]

        total = len(images)
        train_end = int(split_ratios[0] * total)
        val_end = train_end + int(split_ratios[1] * total)

        splits = {
            "train": images[:train_end],
            "valid": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_files in splits.items():
            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for file_path in split_files:
                shutil.copy(file_path, split_class_dir / file_path.name)


    print(f"Dataset successfully split. Saved under: {output_dir}")



if __name__ == "__main__":
    split_dataset(
        input_dir = "./speech_commands",
        output_dir = "./dataset",
        split_ratios = (0.7, 0.2, 0.1),
        class_list = ["yes", "no"],
    )


