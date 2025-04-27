import os
import shutil
import random
from pathlib import Path

def split_dataset(input_dir, output_dir, split_ratios=(0.7, 0.2, 0.1),
                  class_list=None, seed=42, is_balanced=True, unknown_coef=1.0):
    random.seed(seed)
    assert sum(split_ratios) == 1.0, "Sum of split ratios must be 1.0"
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    main_classes = [cls for cls in class_list if cls not in ["unknown", "background"]]
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

    for cls in selected_classes:
        print(f"  - {cls}: {class_counts[cls]} files")

    if is_balanced:
        min_count = min(class_counts[cls] for cls in main_classes)
    else:
        min_count = max(class_counts[cls] for cls in main_classes)

    if "unknown" in selected_classes:
        base_count = min_count if min_count is not None else max(class_counts[cls] for cls in main_classes)
        unknown_target_count = int(base_count * unknown_coef)
    else:
        unknown_target_count = 0

    for class_name in selected_classes:
        class_dir = input_dir / class_name
        images = list(class_dir.glob("*"))
        random.shuffle(images)

        # Określ ile plików bierzemy
        if class_name in main_classes:
            if min_count:
                images = images[:min_count]
        elif class_name == "unknown":
            images = images[:unknown_target_count]
        elif class_name == "background":

            times = (min_count // len(images)) + 1 if images else 1
            images = (images * times)[:min_count]

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

    print(f"Dataset successfully split and saved under: {output_dir}")

class_list = ["down", "up", "go", "stop", "right", "left", "no", "yes", "on", "off", "unknown", "background"]
is_balanced = False
unknown_coef = 1.0

if __name__ == "__main__":
    split_dataset(
        input_dir="./audio",
        output_dir="./dataset",
        split_ratios=(0.7, 0.2, 0.1),
        class_list=class_list,
        seed=42,
        is_balanced=is_balanced,
        unknown_coef=unknown_coef
    )
