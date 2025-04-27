import os
import shutil
import uuid


unknown_commands = ["bed", "bird", "cat", "dog", "eight", "five", "four", "go", "happy", "house",
                   "marvin", "nine", "one", "seven", "sheila", "six",
                   "three", "tree", "two", "wow"]


source_base = "./audio/"
destination_folder = os.path.join(source_base, "unknown")

os.makedirs(destination_folder, exist_ok=True)

for command in unknown_commands:
    folder_path = os.path.join(source_base, command)
    if not os.path.isdir(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping...")
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            source_file = os.path.join(folder_path, filename)
            
            unique_name = f"{command}_{uuid.uuid4().hex}.wav"
            destination_file = os.path.join(destination_folder, unique_name)
            
            shutil.copy(source_file, destination_file)

print("Finished moving files.")