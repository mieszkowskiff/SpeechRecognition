import glob
import matplotlib.pyplot as plt
import numpy as np

path = "./models/2025_04_27_12:56/"

# Get all files in the directory
files = glob.glob(f"{path}model_epoch_*.pth")
# Sort the files by epoch number
files.sort(key=lambda x: int(x.split("_")[-3]))
f1s = [float(filename.split("_")[-1][:-4]) for filename in files]

plt.plot(f1s)
plt.scatter(range(len(f1s)), f1s, s=10)
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Epoch")
plt.grid()
plt.savefig(f"./images/learning_curve.png")


