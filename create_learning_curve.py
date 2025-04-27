import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

path = "./models/2025_04_27_14:10/"

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

ax = plt.gca()  # Pobierz aktualne Axes
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # tylko całkowite wartości

plt.savefig(f"./images/learning_curve.png")


