import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

path = "./models/2026_03_21_17:52/"


files = glob.glob(f"{path}model_epoch_*.pth")

files.sort(key=lambda x: int(x.split("_")[-3]))
f1s = [float(filename.split("_")[-1][:-4]) for filename in files]

plt.plot(f1s)
plt.scatter(range(len(f1s)), f1s, s=10)
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Epoch")
plt.grid()

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.savefig(f"./images/learning_curve.png")


