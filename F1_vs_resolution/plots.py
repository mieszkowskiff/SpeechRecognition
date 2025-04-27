import matplotlib.pyplot as plt

# Load history from txt files
def load_history(file_path):
    history = []
    with open(file_path, "r") as f:
        for line in f:
            history.append([float(x) for x in line.strip().split()])
    return history

# Load data
f1_history = load_history("F1_history.txt")
loss_history = load_history("loss_history.txt")
time_history = load_history("time_history.txt")

# n_mels values used during experiment
n_mels_grid = [256, 128, 64, 32]

# Plot settings
colors = ['blue', 'green', 'red', 'purple']

# Plot F1 history
plt.figure(figsize=(10, 6))
for i, f1_scores in enumerate(f1_history):
    plt.plot(f1_scores, label=f'n_mels = {n_mels_grid[i]}', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Epochs for Different Resolutions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("F1_vs_Epochs.png")
plt.show()

# Plot Loss history
plt.figure(figsize=(10, 6))
for i, losses in enumerate(loss_history):
    plt.plot(losses, label=f'n_mels = {n_mels_grid[i]}', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epochs for Different Resolutions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Loss_vs_Epochs.png")
plt.show()

# Plot Time history
plt.figure(figsize=(10, 6))
for i, times in enumerate(time_history):
    plt.plot(times, label=f'n_mels = {n_mels_grid[i]}', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Time per Epoch (seconds)')
plt.title('Time per Epoch vs Epochs for Different Resolutions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Time_vs_Epochs.png")
plt.show()
