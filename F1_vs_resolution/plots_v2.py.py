import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load history from txt files
def load_history(file_path):
    history = []
    with open(file_path, "r") as f:
        for line in f:
            history.append([float(x) for x in line.strip().split()])
    return history

# Load data
for run in range(4):
    run_directory_path = "./run_" + str(run) + "/"
    f1_history = load_history(run_directory_path + "F1_history.txt")
    loss_history = load_history(run_directory_path + "loss_history.txt")
    time_history = load_history(run_directory_path + "time_history.txt")

    # n_mels values used during experiment
    if run==3:    
        n_mels_grid = [256, 128, 64, 32]
        colors = ['purple', 'red', 'green', 'blue']
    else:
        n_mels_grid = [32, 64, 128, 256]
        colors = ['blue', 'green', 'red', 'purple']   
    
    # Common plot settings
    def setup_plot(xlabel, ylabel, title):
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlim(left=1)
        plt.ylim(top=1.0)
        plt.tight_layout()

    run_directory_path = "./run_" + str(run) + "_"
    # Plot F1 history
    plt.figure(figsize=(10, 6))
    for i, f1_scores in enumerate(f1_history):
        plt.plot(
            range(1, len(f1_scores) + 1), f1_scores,
            label=f'n_mels = {n_mels_grid[i]}',
            color=colors[i],
            marker='o',
            markersize=4,
            linewidth=1,
            alpha=0.7
        )
    setup_plot('Epoch', 'F1 Score', 'F1 Score vs Epochs for Different Resolutions')
    plt.legend()
    plt.savefig(run_directory_path + "F1_vs_Epochs.png")
    plt.show()

    # Plot Loss history
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(loss_history):
        plt.plot(
            range(1, len(losses) + 1), losses,
            label=f'n_mels = {n_mels_grid[i]}',
            color=colors[i],
            marker='o',
            markersize=4,
            linewidth=1,
            alpha=0.7
        )
    setup_plot('Epoch', 'Training Loss', 'Training Loss vs Epochs for Different Resolutions')
    plt.legend()
    plt.savefig(run_directory_path + "Loss_vs_Epochs.png")
    plt.show()

    # Plot Time history
    plt.figure(figsize=(10, 6))
    for i, times in enumerate(time_history):
        plt.plot(
            range(1, len(times) + 1), times,
            label=f'n_mels = {n_mels_grid[i]}',
            color=colors[i],
            marker='o',
            markersize=4,
            linewidth=1,
            alpha=0.7
        )
    setup_plot('Epoch', 'Time per Epoch (seconds)', 'Time per Epoch vs Epochs for Different Resolutions')
    plt.legend()
    plt.savefig(run_directory_path + "Time_vs_Epochs.png")
    plt.show()
