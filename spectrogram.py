import dataset
import torch
import matplotlib.pyplot as plt




def create_spectrogram(picture, filename):
    picture = picture.cpu().numpy().transpose()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(picture, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Mel-Spectrogram')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Mel Frequency Band')
    #fig.colorbar(cax, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def main():
    audio_dataset = dataset.AudioDataset(
        root_dir="./dataset/train",
        n_mels=64,
        n_fft=400,
        hop_length=100
        )
    picture, label = audio_dataset[12892]
    print(label)
    print(picture.shape)
    create_spectrogram(picture, filename="./images/input.png")

if __name__ == "__main__":
    main()





