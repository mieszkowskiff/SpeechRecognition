import dataset
import torch
import matplotlib.pyplot as plt




def create_spectrogram(picture):
    picture = picture.numpy().transpose()
    plt.imshow(picture, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Band')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

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
    create_spectrogram(picture)

if __name__ == "__main__":
    main()





