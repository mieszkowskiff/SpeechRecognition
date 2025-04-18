import dataset
import torch
import matplotlib.pyplot as plt


audio_dataset = dataset.AudioDataset(root_dir="./dataset/train", transform=None)

picture, label = audio_dataset[1]
print(label)
print(picture.shape)
picture = picture.numpy().transpose()




plt.imshow(picture, aspect='auto', origin='lower', cmap='viridis')

# Dodawanie tytułu, etykiet osi
plt.title('Mel-Spectrogram')
plt.xlabel('Time Frame')  # Czas w jednostkach klatek
plt.ylabel('Mel Frequency Band')  # Pasma Mel (częstotliwości)

# Dodanie paska kolorów
plt.colorbar(format='%+2.0f dB')

# Pokazanie wykresu
plt.show()