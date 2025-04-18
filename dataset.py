import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import os
import torch

# Parametry
sample_rate = 16000
n_mels = 80
n_fft = 400  # Długość okna FFT
hop_length = 160  # Przesunięcie okna (np. 160 próbek)
win_length = n_fft  # Długość okna

# Transformacja: Mel-Spectrogram
transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=n_mels,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length
)

log_transform = T.AmplitudeToDB(stype="power")

# Ścieżka do folderu
root_dir = './data/train'

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_len=16000):
        self.root_dir = root_dir
        self.transform = transform
        self.target_len = target_len
        self.filepaths = []
        self.labels = []
        
        # Iteracja po folderach
        for idx, label in enumerate(sorted(os.listdir(root_dir))):
            folder = os.path.join(root_dir, label)
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith('.wav'):
                        self.filepaths.append(os.path.join(folder, file))
                        self.labels.append(idx)
        
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Wczytujemy plik
        waveform, sr = torchaudio.load(self.filepaths[idx])
        print(self.filepaths[idx])
        
        # Jeśli stereo, konwertujemy na mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # waveform = waveform.flip(1)

        # Upewniamy się, że długość sygnału audio to 16000
        if waveform.size(1) < self.target_len:
            # Dodajemy padding na końcu, jeśli za krótka
            pad_len = self.target_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif waveform.size(1) > self.target_len:
            # Przycinamy do 16000 próbek, jeśli za długa
            waveform = waveform[:, :self.target_len]

        # Jeśli mamy transformację, stosujemy ją
        if self.transform:
            waveform = self.transform(waveform)
        
        # Generujemy log-Mel spektrogram
        mel_spec = transform(waveform)
        log_mel_spec = log_transform(mel_spec)

        return log_mel_spec.squeeze(0).transpose(0, 1), self.labels[idx]