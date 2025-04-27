import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import os
import torch

log_transform = T.AmplitudeToDB(stype="power")

# Ścieżka do folderu
root_dir = './data/train'

class AudioDataset(Dataset):
    def __init__(
            self, 
            root_dir, 
            target_len = 16000,
            n_mels = 80,  # Liczba pasm Mel
            n_fft = 400,  # Długość okna FFT
            hop_length = 160, # Przesunięcie okna (np. 160 próbek)
            ):
        self.root_dir = root_dir
        self.target_len = target_len
        self.filepaths = []
        self.labels = []

        print(f"n_fft: {n_fft} hop_length: {hop_length} n_mels: {n_mels}")
        self.transform = T.MelSpectrogram(
            sample_rate = target_len,
            n_mels = n_mels,
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = n_fft
        )

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
        
        # Generujemy log-Mel spektrogram
        mel_spec = self.transform(waveform)
        log_mel_spec = log_transform(mel_spec)

        return log_mel_spec.squeeze(0).transpose(0, 1), self.labels[idx]
    
def get_observation(
        path,
        n_mels = 64,
        n_fft = 400,
        hop_length = 100
    ):
    waveform, sr = torchaudio.load(path)

        
        # Jeśli stereo, konwertujemy na mono
    if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    transform = T.MelSpectrogram(
        sample_rate = sr,
        n_mels = n_mels,
        n_fft = n_fft,
        hop_length = hop_length,
        win_length = n_fft
    )
    mel_spec = transform(waveform)
    log_mel_spec = log_transform(mel_spec)
    return log_mel_spec.squeeze(0).transpose(0, 1)

