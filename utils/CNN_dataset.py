import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn.functional as F
import os
import torch

log_transform = T.AmplitudeToDB(stype="power")

# Ścieżka do folderu
root_dir = './data/train'

class DynamicResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_list, target_n_mels=256):
        self.root_dir = root_dir
        self.filepaths = []
        self.labels = []
        self.target_n_mels = target_n_mels
        self.class_list = class_list

        for i, label in enumerate(class_list):
            folder = os.path.join(root_dir, label)
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith('.pt'):
                        self.filepaths.append(os.path.join(folder, file))
                        self.labels.append(i)
        
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        log_mel_spec = torch.load(self.filepaths[idx])  # (1, 256, 155)

        if self.target_n_mels != 256:
            log_mel_spec = log_mel_spec.unsqueeze(0)  # Add batch dimension
            log_mel_spec = F.interpolate(
                log_mel_spec, 
                size=(self.target_n_mels, 155), 
                mode='bilinear', 
                align_corners=False
            )
            log_mel_spec = log_mel_spec.squeeze(0)  # Remove batch dimension

        return log_mel_spec, self.labels[idx]

class PreprocessedAudioDataset(Dataset):
    def __init__(self, root_dir, class_list):
        self.root_dir = root_dir
        self.filepaths = []
        self.labels = []
        self.class_list = class_list

        for i, label in enumerate(class_list):
            folder = os.path.join(root_dir, label)
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith('.pt'):
                        self.filepaths.append(os.path.join(folder, file))
                        self.labels.append(i)
        
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        log_mel_spec = torch.load(self.filepaths[idx])
        label = self.labels[idx]
        return log_mel_spec, label

class AudioDataset(Dataset):
    def __init__(
            self, 
            root_dir, 
            class_list,
            target_len = 16000,
            n_mels = 80,  # Liczba pasm Mel
            n_fft = 400,  # Długość okna FFT
            hop_length = 160, # Przesunięcie okna (np. 160 próbek)
            ):
        self.root_dir = root_dir
        self.target_len = target_len
        self.filepaths = []
        self.labels = []
        class_list = class_list

        print(f"n_fft: {n_fft} hop_length: {hop_length} n_mels: {n_mels}")
        self.transform = T.MelSpectrogram(
            sample_rate = target_len,
            n_mels = n_mels,
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = n_fft
        )

        # Iteracja po folderach
        '''
        for idx, label in enumerate(sorted(os.listdir(root_dir))):
            if label not in class_list:
                print(idx)
                idx -= 1
                print(idx)
                continue 
            
            print(label)
        '''
        for i in range(len(class_list)):
            label = class_list[i]
            folder = os.path.join(root_dir, label)
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith('.wav'):
                        self.filepaths.append(os.path.join(folder, file))
                        self.labels.append(i)
        
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

        return log_mel_spec, self.labels[idx]