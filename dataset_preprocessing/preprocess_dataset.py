import os
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# Original and target directories
SOURCE_DIR = '../dataset'
TARGET_DIR = '../dataset_high_res_preprocessed'

# Parameters for MelSpectrogram
TARGET_LEN = 16000
N_MELS = 256
N_FFT =512
HOP_LENGTH = 100
'''
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
'''

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_LEN,  # Assuming 16kHz audio
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=N_FFT
)

log_transform = T.AmplitudeToDB(stype='power')

def process_split(split_name):
    print(f'Processing split: {split_name}')
    split_src = os.path.join(SOURCE_DIR, split_name)
    split_dst = os.path.join(TARGET_DIR, split_name)

    for label in os.listdir(split_src):
        label_src = os.path.join(split_src, label)
        label_dst = os.path.join(split_dst, label)

        if not os.path.isdir(label_src):
            continue

        os.makedirs(label_dst, exist_ok=True)

        for file in tqdm(os.listdir(label_src), desc=f"{split_name}/{label}"):
            if not file.endswith('.wav'):
                continue

            file_src = os.path.join(label_src, file)
            file_dst = os.path.join(label_dst, file.replace('.wav', '.pt'))

            # Load waveform
            waveform, sr = torchaudio.load(file_src)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Pad or trim
            if waveform.size(1) < TARGET_LEN:
                pad_len = TARGET_LEN - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            elif waveform.size(1) > TARGET_LEN:
                waveform = waveform[:, :TARGET_LEN]

            # Compute Mel Spectrogram
            mel_spec = mel_transform(waveform)
            log_mel_spec = log_transform(mel_spec)

            # Save preprocessed tensor
            torch.save(log_mel_spec, file_dst)

if __name__ == "__main__":
    os.makedirs(TARGET_DIR, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        process_split(split)

    print("All splits processed and saved into preprocessed_dataset!")
