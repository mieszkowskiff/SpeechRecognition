from pydub import AudioSegment
from pydub.generators import WhiteNoise
import os
import random

input_dir = "./files_to_cut"
base_output_dir = "./samples"

split_ratios = {
    "train": 0.7,
    "validation": 0.2,
    "test": 0.1,
}

chunk_length_ms = 1000
overlap_ms = 500
step_ms = chunk_length_ms - overlap_ms

def enhance_chunk(chunk, gain_range_db=(2, 6), noise_amp_range=(-50, -40)):
    """
    Apply random gain and white noise to an AudioSegment.
    """
    # 1. Random volume gain
    gain_db = random.uniform(*gain_range_db)
    chunk = chunk + gain_db

    # 2. Add white noise
    noise_amp_db = random.uniform(*noise_amp_range)
    noise = WhiteNoise().to_audio_segment(duration=len(chunk), volume=noise_amp_db)
    chunk = chunk.overlay(noise)

    return chunk

# === VALIDATE RATIOS ===
assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, "Ratios must sum to 1.0"

# === PROCESS EACH FILE ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".wav"):
        continue

    file_path = os.path.join(input_dir, filename)
    file_name = os.path.splitext(filename)[0]
    output_path = os.path.join(base_output_dir, file_name)

    print(f"\n Processing: {filename}")

    # Load & convert audio
    audio = AudioSegment.from_wav(file_path).set_channels(1).set_frame_rate(16000)

    # Slice into overlapping chunks
    chunks = []
    for i in range(0, len(audio) - chunk_length_ms + 1, step_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append(chunk)

    print(f" Total chunks: {len(chunks)}")

    # Shuffle and split
    random.shuffle(chunks)
    n = len(chunks)
    train_end = int(split_ratios["train"] * n)
    val_end = train_end + int(split_ratios["validation"] * n)

    splits = {
        "train": chunks[:train_end],
        "validation": chunks[train_end:val_end],
        "test": chunks[val_end:]
    }

    # Export
    for split_name, split_chunks in splits.items():
        split_dir = os.path.join(output_path, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for idx, chunk in enumerate(split_chunks):
            enhanced_chunk = enhance_chunk(chunk)
            out_file = os.path.join(split_dir, f"{file_name}_{idx}.wav")
            enhanced_chunk.export(out_file, format="wav")

        print(f" {split_name}: {len(split_chunks)} chunks saved in {split_dir}")

print("\n All files processed!")
