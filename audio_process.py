import os
import glob
import numpy as np
import librosa
import soundfile as sf
import math
from tqdm import tqdm
import concurrent.futures

# Using 5 seconds as the target duration
TARGET_DURATION = 5
TARGET_SR = 16000
TARGET_SAMPLES = int(TARGET_DURATION * TARGET_SR)


def process_audio_chunk(y, target_len=TARGET_SAMPLES):
    """
    Core logic to normalize and mask a raw audio array.
    Used by both training (file loader) and inference (sliding window).
    """
    # --- FIX 1 (Normalization) ---
    # Safe normalization to [-1, 1] range
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + 1e-9)

    # Pad or Crop
    if len(y) > target_len:
        # Center crop
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
        mask = np.ones(target_len, dtype=np.int32)
    else:
        # --- FIX 2 (Attention Masking) ---
        pad_len = target_len - len(y)
        mask = np.concatenate([np.ones(len(y)), np.zeros(pad_len)])
        y = np.pad(y, (0, pad_len), mode="constant")

    return y.astype(np.float32), mask.astype(np.int32)


def process_audio_folder(source_dir, label):
    """
    Process all audio files in source_dir and save 5-second chunks to dataset/[label].

    Args:
        source_dir (str): Path to folder containing source audio files.
        label (str): Subfolder name in 'dataset/' to save processed files (e.g., 'real' or 'fake').
    """
    # 1. Ensure output directory exists
    output_dir = os.path.join("dataset", label)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Find all files (filtering commonly supported extensions if needed, but 'any' requested)
    # We'll use a recursive glob or just listdir. Let's stick to common audio extensions to avoid failures.
    valid_exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    files = []
    for ext in valid_exts:
        files.extend(glob.glob(os.path.join(source_dir, ext), recursive=True))
        # Also try lowercase/uppercase if glob isn't case sensitive on platform, holding pattern is fine.

    # Fallback: if glob didn't find much or user implies *any* file, iterate listdir
    # A cleaner way for "any format" is to let librosa try to open it.
    all_files_in_dir = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
    ]

    print(f"Found {len(all_files_in_dir)} files in {source_dir}. Processing...")

    for file_path in tqdm(all_files_in_dir):
        try:
            # 3. Load audio
            # sr=16000 to match project standard
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

            # Trim silence to ensure we have useful signal
            y, _ = librosa.effects.trim(y)

            if len(y) == 0:
                print(f"Skipping empty audio: {file_path}")
                continue

            # 4. Handle Duration Logic

            # Logic for Short Audio (< 5s)
            if len(y) < TARGET_SAMPLES:
                # Repeat until it fills the target length
                # num_repeats = math.ceil(TARGET_SAMPLES / len(y))
                # y_repeated = np.tile(y, num_repeats)
                # Trim to exactly 5 seconds
                y_final = y

                # Save
                save_chunk(y_final, output_dir, file_path, 0)

            else:
                # Logic for Long Audio (>= 5s)
                # Split into 5s chunks
                num_chunks = math.ceil(len(y) / TARGET_SAMPLES)

                for i in range(num_chunks):
                    start = i * TARGET_SAMPLES
                    end = start + TARGET_SAMPLES

                    chunk = y[start:end]

                    # Check if last chunk is shorter than target
                    # if len(chunk) < TARGET_SAMPLES:
                    #     # Leftover last chunk: Repeat it to fill 5s
                    #     num_repeats = math.ceil(TARGET_SAMPLES / len(chunk))
                    #     chunk_repeated = np.tile(chunk, num_repeats)
                    #     chunk = chunk_repeated[:TARGET_SAMPLES]

                    # Save each chunk
                    save_chunk(chunk, output_dir, file_path, i)

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")


import uuid


def save_chunk(audio_data, output_dir, original_path, chunk_index):
    """Helper to save the audio chunk."""
    # base_name = os.path.splitext(os.path.basename(original_path))[0]
    # out_filename = f"{base_name}_chunk_{chunk_index}.wav"

    # Requirement: New chunks file name must be uuid.wav
    out_filename = f"{uuid.uuid4()}.wav"
    out_path = os.path.join(output_dir, out_filename)

    sf.write(out_path, audio_data, TARGET_SR)
    # print(f"Saved: {out_path}")


def cache_split(files, label, cache_dir):
    out_dir = os.path.join(cache_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    def process_file(path):
        try:
            y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            y, _ = librosa.effects.trim(y, top_db=20)
            audio, mask = process_audio_chunk(y)

            name = os.path.basename(path).replace(".wav", ".npy")
            np.save(os.path.join(out_dir, name), np.stack([audio, mask]))
        except Exception as e:
            print("Skip:", path, e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(
            tqdm(
                executor.map(process_file, files),
                total=len(files),
                desc=f"Caching {label}",
            )
        )


def count_total_hours_audio(directory):
    def get_duration(file_name):
        y, _ = librosa.load(os.path.join(directory, file_name), sr=TARGET_SR)
        return len(y) / (TARGET_SR * 3600)

    files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        durations = list(tqdm(executor.map(get_duration, files), total=len(files)))

    return sum(durations)


if __name__ == "__main__":
    # Example Usage
    # You can change these paths to test manually
    # path_label = [
    #     ("te_in_female", "real"),
    #     ("te_in_male", "real"),
    # ]
    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #     futures = [executor.submit(process_audio_folder, path, label) for path, label in path_label]
    #     for future in concurrent.futures.as_completed(futures):
    #         future.result()
    # real_count = count_total_hours_audio("dataset\\real")
    # fake_count = count_total_hours_audio("dataset\\fake")
    # print(f"Real count: {real_count}")
    # print(f"Fake count: {fake_count}")
    # print(f"Total count: {real_count + fake_count}")

    fake_files = glob.glob("dataset\\fake\\*.wav")
    real_files = glob.glob("dataset\\real\\*.wav")

    cache_split(fake_files, "fake", "cache")
    cache_split(real_files, "real", "cache")
