import numpy as np
import librosa
from config import config
import tensorflow as tf

def process_audio_chunk(y, target_len=config.INPUT_LEN):
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
        y = np.pad(y, (0, pad_len), mode='constant')
        
    return y.astype(np.float32), mask.astype(np.int32)


def preprocess_from_file(path):
    """Wrapper to load from file"""
    try:
        y, _ = librosa.load(path, sr=config.SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        return process_audio_chunk(y)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return np.zeros(config.INPUT_LEN), np.zeros(config.INPUT_LEN)



class DataGenerator(tf.keras.utils.Sequence):
    """Data Generator for Training"""
    def __init__(self, file_paths, labels, batch_size=config.BATCH_SIZE, training=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.training = training
        self.indices = np.arange(len(file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_audio, X_mask, y = [], [], []
        
        for i in indices:
            audio, mask = preprocess_from_file(self.file_paths[i])
            
            # SpecAugment-style Time Masking
            if self.training and np.random.rand() < 0.3:
                t = np.random.randint(0, len(audio) - 400)
                audio[t:t+400] = 0
                
            X_audio.append(audio)
            X_mask.append(mask)
            y.append(self.labels[i])
            
        return {"audio_in": np.array(X_audio), "mask_in": np.array(X_mask)}, np.array(y)

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indices)
