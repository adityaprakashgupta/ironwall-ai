import numpy as np
import tensorflow as tf
import io
import os

# from model import build_secure_model, config
from data_util import process_audio_chunk
import librosa
from config import config

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def build_reason(
    final_score,
    mean_score,
    max_score,
    ai_ratio,
    strong_ratio,
    max_run_ratio,
    threshold,
    confidence,
    label
):
    try:
        final_score = float(final_score)
        mean_score = float(mean_score)
        max_score = float(max_score)
        ai_ratio = float(ai_ratio)
        strong_ratio = float(strong_ratio)
        max_run_ratio = float(max_run_ratio)
        threshold = float(threshold)
        confidence = float(confidence)
    except (ValueError, TypeError):
        return "Unable to generate a detailed explanation due to invalid or incomplete score data."

    # ---------------- AI GENERATED ----------------
    if label == "AI_GENERATED":

        # Strength description (NOT max-only)
        if strong_ratio > 0.25:
            strength = "strong and repeated indicators of neural voice synthesis"
        elif strong_ratio > 0.10:
            strength = "clear indicators of synthetic voice generation"
        else:
            strength = "moderate synthetic patterns exceeding the detection threshold"

        # Prevalence description
        if ai_ratio > 0.60:
            prevalence = "dominant across most of the audio duration"
        elif ai_ratio > 0.30:
            prevalence = "present across a significant portion of the audio"
        else:
            prevalence = "limited to specific regions of the audio"

        # Temporal behavior (THIS kills spike explanations)
        if max_run_ratio > 0.30:
            temporal = "with sustained temporal consistency"
        elif max_run_ratio > 0.15:
            temporal = "showing repeated consecutive occurrences"
        else:
            temporal = "appearing intermittently but consistently above baseline"

        return (
            f"Analysis identified {strength}, {prevalence}, {temporal}. "
            f"These patterns align with characteristics commonly observed in AI-generated speech. "
            f"The classification was made with {confidence:.0%} confidence."
        )

    # ---------------- HUMAN ----------------
    else:

        # Dominant human quality
        if ai_ratio < 0.05:
            naturality = "natural vocal dynamics including organic pitch variation and breathing patterns"
        elif ai_ratio < 0.15:
            naturality = "predominantly human speech characteristics with minimal anomalous segments"
        else:
            naturality = "human speech patterns with occasional ambiguous artifacts"

        # Stability explanation
        if max_score < threshold + 0.05:
            stability = "without any strong or persistent synthetic indicators"
        else:
            stability = "where detected anomalies were isolated and lacked temporal persistence"

        return (
            f"The audio demonstrates {naturality}, {stability}. "
            f"Overall acoustic behavior is consistent with human speech rather than neural synthesis."
        )

def max_consecutive_ratio(mask):
    max_run = run = 0
    for v in mask:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run / len(mask)

def robust_score(scores):
    N = len(scores)
    if N <= 3:
        return float(np.mean(scores))  # nothing fancy for short audio

    trim = 0.20
    k = int(N * trim)
    s = np.sort(scores)
    return float(np.mean(s[k:-k]))

class SecureInference:
    def __init__(self, model_path=None):
        if model_path:
            self.model = tf.saved_model.load(model_path)
            print(f"Loaded model from {model_path}")
        # else:
        #     self.model = build_secure_model()

    def prepare_chunks(self, audio_bytes, audio_path=None):
        if audio_bytes:
            y, _ = librosa.load(io.BytesIO(audio_bytes), sr=config.SR, mono=True)
        else:
            y, _ = librosa.load(audio_path, sr=config.SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)

        stride = int(config.SR * config.SLIDING_WINDOW_STRIDE)
        win_len = config.INPUT_LEN

        chunks_audio, chunks_mask = [], []

        if len(y) <= win_len:
            a, m = process_audio_chunk(y)
            chunks_audio.append(a)
            chunks_mask.append(m)
        else:
            for i in range(0, len(y) - win_len + 1, stride):
                chunk = y[i : i + win_len]
                a, m = process_audio_chunk(chunk)
                chunks_audio.append(a)
                chunks_mask.append(m)

            remainder = y[i + stride :]
            if len(remainder) > int(1.5 * config.SR):
                a, m = process_audio_chunk(remainder)
                chunks_audio.append(a)
                chunks_mask.append(m)

        return (
            tf.convert_to_tensor(chunks_audio),
            tf.convert_to_tensor(chunks_mask),
        )

    def predict(self, audio_path, audio_bytes=None):
        """
        Analyzes entire file using sliding windows.
        Catches "mixed" fakes where only part of the audio is AI.
        """

        audio_tensor, mask_tensor = self.prepare_chunks(audio_bytes, audio_path)

        results = self.model.serve([audio_tensor, mask_tensor])
        scores = results.numpy()
        return self.post_process(scores)

    def post_process(self, scores: np.ndarray):
        scores = scores.ravel()
        N = len(scores)

        max_score = float(scores.max())
        mean_score = float(scores.mean())

        threshold = config.DEPLOY_THRESHOLD
        margin = 0.1

        ai_mask = scores >= threshold
        strong_mask = scores >= (threshold + margin)

        ai_ratio = ai_mask.mean()
        strong_ratio = strong_mask.mean()
        run_ratio = max_consecutive_ratio(ai_mask)

        MIN_AI_RATIO = max(0.20, 1.0 / N)
        MIN_STRONG_RATIO = max(0.08, 1.0 / N)
        MIN_RUN_RATIO = max(0.15, 1.0 / N)

        is_ai = ai_ratio >= MIN_AI_RATIO and (
            strong_ratio >= MIN_STRONG_RATIO or run_ratio >= MIN_RUN_RATIO
        )

        final_score = float(np.clip(robust_score(scores), 1e-6, 1.0 - 1e-6))

        if is_ai:
            label = "AI_GENERATED"
            confidence = min(1.0, 0.4 * ai_ratio + 0.3 * strong_ratio + 0.3 * run_ratio)
        else:
            label = "HUMAN"
            confidence = min(
                1.0, 0.6 * (1 - ai_ratio) + 0.4 * (threshold - final_score) / threshold
            )

        confidence = max(0.0, min(confidence, 1.0))

        return {
            "label": label,
            "confidence": f"{confidence:.2f}",
            "score": final_score,
            "max_segment_score": max_score,
            "mean_segment_score": mean_score,
            "segments_analyzed": N,
            "explanation": build_reason(
                final_score=final_score,
                mean_score=mean_score,
                max_score=max_score,
                ai_ratio=ai_ratio,
                strong_ratio=strong_ratio,
                max_run_ratio=run_ratio,
                threshold=threshold,
                confidence=confidence,
                label=label,
            ),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Secure Audio Inference")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to the trained model"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file to analyze",
    )
    args = parser.parse_args()

    inferencer = SecureInference(model_path=args.model_path)
    result = inferencer.predict(args.audio_path)

    if "error" in result:
        print(f"Error processing audio: {result['error']}")
    else:
        print(result)
