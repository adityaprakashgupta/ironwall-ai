class Config:
    # XLSR-53: The 300M param beast for multilingual speech
    MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"

    SR = 16000
    MAX_DURATION = 5.0
    INPUT_LEN = int(SR * MAX_DURATION)

    # Inference
    DEPLOY_THRESHOLD = 0.58  # Needs calibration on your validation set
    SLIDING_WINDOW_STRIDE = 2.0  # Seconds

    # Training
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4  # Standard LR is safe because we freeze the backbone

    MAX_INFERENCE_AUDIO_LENGTH = 90


config = Config()
