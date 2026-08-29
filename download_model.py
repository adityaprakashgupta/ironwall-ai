from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "adityaprakashgupta/voice"
MODEL_DIR = Path("./model")


def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}...")
    print(f"Destination: {MODEL_DIR.resolve()}")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(MODEL_DIR),
    )

    print("\nModel downloaded successfully!")
    print(f"Location: {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    download_model()
