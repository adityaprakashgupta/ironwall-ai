# 🛡️ IronWall AI - Audio Deepfake Detection

> **Defending Truth in the Age of Generative AI**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)

## 💡 Inspiration
With the rapid advancement of generative AI, deepfakes have become indistinguishable from reality. From political misinformation to financial scams (like the "CEO voice scam"), synthetic audio poses a massive security threat. We built **IronWall AI** to provided a robust, real-time defense layer against AI-generated voice attacks.

## 🚀 What it does
IronWall AI is a high-performance deepfake detection system that analyzes audio streams to determine if a voice is **Real** or **AI-Generated**.

Key capabilities:
- **High Accuracy Detection:** Uses a custom deep learning model trained on thousands of fake and real audio samples.
- **Micro-Analysis:** Splits audio into small chunks to detect subtle artifacts left by vocoders and neural synthesis engines.
- **Detailed Reasoning:** Doesn't just give a score; it explains *why* (e.g., "strong indicators of neural synthesis").
- **Language Agnostic:** Optimized for **English, Hindi, Tamil, Telugu, and Malayalam**, making it effective across diverse linguistic regions.
- **Secure API:** Built with FastAPI and API Key authentication for enterprise-grade security.

## ⚙️ How we built it
We leveraged a modern tech stack to build a scalable and efficient detection pipeline:

- **Core Model:** Built with **TensorFlow/Keras**. The model analyzes mel-spectrogram features to identify synthetic patterns.
- **Audio Processing:** **Librosa** is used for feature extraction and signal processing. We implemented a custom chunking algorithm to process long audio files without memory overflows.
- **Backend API:** **FastAPI** provides a high-performance asynchronous server. We use `asyncio` to handle concurrent inference requests efficiently.
- **Deployment:** The entire application is containerized using **Docker** for consistent deployment across environments.

## 🧠 Challenges we ran into
- **Latency vs. Accuracy:** Balancing the need for real-time responses with heavy deep learning computations. We solved this by implementing an efficient chunk-based processing pipeline.
- **Generalization:** Ensuring the model works across different languages and accents. We curated a diverse dataset including regional Indian languages to improve robustness.
- **GPU Resource Management:** Managing TensorFlow memory growth to prevent OOM errors during concurrent requests.

## 🏆 Accomplishments that we're proud of
- Achieving high detection accuracy on low-quality audio samples.
- Building a truly asynchronous inference engine that can handle multiple requests without blocking.
- Successfully integrating support for multiple Indian languages, addressing a gap in many existing tools.

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized run)
- CUDA-enabled GPU (recommended for faster inference)

### standard Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ironwall-ai.git
   cd ironwall-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Setup
1. **Build the image**
   ```bash
   docker build -t ironwall-ai .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 --gpus all ironwall-ai
   ```

## 📡 API Reference

### POST `/api/voice-detection`

Detects if the provided audio is AI-generated.

**Headers:**
- `x-api-key`: Your API Key (e.g., `apg_Tnz8en2Rt...`)

**Body (JSON):**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_audio_string>"
}
```

**Response:**
```json
{
  "is_ai": true,
  "confidence": 0.98,
  "reason": "Strong and repeated indicators of neural voice synthesis detected."
}
```

## 🔮 What's next for IronWall AI
- **Real-time Call Monitoring:** Integrate with VoIP systems to flag scams during live calls.
- **Video Deepfake Detection:** Expand the model to analyze visual artifacts in video streams.
- **Browser Extension:** A plugin to verify audio from web sources (YouTube, WhatsApp Web) instantly.

---

built with ❤️ by the IronWall AI Team.
