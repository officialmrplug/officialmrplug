# Self-Hosted Voice Cloning

Zero-shot voice cloning that runs entirely on your own machine — a 10–20 second
reference clip plus text produces that voice speaking that text. No per-voice
training, no cloud API, no per-use cost, nothing leaves the machine.

Built on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI
(MIT licensed).

## Quick start

```bash
./setup.sh                       # detects GPU, installs matching PyTorch
source .venv/bin/activate
python clone_voice.py --ref my_voice.wav --text "Hello, this is my cloned voice."
```

First run downloads ~2 GB of weights from Hugging Face (once). After that it is
fully offline.

## Files

| File | Purpose |
|---|---|
| `clone_voice.py` | Main script — CLI or edit the constants at the top |
| `server.py` | Optional local HTTP API, OpenAI-compatible `/v1/audio/speech` |
| `setup.sh` | Environment bootstrap with hardware detection |
| `voice_cloning_guide.md` | Full setup, tuning and troubleshooting guide |
| `requirements.txt` | Pinned dependencies |

## Common commands

```bash
python clone_voice.py --ref v.wav --text-file script.txt --out narration.wav
python clone_voice.py --ref v.wav --text "Hi" --turbo            # smaller model, less VRAM
python clone_voice.py --ref v.wav --text "Bonjour" --language fr # 23 languages
python clone_voice.py --ref v.wav --text "Hi" --exaggeration 0.7 --cfg-weight 0.3
python server.py                                                 # local API on :8000
```

See [`voice_cloning_guide.md`](voice_cloning_guide.md) for tuning and troubleshooting.

## Requirements

Python 3.11 · ~8 GB disk · NVIDIA GPU (≥6 GB VRAM), Apple Silicon, or CPU.

Roughly per sentence: NVIDIA ~1–3 s · Apple Silicon ~5–15 s · CPU ~1–3 min.

## Responsible use

- **Only clone voices you have permission to use.**
- Every output carries Resemble's inaudible Perth watermark by design. Leave it in place.
