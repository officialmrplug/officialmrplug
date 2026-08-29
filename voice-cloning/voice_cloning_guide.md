# Self-Hosted Voice Cloning — Setup Guide

Zero-shot voice cloning with [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
(Resemble AI, MIT). Give it a 10–20 second reference clip plus some text and it
speaks that text in that voice. No per-voice training, no cloud API, no per-use cost.

Everything runs locally. The only network access ever required is a one-time
~2 GB weight download from Hugging Face on first run.

---

## 1. Requirements

| | |
|---|---|
| Python | **3.11** (3.10+ works; 3.11 is what the stack is tested against) |
| Disk | ~8 GB (≈2 GB weights + ≈5 GB PyTorch/CUDA libraries) |
| GPU | NVIDIA ≥6 GB VRAM, or Apple Silicon, or CPU (slow but functional) |
| RAM | 8 GB minimum, 16 GB comfortable |

Speed, very roughly, for one sentence: NVIDIA GPU ~1–3 s · Apple Silicon ~5–15 s ·
CPU ~1–3 min. CPU is fine for testing, painful for bulk work.

---

## 2. Install

```bash
cd voice-cloning
./setup.sh              # detects your GPU and installs the matching torch build
source .venv/bin/activate
```

`setup.sh --cpu`, `--mps`, or `--cuda cu126` overrides the detection.

### Apple Silicon (M1/M2/M3/M4)

The default PyPI torch wheel already includes Metal/MPS — **do not pass an
index URL on macOS**; the cpu/cuda indexes publish no macOS arm64 wheels and
resolution fails. `setup.sh` handles this automatically.

```bash
brew install python@3.11 ffmpeg
./setup.sh                 # prints: mps available True
```

`clone_voice.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing torch,
so ops Metal has not implemented fall back to CPU rather than crashing.

MPS is the least-exercised path in this stack. If a run errors out or the audio
comes back garbled, check whether it is Metal rather than your clip:

```bash
python clone_voice.py --text "Testing one two three." --device cpu
```

If CPU sounds right and MPS does not, it is an MPS issue — stay on `--device cpu`.
Apple Silicon has no separate VRAM pool, so unified memory means you are
unlikely to hit an out-of-memory wall; `--turbo` there buys speed, not headroom. It also
creates `.env` from `.env.example` on first run.

### Configuration (`.env`)

Nothing is hard-coded. Settings resolve in this order, first win:

1. command-line flag — `--ref my_voice.wav`
2. shell environment variable — `VOICE_REF=my_voice.wav`
3. `.env` in the project folder
4. auto-discovery — the only clip in `voices/`

So the usual workflow is: drop a clip in `voices/`, run
`python clone_voice.py --text "..."`. With more than one clip there, set
`VOICE_REF` in `.env` or pass `--ref`.

`.env` is gitignored and never committed; `.env.example` is the template and
lists every supported key. `HF_HOME` relocates the ~2 GB weight cache if you
want it off your system drive.

<details>
<summary>Manual install</summary>

```bash
conda create -n voiceclone python=3.11 -y && conda activate voiceclone

# Pick ONE torch line:
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  # NVIDIA
pip install torch==2.6.0 torchaudio==2.6.0                                                     # Apple Silicon
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu    # CPU

pip install chatterbox-tts
```
</details>

Confirm the GPU is actually visible to PyTorch — this is the single most common
setup mistake:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # want: True
```

`False` on a machine with an NVIDIA card means pip installed the CPU-only wheel.
Reinstall using the CUDA index URL above that matches your driver
(`nvidia-smi` shows the CUDA version; use `cu118`, `cu124`, or `cu126`).

---

## 3. Prepare a reference clip

**Only clone voices you have permission to use.**

Requirements: **10–20 seconds**, one speaker, no background music or noise, no
reverb, natural speaking pace. Quality of this clip matters more than any
parameter you can tune.

Chatterbox internally uses only the **first ~6 s** (speaker encoder) and
**~10 s** (decoder), so a 60-second clip is not better than a clean 15-second
one — it just wastes the good part if the start is weak.

Record, or convert anything you already have:

```bash
# convert any audio/video to the expected format
ffmpeg -i input.m4a -ar 24000 -ac 1 reference_voice.wav

# trim to the best 15 seconds (start at 00:00:05, take 15s)
ffmpeg -i long.wav -ss 00:00:05 -t 15 -ar 24000 -ac 1 reference_voice.wav

# record directly (Linux)
arecord -f cd -d 15 -r 24000 -c 1 reference_voice.wav
```

Mono 24 kHz WAV is ideal. Stereo, other sample rates, and mp3/flac/m4a all work —
they are converted internally — but `clone_voice.py` will note when it adjusts.

---

## 4. Generate

```bash
python clone_voice.py --ref reference_voice.wav --text "Hello, this is my cloned voice."
```

First run downloads ~2 GB to `~/.cache/huggingface`. Later runs start in seconds.

```bash
# read a script from a file
python clone_voice.py --ref voice.wav --text-file script.txt --out narration.wav

# smaller/faster model (350M instead of 500M) — use if you hit VRAM limits
python clone_voice.py --ref voice.wav --text "Hi" --turbo

# other languages (23 supported)
python clone_voice.py --ref voice.wav --text "Bonjour le monde" --language fr

# reproducible output
python clone_voice.py --ref voice.wav --text "Hi" --seed 42
```

Or edit `REFERENCE_VOICE` / `TEXT_TO_SPEAK` at the top of `clone_voice.py` and
just run `python clone_voice.py`.

---

## 5. Tuning

| Flag | Default | Effect |
|---|---|---|
| `--exaggeration` | 0.5 | 0 = flat and even, 1 = dramatic. Above ~0.7 gets unstable. |
| `--cfg-weight` | 0.5 | How tightly it follows the reference's cadence. Lower = freer pacing. |
| `--temperature` | 0.8 | Randomness. Lower = more consistent, higher = more varied. |
| `--seed` | — | Fix it to reproduce a take exactly. |

Practical starting points:

- **Doesn't sound like the person** → the clip is the problem, not the settings.
  Use a cleaner/longer one. Then try `--cfg-weight 0.3`.
- **Fast talker / rushed delivery** → `--cfg-weight 0.3`
- **Flat, robotic narration** → `--exaggeration 0.7 --cfg-weight 0.3`
- **Rambling or garbled on long text** → lower `--max-chars` (e.g. 200)
- **Audible artifacts** → `--temperature 0.6`

Generation is stochastic: the same command gives a different take each run unless
you pass `--seed`. If a line comes out wrong, just run it again.

Long text is split on sentence boundaries into `--max-chars` chunks (default 300)
and concatenated with a short gap, because quality degrades on very long inputs.

---

## 6. Local HTTP API (optional)

`server.py` exposes an OpenAI-compatible endpoint so other apps can use it,
still entirely offline:

```bash
mkdir -p voices && cp reference_voice.wav voices/my_voice.wav
python server.py                 # http://127.0.0.1:8000
```

A "voice" is the filename stem: `voices/my_voice.wav` → `"voice": "my_voice"`.

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from my local server","voice":"my_voice"}' \
  --output hello.wav
```

Because it implements `/v1/audio/speech`, the official OpenAI client works
against it unchanged:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
client.audio.speech.create(model="chatterbox", voice="my_voice",
                           input="Hello").stream_to_file("out.wav")
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | Generate speech (OpenAI-compatible) |
| `GET /voices` | List available reference clips |
| `GET /v1/models` | Model list (OpenAI-compatible) |
| `GET /health` | Device, variant, whether weights are loaded |

Formats: `wav`, `flac`, `ogg` natively; `mp3`, `opus`, `aac` if `ffmpeg` is on PATH.
Extra body fields `exaggeration`, `cfg_weight`, `temperature`, `language`, and
`max_chars` are accepted alongside the standard OpenAI fields. `speed` is accepted
for compatibility but ignored — Chatterbox has no speed control; use
`--cfg-weight` to influence pacing instead.

The server **binds to 127.0.0.1** by default, so only this machine can reach it.

Set `SERVER_API_KEY` in `.env` to require a bearer token:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer $SERVER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello","voice":"my_voice"}' --output hello.wav
```

With no key set, access is open — fine on 127.0.0.1. If you pass
`--host 0.0.0.0` to expose it on your LAN, set a key; the server warns loudly
if you do not. `/health` stays unauthenticated so you can monitor it.

Requests are serialised with a lock — one generation at a time, since a single
GPU cannot serve concurrent requests anyway.

---

## 7. Troubleshooting

**`torch.cuda.is_available()` is False** — CPU-only wheel installed. Reinstall
torch with the CUDA index URL matching your driver (see §2).

**CUDA out of memory** — use `--turbo` (350M vs 500M params), or lower
`--max-chars`, or `--device cpu`. Close other GPU processes.

**numpy version conflict** — `chatterbox-tts` requires `numpy<2.0`. A current pip
resolves this automatically; if something else forced numpy 2.x:
`pip install "numpy==1.26.4" --force-reinstall`

**Weight download fails / no internet on this machine** — run once on a connected
machine, then copy `~/.cache/huggingface` across. After that it never needs the
network again.

**Apple Silicon: "MPS backend out of memory" or unimplemented-op errors** —
`clone_voice.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically. If it still
fails, `--device cpu`.

**Output is silent or truncated** — the reference clip is probably near-silent or
mostly background noise. Check it plays properly first.

**`ffmpeg not found`** for mp3 output — install ffmpeg, or request `wav`/`flac`.

---

## 8. Notes

- Every output carries Resemble's inaudible **Perth watermark** by design. Leave
  it in place — it is what makes this responsible to use.
- Base and Turbo models are **English only**; use `--language` for the other 22.
- Model weights cache in `~/.cache/huggingface` and are shared across projects.
- Voice conversion (speak-in-someone-else's-voice from existing audio) is also
  available in the package via `chatterbox.vc`, not wired up here.

### Supported languages

`ar` Arabic · `da` Danish · `de` German · `el` Greek · `en` English ·
`es` Spanish · `fi` Finnish · `fr` French · `he` Hebrew · `hi` Hindi ·
`it` Italian · `ja` Japanese · `ko` Korean · `ms` Malay · `nl` Dutch ·
`no` Norwegian · `pl` Polish · `pt` Portuguese · `ru` Russian ·
`sv` Swedish · `sw` Swahili · `tr` Turkish · `zh` Chinese
