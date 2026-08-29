# Project Brief: Self-Hosted Voice Cloning

> **Using this file with Claude Code:** it works under any filename. Either keep
> it as `PROJECT_BRIEF.md` and open a session with *"Read PROJECT_BRIEF.md and
> start on the task list"*, or copy it to `CLAUDE.md` (`cp PROJECT_BRIEF.md
> CLAUDE.md`) so it loads automatically every session. Nothing in the code
> depends on this file's name.

## Goal

Voice cloning that runs entirely on my own machine — like ElevenLabs or Fish
Audio, but self-hosted. No third-party API, no per-use cost, nothing leaves the
computer. Clone **any voice zero-shot**: a 10–20 s reference clip plus text
produces that voice speaking that text. No per-voice training or fine-tuning.

## Approach (decided — do not re-pick a model)

**Chatterbox TTS** by Resemble AI. MIT licensed, consumer hardware, beat
ElevenLabs in blind listening tests (63.75% preference).
GitHub `resemble-ai/chatterbox` · PyPI `chatterbox-tts` (pinned to **0.1.7**).

## Hardware

**NVIDIA GPU.** Target device is `cuda`. Expect ~1–3 s per sentence.
Base model needs ~6 GB VRAM; Turbo (350M) is the fallback if that OOMs.

## Verified API — checked against chatterbox-tts 0.1.7 source

```python
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="cuda")   # or "mps" / "cpu"
wav = model.generate(
    text,
    audio_prompt_path="reference.wav",
    exaggeration=0.5,   # 0 = flat, 1 = very expressive
    cfg_weight=0.5,     # how tightly it follows reference cadence
    temperature=0.8, repetition_penalty=1.2, min_p=0.05, top_p=1.0,
)
import torchaudio as ta
ta.save("output.wav", wav, model.sr)   # returns a CPU tensor (1, N); sr = 24000
```

### Three corrections to the commonly-copied version of this API

These were verified by reading the installed package source, and getting them
wrong causes silent misbehaviour:

1. **Turbo ignores `exaggeration` and `cfg_weight`.** `ChatterboxTurboTTS`
   (`chatterbox.tts_turbo`) defaults both to `0.0`, logs a warning, and
   discards them. Pass only `temperature`, `top_p`, `top_k`,
   `repetition_penalty`. It also pulls from a *different* HF repo
   (`ResembleAI/chatterbox-turbo`).
2. **The multilingual model requires `language_id`.** In
   `ChatterboxMultilingualTTS.generate()` (`chatterbox.mtl_tts`) it is the
   second positional parameter with **no default**, and
   `repetition_penalty` defaults to **2.0**, not 1.2.
3. **The reference clip is internally capped** at ~6 s (speaker encoder,
   `ENC_COND_LEN`) and ~10 s (decoder, `DEC_COND_LEN`). A 60 s clip is not
   better than a clean 15 s one.

## Environment

- **Python 3.11.** (Package metadata says `>=3.10`, but 3.11 is what this is
  tested against. Do not use a newer system Python.)
- PyTorch must be the **CUDA** build: `python -c "import torch; print(torch.cuda.is_available())"`
  must print `True`. If `False`, pip installed the CPU wheel — reinstall from
  the CUDA index URL matching the driver (`nvidia-smi` shows the CUDA version →
  `cu118` / `cu124` / `cu126`).
- `chatterbox-tts` already pins `numpy<2.0`, so a current pip resolves numpy
  correctly on its own. Only if something else forces numpy 2.x:
  `pip install numpy==1.26.4 --force-reinstall`.
- First `from_pretrained()` downloads ~2 GB from Hugging Face, once.
  **This machine needs one-time outbound access to `huggingface.co`.**

## Configuration — nothing is hard-coded

Resolution order, first win: **CLI flag → shell env var → `.env` → auto-discovery**.

```bash
cp .env.example .env       # then edit
```

Drop a clip into `voices/` and it is picked up automatically when it is the only
one there. With several clips, set `VOICE_REF` or pass `--ref`. `.env` is
gitignored; `.env.example` is the committed template.

## Files

| File | Purpose |
|---|---|
| `clone_voice.py` | Main script (CLI + `.env`) |
| `server.py` | Local FastAPI OpenAI-compatible `/v1/audio/speech` |
| `setup.sh` | Env bootstrap, detects GPU, installs matching torch |
| `voice_cloning_guide.md` | Full setup / tuning / troubleshooting |
| `.env.example` | Config template |

## Task list

1. ~~Detect hardware and OS~~ — **NVIDIA GPU, done.**
2. Create the Python 3.11 env and install; confirm `torch.cuda.is_available()` is `True`.
   → `./setup.sh` (add `--cuda cu126` / `--cuda cu118` if the driver needs it)
3. Put a 10–20 s reference clip in `voices/` (or convert one:
   `ffmpeg -i in.m4a -ss 5 -t 15 -ar 24000 -ac 1 voices/me.wav`).
4. Run `python clone_voice.py --text "..."` until it produces `output.wav`.
   Expect the ~2 GB download on first run.
5. Play it and check it resembles the reference; tune `--exaggeration` /
   `--cfg-weight` if not.
6. *Optional:* run the local API — `python server.py` — and call
   `/v1/audio/speech`. Still 100% local.
7. *Optional:* multilingual via `--language <iso>` (23 languages).

## Tuning cheatsheet

| Symptom | Try |
|---|---|
| Doesn't sound like the person | **Better clip first** — cleaner, single speaker. Then `--cfg-weight 0.3` |
| Rushed / clipped delivery | `--cfg-weight 0.3` |
| Flat, robotic | `--exaggeration 0.7 --cfg-weight 0.3` |
| Garbled on long text | lower `--max-chars` (e.g. 200) |
| Artifacts / warbling | `--temperature 0.6` |
| CUDA OOM | `--turbo`, or lower `--max-chars`, or `--device cpu` |

Generation is stochastic — same command, different take each run. Use `--seed`
to lock one in.

## Rules

- **Fully local only.** Never call ElevenLabs, Fish Audio, or any cloud TTS API.
- **Only clone voices I have permission to use.**
- Leave Resemble's inaudible **Perth watermark** in place on all output.
- Never commit `.env` or anything in `voices/`.
