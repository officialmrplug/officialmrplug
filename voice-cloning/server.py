#!/usr/bin/env python3
"""
Local HTTP API for Chatterbox voice cloning, OpenAI-compatible.

    python server.py --voices ./voices

Then point any OpenAI-compatible client at http://127.0.0.1:8000/v1 :

    curl http://127.0.0.1:8000/v1/audio/speech \
      -H 'Content-Type: application/json' \
      -d '{"input":"Hello there","voice":"my_voice"}' \
      --output hello.wav

A "voice" is the basename of a reference clip in the voices directory, so
voices/my_voice.wav is requested as {"voice": "my_voice"}.

Binds to 127.0.0.1 by default: reachable only from this machine, no auth.
Pass --host 0.0.0.0 only if you understand you are exposing it to your network.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from clone_voice import pick_device, describe_device, split_text

# Formats soundfile writes directly; everything else needs ffmpeg.
NATIVE_FORMATS = {"wav": "WAV", "flac": "FLAC", "ogg": "OGG"}
FFMPEG_FORMATS = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac"}
MEDIA_TYPES = {
    "wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg",
    "mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
}

STATE: dict = {"model": None, "sr": None, "lock": threading.Lock(), "cfg": None}
app = FastAPI(title="Chatterbox TTS (local)", version="1.0")


class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech body, plus Chatterbox-specific extras."""
    input: str
    voice: str = "default"
    model: str = "chatterbox"
    response_format: str = "wav"
    speed: float = 1.0                       # accepted for compatibility; see below
    language: str | None = None
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(0.8, gt=0.0, le=2.0)
    max_chars: int = Field(300, ge=50, le=1000)


def voices_dir() -> Path:
    return Path(STATE["cfg"].voices)


def resolve_voice(name: str) -> Path:
    """Map a voice name to a reference clip, refusing anything outside the dir."""
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, f"invalid voice name: {name!r}")
    root = voices_dir()
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        candidate = root / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    available = sorted(p.stem for p in root.glob("*") if p.suffix in
                       (".wav", ".mp3", ".flac", ".ogg", ".m4a"))
    raise HTTPException(
        404,
        f"unknown voice {name!r}. Available: {available or 'none - add a clip to ' + str(root)}",
    )


def encode(wav: torch.Tensor, sr: int, fmt: str) -> bytes:
    """Serialise a (1, N) tensor to the requested container."""
    fmt = fmt.lower()
    samples = wav.squeeze(0).numpy()

    if fmt in NATIVE_FORMATS:
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format=NATIVE_FORMATS[fmt])
        return buf.getvalue()

    if fmt in FFMPEG_FORMATS:
        if not shutil.which("ffmpeg"):
            raise HTTPException(
                400, f"response_format {fmt!r} needs ffmpeg on PATH; use wav/flac/ogg instead")
        raw = io.BytesIO()
        sf.write(raw, samples, sr, format="WAV")
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-c:a", FFMPEG_FORMATS[fmt], "-f", fmt, "pipe:1"],
            input=raw.getvalue(), capture_output=True,
        )
        if proc.returncode != 0:
            raise HTTPException(500, f"ffmpeg failed: {proc.stderr.decode()[:200]}")
        return proc.stdout

    raise HTTPException(
        400, f"unsupported response_format {fmt!r}; "
             f"supported: {', '.join(sorted(NATIVE_FORMATS | FFMPEG_FORMATS.keys()))}")


def get_model():
    """Load the model once, on first request, and keep it resident."""
    if STATE["model"] is None:
        cfg = STATE["cfg"]
        print(f"Loading model on {cfg.device} (first run downloads ~2GB)...", flush=True)
        started = time.time()
        if cfg.turbo:
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
        elif cfg.multilingual:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
        else:
            from chatterbox.tts import ChatterboxTTS as Model
        STATE["model"] = Model.from_pretrained(device=cfg.device)
        STATE["sr"] = STATE["model"].sr
        print(f"Model ready in {time.time() - started:.1f}s", flush=True)
    return STATE["model"]


@app.get("/health")
def health():
    cfg = STATE["cfg"]
    return {
        "status": "ok",
        "device": cfg.device,
        "model_loaded": STATE["model"] is not None,
        "variant": "turbo" if cfg.turbo else ("multilingual" if cfg.multilingual else "base"),
        "sample_rate": STATE["sr"],
    }


@app.get("/voices")
def list_voices():
    root = voices_dir()
    return {"voices": sorted(p.stem for p in root.glob("*")
                             if p.suffix in (".wav", ".mp3", ".flac", ".ogg", ".m4a"))}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "chatterbox", "object": "model",
                                        "owned_by": "resemble-ai"}]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    ref = resolve_voice(req.voice)
    cfg = STATE["cfg"]
    model = get_model()

    chunks = split_text(req.input, req.max_chars)

    # One generation at a time: the model holds per-request conditioning state
    # and a single GPU cannot serve concurrent requests anyway.
    with STATE["lock"]:
        try:
            if cfg.turbo:
                model.prepare_conditionals(str(ref), exaggeration=0.0)
            else:
                model.prepare_conditionals(str(ref), exaggeration=req.exaggeration)

            pieces = []
            for chunk in chunks:
                if cfg.turbo:
                    wav = model.generate(chunk, temperature=req.temperature)
                elif cfg.multilingual:
                    wav = model.generate(
                        chunk, language_id=(req.language or cfg.language or "en").lower(),
                        exaggeration=req.exaggeration, cfg_weight=req.cfg_weight,
                        temperature=req.temperature, repetition_penalty=2.0)
                else:
                    wav = model.generate(
                        chunk, exaggeration=req.exaggeration, cfg_weight=req.cfg_weight,
                        temperature=req.temperature)
                pieces.append(wav)
        except ValueError as exc:            # e.g. unsupported language_id
            raise HTTPException(400, str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(
                507, "GPU out of memory; restart the server with --turbo or --device cpu"
            ) from exc

    wav = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
    audio = encode(wav, model.sr, req.response_format)
    fmt = req.response_format.lower()
    return Response(
        content=audio,
        media_type=MEDIA_TYPES.get(fmt, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Local OpenAI-compatible Chatterbox TTS server.")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to expose on your LAN")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--voices", default="voices", help="directory of reference clips")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--turbo", action="store_true")
    p.add_argument("--multilingual", action="store_true")
    p.add_argument("--language", default="en", help="default language for --multilingual")
    p.add_argument("--preload", action="store_true", help="load weights at startup, not on first request")
    args = p.parse_args(argv)
    if args.turbo and args.multilingual:
        p.error("--turbo and --multilingual cannot be combined")
    args.device = pick_device(args.device)
    return args


def main(argv=None):
    import uvicorn
    args = parse_args(argv)
    root = Path(args.voices)
    root.mkdir(parents=True, exist_ok=True)
    STATE["cfg"] = args

    print(f"Device : {describe_device(args.device)}")
    print(f"Voices : {root.resolve()}")
    if args.host != "127.0.0.1":
        print(f"WARNING: binding to {args.host} - this API has no authentication.")
    if args.preload:
        get_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    sys.exit(main())
