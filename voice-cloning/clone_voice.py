#!/usr/bin/env python3
"""
Zero-shot voice cloning with Chatterbox TTS (Resemble AI) - 100% local.

Nothing is hard-coded. Configuration is resolved in this order, first win:

    1. command-line flag        --ref my_voice.wav
    2. environment variable     VOICE_REF=my_voice.wav
    3. .env file in this folder VOICE_REF=my_voice.wav
    4. auto-discovery           the only clip in the voices/ folder

So after copying .env.example to .env and dropping a clip in voices/, this
is the whole workflow:

    python clone_voice.py --text "Hello world"

Nothing leaves this machine. Model weights are downloaded once from Hugging
Face on the first run (~2GB), then cached locally (see HF_HOME in .env).
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")


def load_dotenv_file(path: Path) -> None:
    """Minimal .env loader: KEY=value, # comments, optional quotes.

    Kept dependency-free on purpose, and it never overrides a variable that is
    already set in the real environment.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # A blank value means "unset", not "empty string". Exporting an empty
        # HF_HOME, for instance, makes Hugging Face cache into a relative ./hub
        # directory and download ~2GB into the project folder.
        if value == "":
            continue
        os.environ.setdefault(key, value)


load_dotenv_file(Path(os.environ.get("VOICE_ENV_FILE", HERE / ".env")))


def env_str(key: str, default=None):
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def env_float(key: str, default):
    raw = env_str(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        sys.exit(f"ERROR: {key} in .env must be a number, got {raw!r}")


def env_int(key: str, default):
    raw = env_str(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"ERROR: {key} in .env must be an integer, got {raw!r}")


def env_bool(key: str, default=False):
    raw = env_str(key)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


# Honour a custom weight-cache location before torch/HF are imported.
if env_str("HF_HOME"):
    os.environ.setdefault("HF_HOME", env_str("HF_HOME"))

# Must be set before torch is imported, otherwise unimplemented MPS ops hard-fail
# on Apple Silicon instead of falling back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio as ta


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #
def pick_device(requested: str | None = None) -> str:
    """Choose cuda > mps > cpu, or validate an explicitly requested device."""
    if requested and requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            sys.exit(
                "ERROR: --device cuda requested but torch.cuda.is_available() is False.\n"
                "       Your PyTorch is probably a CPU-only build. Reinstall with the\n"
                "       CUDA wheel for your driver, e.g.:\n"
                "         pip install torch==2.6.0 torchaudio==2.6.0 \\\n"
                "             --index-url https://download.pytorch.org/whl/cu124"
            )
        if requested == "mps" and not torch.backends.mps.is_available():
            sys.exit("ERROR: --device mps requested but MPS is not available on this machine.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"cuda ({name}, {vram:.1f} GB VRAM)"
    if device == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu (slow - expect roughly 1-3 minutes per sentence)"


# --------------------------------------------------------------------------- #
# Reference clip resolution and checks
# --------------------------------------------------------------------------- #
def resolve_reference(explicit: str | None, voices_dir: str) -> str:
    """Find the reference clip without assuming any particular filename."""
    if explicit:
        return explicit

    root = Path(voices_dir)
    if not root.is_absolute():
        root = HERE / root
    clips = sorted(p for p in root.glob("*") if p.suffix.lower() in AUDIO_SUFFIXES)

    if len(clips) == 1:
        print(f"  (using the only clip in {root.name}/: {clips[0].name})")
        return str(clips[0])
    if not clips:
        sys.exit(
            f"ERROR: no reference clip given and none found in {root}/\n"
            f"       Do one of these:\n"
            f"         - drop a 10-20s clip into {root}/\n"
            f"         - set VOICE_REF=<path> in .env\n"
            f"         - pass --ref <path>"
        )
    sys.exit(
        f"ERROR: {len(clips)} clips in {root}/ - pick one explicitly.\n"
        f"       Available: {', '.join(p.stem for p in clips)}\n"
        f"       e.g. --ref {clips[0]}   (or set VOICE_REF in .env)"
    )



def check_reference(path: str) -> None:
    """Fail fast on a missing clip; warn about clips likely to clone badly."""
    p = Path(path)
    if not p.is_file():
        sys.exit(
            f"ERROR: reference clip not found: {p}\n"
            f"       Record 10-20 seconds of the voice you have permission to clone,\n"
            f"       Set VOICE_REF in .env, pass --ref, or drop a clip into voices/.\n"
            f"       Convert anything to WAV with:  ffmpeg -i input.m4a -ar 24000 -ac 1 {p}"
        )

    try:
        info = ta.info(str(p))
    except Exception as exc:  # unreadable / unsupported container
        sys.exit(
            f"ERROR: could not read '{p}' as audio ({exc}).\n"
            f"       Convert it first:  ffmpeg -i '{p}' -ar 24000 -ac 1 reference_voice.wav"
        )

    seconds = info.num_frames / info.sample_rate if info.sample_rate else 0.0
    print(f"  reference : {p.name}  ({seconds:.1f}s, {info.sample_rate} Hz, "
          f"{info.num_channels} channel{'s' if info.num_channels > 1 else ''})")

    if seconds < 5:
        print(f"  WARNING   : clip is only {seconds:.1f}s. 10-20s clones far better.")
    elif seconds > 30:
        # Chatterbox only consumes the first ~6s (encoder) / ~10s (decoder) anyway.
        print(f"  NOTE      : clip is {seconds:.1f}s; only the first ~10s is actually used.")
    if info.num_channels > 1:
        print("  NOTE      : stereo clip; it will be downmixed to mono internally.")


# --------------------------------------------------------------------------- #
# Text chunking (the model degrades / truncates on very long inputs)
# --------------------------------------------------------------------------- #
def split_text(text: str, max_chars: int) -> list[str]:
    """Split into sentence-aligned chunks of at most max_chars characters."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        # A single sentence longer than the budget gets split on whitespace.
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return [c for c in chunks if c]


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(device: str, turbo: bool, language: str | None):
    """Import and load the right Chatterbox variant. Weights download on first use."""
    if turbo:
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
        label = "ChatterboxTurboTTS (350M, English)"
    elif language:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
        supported = Model.get_supported_languages()
        if language.lower() not in supported:
            sys.exit(
                f"ERROR: unsupported --language '{language}'.\n"
                f"       Supported: {', '.join(f'{k} ({v})' for k, v in supported.items())}"
            )
        label = f"ChatterboxMultilingualTTS ({supported[language.lower()]})"
    else:
        from chatterbox.tts import ChatterboxTTS as Model
        label = "ChatterboxTTS (500M, English)"

    print(f"  model     : {label}")
    print("  loading weights (first run downloads ~2GB from Hugging Face)...")
    started = time.time()
    try:
        model = Model.from_pretrained(device=device)
    except Exception as exc:
        raise_load_error(exc, device)
    print(f"  loaded in {time.time() - started:.1f}s")
    return model


def raise_load_error(exc: Exception, device: str) -> None:
    """Translate the common weight-download / OOM failures into actionable advice."""
    msg = str(exc)
    if is_oom(exc):
        sys.exit(
            "ERROR: ran out of GPU memory while loading the model.\n"
            "       Try the smaller Turbo model:  python clone_voice.py --turbo\n"
            "       Or fall back to CPU:          python clone_voice.py --device cpu"
        )
    if any(s in msg for s in ("ConnectionError", "Max retries", "Could not reach",
                              "Name or service not known", "NewConnectionError",
                              "couldn't connect", "ProxyError", "403")):
        sys.exit(
            f"ERROR: could not download the model weights from Hugging Face.\n"
            f"       ({type(exc).__name__}: {msg[:200]})\n"
            f"       This machine needs one-time outbound access to huggingface.co.\n"
            f"       If it is offline, download the weights on another machine and copy\n"
            f"       the ~/.cache/huggingface directory across, then re-run."
        )
    raise exc


def is_oom(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def synthesize(model, chunks, args, device):
    """Prepare the voice conditionals once, then generate each chunk."""
    # Doing this once (rather than passing audio_prompt_path on every call)
    # avoids re-encoding the reference clip for each chunk.
    if args.turbo:
        model.prepare_conditionals(args.ref, exaggeration=0.0)
    else:
        model.prepare_conditionals(args.ref, exaggeration=args.exaggeration)

    pieces = []
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk if len(chunk) <= 60 else chunk[:57] + "..."
        print(f"  [{i}/{len(chunks)}] {preview}")
        started = time.time()

        # Turbo ignores exaggeration/cfg_weight/min_p and warns if they are set,
        # so each variant gets only the knobs it actually supports.
        try:
            if args.turbo:
                wav = model.generate(
                    chunk,
                    temperature=args.temperature,
                    top_p=0.95,
                    repetition_penalty=args.repetition_penalty,
                )
            elif args.language:
                wav = model.generate(
                    chunk,
                    language_id=args.language.lower(),
                    exaggeration=args.exaggeration,
                    cfg_weight=args.cfg_weight,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                )
            else:
                wav = model.generate(
                    chunk,
                    exaggeration=args.exaggeration,
                    cfg_weight=args.cfg_weight,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                )
        except Exception as exc:
            if is_oom(exc):
                sys.exit(
                    "ERROR: CUDA out of memory during generation.\n"
                    "       Use --turbo (smaller model), --device cpu, or a smaller\n"
                    "       --max-chars so each chunk is shorter."
                )
            raise

        print(f"        -> {wav.shape[-1] / model.sr:.1f}s audio in {time.time() - started:.1f}s")
        pieces.append(wav)

        if i < len(chunks) and args.gap > 0:
            pieces.append(torch.zeros(1, int(args.gap * model.sr)))

    return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Zero-shot voice cloning with Chatterbox TTS (fully local).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ref", default=env_str("VOICE_REF"),
                   help="reference voice clip (default: VOICE_REF, else the only clip in --voices-dir)")
    p.add_argument("--voices-dir", default=env_str("VOICES_DIR", "voices"),
                   help="folder searched when --ref is not given")
    p.add_argument("--text", default=None, help="text to speak")
    p.add_argument("--text-file", default=None, help="read the text from a file instead")
    p.add_argument("--out", default=env_str("VOICE_OUTPUT", "output.wav"), help="output WAV path")
    p.add_argument("--device", default=env_str("VOICE_DEVICE", "auto"), choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--turbo", action="store_true", default=env_bool("VOICE_TURBO"),
                   help="use the 350M Turbo model (less VRAM; ignores exaggeration/cfg-weight)")
    p.add_argument("--language", default=env_str("VOICE_LANGUAGE"),
                   help="ISO code for the multilingual model, e.g. fr, de, ja")
    p.add_argument("--exaggeration", type=float, default=env_float("VOICE_EXAGGERATION", 0.5), help="0 = flat, 1 = very expressive")
    p.add_argument("--cfg-weight", type=float, default=env_float("VOICE_CFG_WEIGHT", 0.5), help="how closely to follow reference cadence")
    p.add_argument("--temperature", type=float, default=env_float("VOICE_TEMPERATURE", 0.8))
    p.add_argument("--repetition-penalty", type=float, default=None,
                   help="default: 1.2 (base/turbo), 2.0 (multilingual)")
    p.add_argument("--max-chars", type=int, default=env_int("VOICE_MAX_CHARS", 300), help="split text into chunks this long")
    p.add_argument("--gap", type=float, default=env_float("VOICE_GAP", 0.15), help="seconds of silence between chunks")
    p.add_argument("--seed", type=int, default=env_int("VOICE_SEED", None), help="set for reproducible output")
    args = p.parse_args(argv)

    if args.text and args.text_file:
        p.error("use --text or --text-file, not both")
    if args.turbo and args.language:
        p.error("--turbo and --language cannot be combined; Turbo is English-only")
    if args.text_file:
        f = Path(args.text_file)
        if not f.is_file():
            p.error(f"--text-file not found: {f}")
        args.text = f.read_text(encoding="utf-8")
    if args.text is None:
        args.text = env_str("VOICE_TEXT",
                            "Hello! This is my cloned voice, generated on my own machine.")
    if not args.text.strip():
        p.error("no text to speak")

    if args.repetition_penalty is None:
        args.repetition_penalty = 2.0 if args.language else 1.2
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    print("Chatterbox voice cloning (local)")
    device = pick_device(args.device)
    print(f"  device    : {describe_device(device)}")
    args.ref = resolve_reference(args.ref, args.voices_dir)
    check_reference(args.ref)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    chunks = split_text(args.text, args.max_chars)
    print(f"  text      : {len(args.text)} chars in {len(chunks)} chunk(s)")

    model = load_model(device, args.turbo, args.language)
    wav = synthesize(model, chunks, args, device)

    out = Path(args.out)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(out), wav, model.sr)

    duration = wav.shape[-1] / model.sr
    print(f"\nWrote {out.resolve()}  ({duration:.1f}s @ {model.sr} Hz)")
    print("Output carries Resemble's inaudible Perth watermark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
