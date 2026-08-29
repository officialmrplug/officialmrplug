#!/usr/bin/env bash
# Create the Python 3.11 environment for local voice cloning and install a
# PyTorch build that matches this machine's hardware.
#
#   ./setup.sh              # auto-detect GPU
#   ./setup.sh --cpu        # force the CPU build
#   ./setup.sh --cuda cu126 # force a specific CUDA build (cu118|cu124|cu126)
set -euo pipefail

VENV=".venv"
FORCE=""
CUDA_TAG="cu124"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)  FORCE="cpu"; shift ;;
    --cuda) FORCE="cuda"; CUDA_TAG="${2:-cu124}"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# --- locate a Python 3.11 interpreter ------------------------------------- #
PY=""
for c in python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done

if [[ -z "$PY" ]]; then
  echo "ERROR: Python 3.11 not found." >&2
  echo "  conda:  conda create -n voiceclone python=3.11 && conda activate voiceclone" >&2
  echo "  pyenv:  pyenv install 3.11.9 && pyenv local 3.11.9" >&2
  echo "  Debian: sudo apt install python3.11 python3.11-venv" >&2
  echo "  macOS:  brew install python@3.11" >&2
  exit 1
fi
echo "Python     : $PY ($($PY -V 2>&1))"

# --- decide which torch build to install ---------------------------------- #
OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ -n "$FORCE" ]]; then
  TARGET="$FORCE"
elif [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  TARGET="mps"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  TARGET="cuda"
else
  TARGET="cpu"
fi

case "$TARGET" in
  cuda) echo "Hardware   : NVIDIA GPU detected -> CUDA build ($CUDA_TAG)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
        INDEX="--index-url https://download.pytorch.org/whl/${CUDA_TAG}" ;;
  mps)  echo "Hardware   : Apple Silicon -> default build (Metal/MPS)"
        INDEX="" ;;
  *)    echo "Hardware   : no GPU detected -> CPU build (generation will be slow)"
        INDEX="--index-url https://download.pytorch.org/whl/cpu" ;;
esac

# --- build the environment ------------------------------------------------ #
echo "Creating   : $VENV"
"$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"
[[ -x "$PIP" ]] || PIP="$VENV/Scripts/pip"   # Git Bash on Windows

"$PIP" install --quiet --upgrade pip
echo "Installing : torch 2.6.0 ($TARGET)"
# shellcheck disable=SC2086
"$PIP" install torch==2.6.0 torchaudio==2.6.0 $INDEX
echo "Installing : chatterbox-tts + API server deps"
"$PIP" install chatterbox-tts fastapi uvicorn

# --- verify --------------------------------------------------------------- #
echo
echo "Verifying..."
"$VENV/bin/python" - <<'PYEOF'
import torch
print(f"  torch            {torch.__version__}")
print(f"  cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu              {torch.cuda.get_device_name(0)}")
print(f"  mps available    {torch.backends.mps.is_available()}")
import chatterbox.tts  # noqa: F401
print("  chatterbox       import OK")
PYEOF

cat <<EOF

Done. Next:
  source $VENV/bin/activate
  python clone_voice.py --ref your_voice.wav --text "Hello world"

If 'cuda available' printed False but you do have an NVIDIA GPU, re-run:
  ./setup.sh --cuda cu126     (or cu118, matching your driver)
EOF
