#!/usr/bin/env bash
# Move voice-cloning/ out of the profile repo into its own repository,
# preserving the commit history for these files.
#
# First create an EMPTY private repo on GitHub (no README, no .gitignore):
#   https://github.com/new  ->  name it, tick Private, create
#
# Then, from the root of the profile repo checkout:
#   ./voice-cloning/scripts/extract-to-private-repo.sh git@github.com:USER/voice-cloning.git
set -euo pipefail

REMOTE="${1:-}"
PREFIX="${PREFIX:-voice-cloning}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-../voice-cloning-standalone}"

if [[ -z "$REMOTE" ]]; then
  echo "usage: $0 <git-remote-url>" >&2
  echo "  e.g. $0 git@github.com:youruser/voice-cloning.git" >&2
  exit 1
fi
if [[ ! -d "$PREFIX" ]]; then
  echo "ERROR: '$PREFIX/' not found. Run this from the repo root." >&2
  exit 1
fi
if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: $WORKDIR already exists; remove it or set WORKDIR=..." >&2
  exit 1
fi

# The split must land on a real branch, otherwise the commit is unreachable
# and a clone of this repo cannot see it.
SPLIT_BRANCH="__vc_export_$$"
echo "Splitting '$PREFIX/' into its own history..."
git subtree split --prefix="$PREFIX" -b "$SPLIT_BRANCH" >/dev/null
echo "  split branch: $SPLIT_BRANCH ($(git rev-parse --short "$SPLIT_BRANCH"))"

echo "Creating standalone checkout at $WORKDIR"
git clone --quiet --no-local --branch "$SPLIT_BRANCH" --single-branch "$(pwd)" "$WORKDIR"

# The temporary branch has served its purpose in the source repo.
git branch -D "$SPLIT_BRANCH" >/dev/null

cd "$WORKDIR"
git checkout --quiet -B "$BRANCH"
git branch --quiet -D "$SPLIT_BRANCH" 2>/dev/null || true
git remote remove origin
git remote add origin "$REMOTE"

echo
echo "Standalone repo ready at $(pwd)"
echo "  files at root : $(ls | tr '\n' ' ')"
echo "  commits       : $(git rev-list --count HEAD)"
echo
echo "Push it with:"
echo "  cd $(pwd)"
echo "  git push -u origin $BRANCH"
echo
echo "Afterwards, remove it from the public profile repo:"
echo "  cd - && git rm -r --cached $PREFIX && rm -rf $PREFIX"
echo "  git commit -m 'Move voice-cloning to its own private repo' && git push"
echo "  # and delete the old branch so it is not public in history:"
echo "  git push origin --delete claude/self-hosted-voice-cloning-2ono52"
