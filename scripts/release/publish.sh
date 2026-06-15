#!/usr/bin/env bash
#
# publish.sh — create the PUBLIC GitHub repo and cut the first release.
#
# RUN ONLY AFTER:
#   - cofounder / board sign-off (this is a company-owned OSS release),
#   - the code rename (scripts/release/rename.sh) is done, and
#   - the FULL build + `make smoke` passed on the GB10.
#
# This is IRREVERSIBLE: a public repo is indexed and cached the moment it exists.
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ORG="hikmaai-io"
NAME="fucina"
DESC="Gemma 4 inference forged for the NVIDIA DGX Spark GB10 — experimental, no support"

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree not clean — commit everything you intend to publish first." >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "About to create PUBLIC repo ${ORG}/${NAME} and push branch '${BRANCH}' as default."
read -r -p "Type the repo name (${NAME}) to confirm: " ack
[ "$ack" = "$NAME" ] || { echo "aborted."; exit 1; }

# Create + push (sets 'origin', pushes the current branch as the default branch).
gh repo create "${ORG}/${NAME}" --public --description "$DESC" \
  --source . --remote origin --push

# Topics + features.
gh repo edit "${ORG}/${NAME}" \
  --add-topic gemma --add-topic gemma4 --add-topic cuda --add-topic llm-inference \
  --add-topic inference-engine --add-topic dgx-spark --add-topic blackwell \
  --add-topic gpu --add-topic golang \
  --enable-issues --enable-wiki=false

# Discussions (for "does it run on X" GPU-support threads) + private vuln reporting.
gh api -X PATCH "repos/${ORG}/${NAME}" -F has_discussions=true >/dev/null \
  || echo "note: enable Discussions manually (Settings > General > Features)."
gh api -X PUT "repos/${ORG}/${NAME}/private-vulnerability-reporting" >/dev/null 2>&1 \
  || echo "note: enable private vulnerability reporting manually (Settings > Security)."

echo
echo "repo live: https://github.com/${ORG}/${NAME}"
echo "verify CI is green, then cut the release:"
echo "  git tag -a v0.1.0 -m 'fucina v0.1.0'"
echo "  git push origin v0.1.0"
echo "  gh release create v0.1.0 --title 'fucina v0.1.0' \\"
echo "    --notes-file docs/launch/RELEASE_NOTES_v0.1.0.md"
