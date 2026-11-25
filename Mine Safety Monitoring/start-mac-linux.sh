#!/usr/bin/env bash
# One command for macOS / Linux: local LLM (Ollama) + the app on http://127.0.0.1:8080
cd "$(dirname "$0")" || exit 1
exec node scripts/local.mjs "$@"
