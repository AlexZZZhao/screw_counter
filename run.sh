#!/bin/sh
# macOS / Linux launcher. First run creates a local venv and installs dependencies.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python screw_counter.py "$@"
