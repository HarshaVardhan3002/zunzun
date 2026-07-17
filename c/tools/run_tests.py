#!/usr/bin/env python3
"""Run C test binaries, exiting non-zero on the first failure.

Used by `make test-c` so the test runner works from any shell (cmd.exe,
PowerShell, Git Bash, MSYS2) without a POSIX `for` loop. Each test binary
is a positional argument.
"""
import os
import subprocess
import sys

failed = []
for binary in sys.argv[1:]:
    # abspath, not normpath: MSYS2 ucrt64 CPython's CreateProcess refuses
    # cwd-relative "tests\x.exe" paths (WinError 2) even when the file exists
    rc = subprocess.call([os.path.abspath(binary)])
    if rc != 0:
        failed.append(binary)
if failed:
    for f in failed:
        print(f"FAILED: {f}", file=sys.stderr)
    sys.exit(1)
