#!/usr/bin/env python3
"""Repo integrity guard.

Catches the corruption this environment's editor/mount sync can leave behind:
NUL bytes, invalid UTF-8, truncation. Run after any change.

Rule learned the hard way: in this sandbox the file-EDITOR tool can corrupt an
existing file when it rewrites it through the mount (trailing-NUL padding on
shrink, or mid-line truncation). So modify tracked source with a single
deterministic writer (shell/python) and VERIFY with this guard afterwards.
Brand-new files are fine; it is in-place edits of large files that bite.

Usage:
  python scripts/check_integrity.py            # scan default source globs
  python scripts/check_integrity.py c/glm.c    # scan specific files
  python scripts/check_integrity.py --fix      # also strip trailing-NUL padding
Exit 0 = clean, 1 = problems found.

--fix strips a TRAILING run of NUL bytes (provably safe: real content precedes
it). Interior NULs or truncation are NOT auto-fixed: genuine damage, rebuild
the file deterministically instead.
"""
import sys, os, glob, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBS = ["c/*.c", "c/*.h", "c/*.py", "c/tests/*.c", "c/tools/*.py",
         "router/*.py", "router/*.yaml", "engines/**/*", "phase0/*",
         "scripts/*.py", "*.md", "c/*.cpp", "bench/*.cpp", "bench/*.md"]
CC_UNITS = ["glm.c", "olmoe.c", "iobench.c"]   # standalone translation units

def targets(args):
    if args:
        return [a if os.path.isabs(a) else os.path.join(ROOT, a) for a in args]
    out = []
    for g in GLOBS:
        out += [p for p in glob.glob(os.path.join(ROOT, g), recursive=True) if os.path.isfile(p)]
    return sorted(set(out))

def scan(path):
    b = open(path, "rb").read()
    errs = []
    if b.count(0):
        errs.append(f"{b.count(0)} NUL byte(s) (corruption)")
    try:
        b.decode("utf-8")
    except UnicodeDecodeError as e:
        errs.append(f"invalid utf-8 at byte {e.start}")
    if len(b) == 0:
        errs.append("empty file")
    elif not b.endswith(b"\n"):
        errs.append("no trailing newline (possible truncation)")
    return errs

def cc_ok(path):
    if not any(path.endswith(u) for u in CC_UNITS):
        return None
    try:
        r = subprocess.run(
            ["gcc", "-fsyntax-only", "-I", os.path.join(ROOT, "c"),
             "-D_GNU_SOURCE", "-fopenmp", path],
            capture_output=True, text=True, cwd=os.path.join(ROOT, "c"))
    except FileNotFoundError:
        return None
    return (r.returncode == 0, r.stderr.strip().splitlines()[:4])

def strip_trailing_nuls(path):
    b = open(path, "rb").read()
    if not b.endswith(b"\x00"):
        return 0
    s = b.rstrip(b"\x00")
    if b"\x00" in s:
        return -1
    open(path, "wb").write(s)
    return len(b) - len(s)

def main():
    argv = sys.argv[1:]
    fix = "--fix" in argv
    argv = [a for a in argv if a != "--fix"]
    bad = 0
    for p in targets(argv):
        rel = os.path.relpath(p, ROOT)
        if fix:
            n = strip_trailing_nuls(p)
            if n > 0:
                print(f"fixed {rel}: stripped {n} trailing NUL byte(s)")
            elif n < 0:
                print(f"WARN  {rel}: interior NULs -- genuine damage, rebuild from source")
        errs = scan(p)
        cc = cc_ok(p)
        if cc is not None and not cc[0]:
            errs.append("does NOT compile: " + " | ".join(cc[1]))
        if errs:
            bad += 1
            print(f"FAIL  {rel}")
            for e in errs:
                print(f"        - {e}")
        else:
            tag = " (compiles)" if cc and cc[0] else ""
            print(f"ok    {rel}{tag}")
    print(f"\n{'PROBLEMS: ' + str(bad) if bad else 'ALL CLEAN'}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
