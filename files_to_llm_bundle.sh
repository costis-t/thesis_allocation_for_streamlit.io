#!/usr/bin/env bash
set -euo pipefail

# Bundle multiple files into a single LLM-friendly Markdown document.
# For each input file, writes:
# ===== FILE: <relative/path> =====
# ```<lang>
# <contents>
# ```
#
# Usage:
#   files_to_llm_bundle.sh [-o OUTPUT] [--base DIR] [--max-bytes N] [--include-binary] [--append] FILE...
#
# Options:
#   -o, --output FILE     Output bundle file (default: llm_bundle.md)
#   --base DIR            Base directory to compute relative paths from (default: current directory)
#   --max-bytes N         Skip files larger than N bytes (default: 0 = no limit)
#   --include-binary      Include binary files as base64 blocks (default: skip binaries)
#   --append              Append to OUTPUT instead of overwriting
#   -h, --help            Show help

OUTPUT="llm_bundle.md"
BASE="$PWD"
MAX_BYTES=0
INCLUDE_BINARY=0
APPEND=0

usage() {
  sed -n '1,100p' "$0" | sed -n '2,60p' | sed 's/^# \{0,1\}//'
  exit 0
}

# Cross-platform relative path
relpath() {
  local base="$1" target="$2"
  if command -v realpath >/dev/null 2>&1; then
    realpath --relative-to="$base" "$target"
  else
    # Fallback to Python for portability (macOS, etc.)
    python3 - "$base" "$target" <<'PY'
import os, sys
base=os.path.abspath(sys.argv[1])
target=sys.argv[2]
print(os.path.relpath(target, start=base))
PY
  fi
}

# Guess codefence language from extension
fence_lang() {
  case "${1##*.}" in
    sh|bash) echo "bash" ;;
    py) echo "python" ;;
    js) echo "javascript" ;;
    ts) echo "typescript" ;;
    json) echo "json" ;;
    md) echo "markdown" ;;
    yml|yaml) echo "yaml" ;;
    html|htm) echo "html" ;;
    css) echo "css" ;;
    java) echo "java" ;;
    cpp|cc|cxx|hpp|hh|hxx) echo "cpp" ;;
    c|h) echo "c" ;;
    cs) echo "csharp" ;;
    go) echo "go" ;;
    rs) echo "rust" ;;
    php) echo "php" ;;
    rb) echo "ruby" ;;
    kt|kts) echo "kotlin" ;;
    sql) echo "sql" ;;
    ps1|psm1|psd1) echo "powershell" ;;
    txt|log|ini|cfg) echo "" ;;  # treat as generic fenced block
    *) echo "" ;;
  esac
}

# Heuristic: is text file?
is_text_file() {
  # Portable-ish test: grep -Iq . FILE returns 0 for "likely text"
  # (This is heuristic; for robust MIME, use `file -bi`, but it's less portable.)
  grep -Iq . -- "$1"
}

# Parse args
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      OUTPUT="${2:?Missing value for --output}"
      shift 2
      ;;
    --base)
      BASE="${2:?Missing value for --base}"
      shift 2
      ;;
    --max-bytes)
      MAX_BYTES="${2:?Missing value for --max-bytes}"
      [[ "$MAX_BYTES" =~ ^[0-9]+$ ]] || { echo "ERROR: --max-bytes must be an integer" >&2; exit 1; }
      shift 2
      ;;
    --include-binary)
      INCLUDE_BINARY=1
      shift
      ;;
    --append)
      APPEND=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    --) shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Append remaining positional args after --
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "ERROR: No input files. See --help." >&2
  exit 1
fi

# Ensure BASE exists
if [[ ! -d "$BASE" ]]; then
  echo "ERROR: --base directory does not exist: $BASE" >&2
  exit 1
fi

# Prepare output
mkdir -p "$(dirname -- "$OUTPUT")"
if [[ $APPEND -eq 0 ]]; then
  : > "$OUTPUT"
fi

# Write header
{
  echo "# LLM File Bundle"
  echo
  echo "- Generated: $(date -Iseconds)"
  echo "- Base directory: $BASE"
  echo
} >> "$OUTPUT"

# Process files
included=0
skipped=0

for f in "${ARGS[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "WARN: Skipping (not found): $f" >&2
    ((skipped++)) || true
    continue
  fi
  if [[ -d "$f" ]]; then
    echo "WARN: Skipping directory: $f" >&2
    ((skipped++)) || true
    continue
  fi

  # Size filter
  if [[ "$MAX_BYTES" -gt 0 ]]; then
    size=$(wc -c < "$f" | tr -d ' ')
    if (( size > MAX_BYTES )); then
      echo "WARN: Skipping (>$MAX_BYTES bytes): $f (size=$size)" >&2
      ((skipped++)) || true
      continue
    fi
  fi

  # Compute relative path
  rel="$(relpath "$BASE" "$f" 2>/dev/null || echo "$f")"
  lang="$(fence_lang "$f")"

  if is_text_file "$f"; then
    {
      echo "===== FILE: $rel ====="
      if [[ -n "$lang" ]]; then
        echo "\`\`\`$lang"
      else
        echo "\`\`\`"
      fi
      cat -- "$f"
      echo
      echo "\`\`\`"
      echo
    } >> "$OUTPUT"
    ((included++)) || true
  else
    if [[ $INCLUDE_BINARY -eq 1 ]]; then
      {
        echo "===== FILE: $rel (binary, base64-encoded) ====="
        echo "\`\`\`base64"
        # base64 with no assumptions about flags; allow wrapped output
        base64 < "$f"
        echo
        echo "\`\`\`"
        echo
      } >> "$OUTPUT"
      ((included++)) || true
    else
      echo "WARN: Skipping binary file (use --include-binary to include): $f" >&2
      ((skipped++)) || true
    fi
  fi
done

# Footer summary
{
  echo "---"
  echo "Summary: included=$included, skipped=$skipped"
} >> "$OUTPUT"

echo "Done. Wrote bundle to: $OUTPUT"
