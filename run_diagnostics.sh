#!/bin/zsh
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Optional development convenience. MyApplication() does not need this script.
IMAGE="${1:-AllBalls}"
case "$IMAGE" in
    AllBalls|Table1|Table2|Table3|Table4|Table5|Table6|all) ;;
    *) print -u2 'Usage: zsh run_diagnostics.sh [AllBalls|Table1|...|Table6|all]'; exit 2 ;;
esac

RUN="${POOL_RUN:-${IMAGE}_$(date +%Y%m%d_%H%M%S)}"
case "$RUN" in
    (*[^a-zA-Z0-9_-]*|'')
        print -u2 'POOL_RUN must contain only letters, numbers, underscore or hyphen.'
        exit 2
        ;;
esac

RUN_DIR="diagnostics/runs/$RUN"
LOG="diagnostics/logs/$RUN.log"
SNAPSHOT="diagnostics/logs/${RUN}_source"
if [[ -e "$RUN_DIR" || -e "$LOG" || -e "$SNAPSHOT" ||
      -e "diagnostics/exports/$RUN.zip" ]]; then
    print -u2 "Run already exists: $RUN. Choose a different POOL_RUN."
    exit 2
fi

if [[ ! -f build/CMakeCache.txt ]]; then
    cmake -S . -B build
fi
mkdir -p "$SNAPSHOT"
cp myApplication.cpp CMakeLists.txt dev_main.cpp run_diagnostics.sh .clang-format "$SNAPSHOT/"
cp README.md "$SNAPSHOT/"
cp -R docs "$SNAPSHOT/docs"
cp -R tools "$SNAPSHOT/tools"

# Capture the source before building/running, not after results have been produced.
# Failures remain in diagnostics/logs/ instead of being mistaken for completed runs.
finish_run()
{
    local result=$?
    if (( result != 0 )); then
        print -u2 "Run failed. Log: $LOG; source snapshot: $SNAPSHOT"
    fi
}
trap finish_run EXIT

{
    print "Image selection: $IMAGE"
    print "Run: $RUN"
    print "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    cmake --version | head -n 1
    grep -E '^(OpenCV_DIR|CMAKE_CXX_COMPILER):' build/CMakeCache.txt || true
    print '\nSource checksums:'
    (cd "$SNAPSHOT" && shasum -a 256 myApplication.cpp CMakeLists.txt dev_main.cpp)
} > "$SNAPSHOT/environment.txt"

cmake --build build --parallel 4 2>&1 | tee "$LOG"

if [[ "$IMAGE" == all ]]; then
    POOL_IMAGE='' POOL_RUN="$RUN" POOL_GUI=0 ./build/PoolBallVision 2>&1 | tee -a "$LOG"
else
    POOL_IMAGE="$IMAGE" POOL_RUN="$RUN" POOL_GUI=0 ./build/PoolBallVision 2>&1 | tee -a "$LOG"
fi

mv "$SNAPSHOT" "$RUN_DIR/source"
mv "$LOG" "$RUN_DIR/run.log"
print "\nResults: $RUN_DIR"
print "Start with: $RUN_DIR/INDEX.md"

# Most reviews can use the local run folder; avoid accumulating duplicate ZIPs.
if [[ "${POOL_ARCHIVE:-0}" == 1 ]]; then
    mkdir -p diagnostics/exports
    (cd diagnostics/runs && /usr/bin/zip -qr "../exports/$RUN.zip" "$RUN")
    print "Optional archive: diagnostics/exports/$RUN.zip"
fi
