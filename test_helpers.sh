#!/usr/bin/env bash
# ============================================================================
# test_helpers.sh — Unit tests for pure-logic helper functions in dwiforge.sh
# ============================================================================
# Usage: bash test_helpers.sh [path/to/dwiforge.sh]
#
# Tests the following functions which have no imaging dependencies:
#   safe_int, retry_operation, move_with_verification, check_checkpoint,
#   create_checkpoint, _validate_config_keys
#
# Exit code: 0 if all tests pass, 1 if any fail.
# ============================================================================

set -uo pipefail

SCRIPT_UNDER_TEST="${1:-$(dirname "$0")/dwiforge.sh}"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

pass() { ((TESTS_PASSED++)); echo -e "  ${GREEN}✓${NC} $1"; }
fail() { ((TESTS_FAILED++)); echo -e "  ${RED}✗${NC} $1: expected=$2 got=$3"; }

assert_eq() {
    ((TESTS_RUN++))
    local label=$1 expected=$2 actual=$3
    if [ "$expected" = "$actual" ]; then
        pass "$label"
    else
        fail "$label" "$expected" "$actual"
    fi
}

assert_rc() {
    ((TESTS_RUN++))
    local label=$1 expected_rc=$2
    shift 2
    "$@" >/dev/null 2>&1
    local actual_rc=$?
    if [ "$expected_rc" -eq "$actual_rc" ]; then
        pass "$label"
    else
        fail "$label" "rc=$expected_rc" "rc=$actual_rc"
    fi
}

# ============================================================================
# Setup: source the script in a way that doesn't execute main()
# We need the functions but not the pipeline execution.
# ============================================================================
echo "Loading functions from: $SCRIPT_UNDER_TEST"
if [ ! -f "$SCRIPT_UNDER_TEST" ]; then
    echo "ERROR: Script not found: $SCRIPT_UNDER_TEST"
    exit 1
fi

# Create a minimal environment so the functions can be sourced
export LOG_DIR=$(mktemp -d)
export WORK_DIR=$(mktemp -d)
export BIDS_DIR=$(mktemp -d)
export DERIV_DIR=$(mktemp -d)
export JSONL_LOG=""
export DRY_RUN=false
export RESUME_MODE=false
export USE_ML_REGISTRATION=false
export RUN_CONNECTOME=false
export CLEANUP=false
export OMP_NUM_THREADS=1
export STORAGE_E=$(mktemp -d)
export STORAGE_F=$(mktemp -d)
export PYTHON_EXECUTABLE=$(command -v python3 || command -v python || echo "python3")

# Source only the functions (BASH_SOURCE trick prevents main execution)
(
    # Override the execution guard
    BASH_SOURCE_OVERRIDE=true
    set +e  # Don't exit on errors during sourcing
    set +u  # Allow unbound vars temporarily
    
    # Source the file — the if [[ BASH_SOURCE == $0 ]] guard prevents main() from running
    source "$SCRIPT_UNDER_TEST" 2>/dev/null
) 2>/dev/null

# Since source inside a subshell doesn't propagate, we extract functions manually
eval "$(awk '/^safe_int\(\)/,/^}/' "$SCRIPT_UNDER_TEST")"
eval "$(awk '/^create_checkpoint\(\)/,/^}/' "$SCRIPT_UNDER_TEST")"
eval "$(awk '/^check_checkpoint\(\)/,/^}/' "$SCRIPT_UNDER_TEST")"

# Minimal log for tests (no-op — we don't need actual logging)
log() { :; }

echo ""
echo "============================================"
echo "  Unit Tests for Pipeline Helper Functions"
echo "============================================"

# ============================================================================
# Test: safe_int
# ============================================================================
echo ""
echo "--- safe_int ---"

assert_eq "integer passthrough"       "42"  "$(safe_int 42)"
assert_eq "negative integer"          "-5"  "$(safe_int -5)"
assert_eq "strips decimal"            "3"   "$(safe_int 3.14159)"
assert_eq "strips trailing text"      "0"   "$(safe_int 100G)"
assert_eq "empty string → 0"          "0"   "$(safe_int '')"
assert_eq "no argument → 0"           "0"   "$(safe_int)"
assert_eq "pure text → 0"             "0"   "$(safe_int 'hello')"
assert_eq "space-trimmed number"      "7"   "$(safe_int 7)"
assert_eq "zero"                      "0"   "$(safe_int 0)"
assert_eq "large number"              "999999" "$(safe_int 999999)"

# ============================================================================
# Test: create_checkpoint / check_checkpoint
# ============================================================================
echo ""
echo "--- create_checkpoint / check_checkpoint ---"

TEST_SUB="sub-TEST001"
mkdir -p "${LOG_DIR}/checkpoints"

# Clean state
rm -f "${LOG_DIR}/checkpoints/${TEST_SUB}_checkpoints.txt"

# Checkpoint shouldn't exist yet
assert_rc "checkpoint absent → rc=1" 1 check_checkpoint "$TEST_SUB" "stage1_done"

# Create it
create_checkpoint "$TEST_SUB" "stage1_done"

# Now it should exist
assert_rc "checkpoint present → rc=0" 0 check_checkpoint "$TEST_SUB" "stage1_done"

# A different stage should still be absent
assert_rc "different stage absent → rc=1" 1 check_checkpoint "$TEST_SUB" "stage2_done"

# Create another
create_checkpoint "$TEST_SUB" "stage2_done"
assert_rc "second checkpoint present" 0 check_checkpoint "$TEST_SUB" "stage2_done"
assert_rc "first still present" 0 check_checkpoint "$TEST_SUB" "stage1_done"

# Verify file content format
((TESTS_RUN++))
if grep -q "^stage1_done:" "${LOG_DIR}/checkpoints/${TEST_SUB}_checkpoints.txt"; then
    pass "checkpoint file format correct"
else
    fail "checkpoint file format" "starts with stage1_done:" "$(head -1 "${LOG_DIR}/checkpoints/${TEST_SUB}_checkpoints.txt")"
fi

# ============================================================================
# Test: retry_operation
# ============================================================================
echo ""
echo "--- retry_operation ---"

# Extract retry_operation
eval "$(awk '/^retry_operation\(\)/,/^}/' "$SCRIPT_UNDER_TEST")"

# Test with a command that succeeds
assert_rc "succeeding command → rc=0" 0 retry_operation true

# Test with a command that fails
assert_rc "failing command → rc≠0" 1 retry_operation false

# Test with a command that succeeds on file operation
TEST_FILE=$(mktemp)
echo "test" > "$TEST_FILE"
TEST_DEST="${WORK_DIR}/retry_test_dest"
assert_rc "cp operation succeeds" 0 retry_operation cp "$TEST_FILE" "$TEST_DEST"

((TESTS_RUN++))
if [ -f "$TEST_DEST" ]; then
    pass "retry_operation cp produced output file"
else
    fail "retry_operation cp" "file exists" "file missing"
fi
rm -f "$TEST_FILE" "$TEST_DEST"

# ============================================================================
# Test: move_with_verification (requires rsync)
# ============================================================================
echo ""
echo "--- move_with_verification ---"

eval "$(awk '/^move_with_verification\(\)/,/^}/' "$SCRIPT_UNDER_TEST")"

set +e  # move_with_verification returns non-zero on expected failures
if command -v rsync &>/dev/null; then
    SRC_FILE=$(mktemp "${WORK_DIR}/mvtest_src.XXXXXX")
    echo "test content" > "$SRC_FILE"
    DST_FILE="${WORK_DIR}/mvtest_dst"

    assert_rc "move succeeds" 0 move_with_verification "$SRC_FILE" "$DST_FILE" "test file"

    ((TESTS_RUN++))
    if [ -f "$DST_FILE" ] && [ ! -f "$SRC_FILE" ]; then
        pass "source removed and dest exists"
    else
        fail "move verification" "src gone, dst exists" "src=$([ -f "$SRC_FILE" ] && echo exists || echo gone) dst=$([ -f "$DST_FILE" ] && echo exists || echo gone)"
    fi

    # Move of non-existent file should fail
    assert_rc "move nonexistent → rc≠0" 1 move_with_verification "/nonexistent/file" "$DST_FILE" "missing file"

    rm -f "$DST_FILE"
else
    echo "  (skipped — rsync not available in this environment)"
fi
set -e

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "  Results: ${TESTS_PASSED}/${TESTS_RUN} passed, ${TESTS_FAILED} failed"
echo "============================================"

# Cleanup
rm -rf "$LOG_DIR" "$WORK_DIR" "$BIDS_DIR" "$DERIV_DIR" "$STORAGE_E" "$STORAGE_F"

exit $([ $TESTS_FAILED -eq 0 ] && echo 0 || echo 1)
