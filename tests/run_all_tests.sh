#!/bin/bash
#
# Run all 15 model tests on GPU.
# Usage:  bash tests/run_all_tests.sh [--gpu=0]
#
# Run from the repo root:  cd pme_gpu_public && bash tests/run_all_tests.sh
#

set -e

GPU_ARG="${1:---gpu=0}"
JULIA="${JULIA:-julia}"
SCRIPT="run_pme_gpu.jl"
TESTS_DIR="tests"
LOG_DIR="tests/logs"

mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
FAILED_MODELS=""

# ── helpers ──────────────────────────────────────────────────────────────────

run_test() {
    local name="$1"
    local config="$2"
    local model="$3"
    local log="$LOG_DIR/${name}.log"

    printf "%-35s " "$name"

    if "$JULIA" "$SCRIPT" "$config" "$model" "$GPU_ARG" --threads=4 \
         > "$log" 2>&1; then
        echo "PASS  ($(tail -5 "$log" | grep -oP 'Runtime: \K[0-9.]+s' || echo '?'))"
        PASS=$((PASS + 1))
    else
        echo "FAIL  (see $log)"
        FAIL=$((FAIL + 1))
        FAILED_MODELS="$FAILED_MODELS $name"
    fi
}

# ── header ───────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  PME GPU - Model Test Suite"
echo "  GPU: $GPU_ARG   Julia: $($JULIA --version 2>&1)"
echo "============================================================"
echo ""

T0=$SECONDS

# ── Sharp-edge models (generalized matrix method) ───────────────────────────

echo "--- Sharp-edge models (sharp_df=true) ---"
run_test "Isochrone"              "$TESTS_DIR/config_sharp.toml"   "$TESTS_DIR/model_isochrone.toml"
run_test "Miyamoto"               "$TESTS_DIR/config_sharp.toml"   "$TESTS_DIR/model_miyamoto.toml"
run_test "Kuzmin"                 "$TESTS_DIR/config_sharp.toml"   "$TESTS_DIR/model_kuzmin.toml"
echo ""

# ── Isochrone tapered models ────────────────────────────────────────────────

echo "--- Isochrone tapered models ---"
run_test "IsochroneTaperJH"       "$TESTS_DIR/config_tapered.toml" "$TESTS_DIR/model_isochrone_taper_jh.toml"
run_test "IsochroneTaperZH"       "$TESTS_DIR/config_tapered.toml" "$TESTS_DIR/model_isochrone_taper_zh.toml"
run_test "IsochroneTaperTanh"     "$TESTS_DIR/config_tapered.toml" "$TESTS_DIR/model_isochrone_taper_tanh.toml"
run_test "IsochroneTaperExp"      "$TESTS_DIR/config_tapered.toml" "$TESTS_DIR/model_isochrone_taper_exp.toml"
run_test "IsochroneTaperPoly3"    "$TESTS_DIR/config_poly3.toml"   "$TESTS_DIR/model_isochrone_taper_poly3.toml"
echo ""

# ── Miyamoto tapered models ─────────────────────────────────────────────────

echo "--- Miyamoto tapered models ---"
run_test "MiyamotoTaperExp"       "$TESTS_DIR/config_tapered.toml" "$TESTS_DIR/model_miyamoto_taper_exp.toml"
run_test "MiyamotoTaperTanh"      "$TESTS_DIR/config_tanh.toml"    "$TESTS_DIR/model_miyamoto_taper_tanh.toml"
run_test "MiyamotoTaperPoly3"     "$TESTS_DIR/config_poly3.toml"   "$TESTS_DIR/model_miyamoto_taper_poly3.toml"
echo ""

# ── Kuzmin tapered models ───────────────────────────────────────────────────

echo "--- Kuzmin tapered models ---"
run_test "KuzminTaperPoly3"       "$TESTS_DIR/config_poly3.toml"   "$TESTS_DIR/model_kuzmin_taper_poly3.toml"
run_test "KuzminTaperPoly3L"      "$TESTS_DIR/config_tapered.toml" "$TESTS_DIR/model_kuzmin_taper_poly3L.toml"
echo ""

# ── Other models ─────────────────────────────────────────────────────────────

echo "--- Other models ---"
run_test "Toomre"                 "$TESTS_DIR/config_toomre.toml"  "$TESTS_DIR/model_toomre.toml"
run_test "ExpDisk"                "$TESTS_DIR/config_expdisk.toml" "$TESTS_DIR/model_expdisk.toml"
echo ""

# ── summary ──────────────────────────────────────────────────────────────────

ELAPSED=$(( SECONDS - T0 ))
echo "============================================================"
echo "  Results:  $PASS passed,  $FAIL failed   (${ELAPSED}s total)"
if [ $FAIL -gt 0 ]; then
    echo "  FAILED:  $FAILED_MODELS"
fi
echo "  Logs in:  $LOG_DIR/"
echo "============================================================"

exit $FAIL
