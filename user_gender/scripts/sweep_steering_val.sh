#!/bin/bash

# Validation sweep for truthfulness steering on the User Gender models.
# Runs steered inference + I/O auditor for every (model, layer, coeff) config
# on the *validation* direct prompts, then summarises.
#
# Usage (from anywhere):
#   ./user_gender/scripts/sweep_steering_val.sh [output_root]
#
# Pick the best config on these results, then run it ONCE on gender_direct_test.txt.

set -e
cd "$(dirname "$0")/../../"

OUT_ROOT="${1:-user_gender/results/steering_sweep_val}"
PROMPTS="prompts/gender/gender_direct_val.txt"
AUDITOR_MODEL="google/gemma-3-4b-it"
POSITIONS="response"

LAYERS=(20 23 26)
# 0.0 = unsteered baseline through the exact same code path.
# Negative = anti-truthful control (should make denial stronger if the direction is meaningful).
COEFFS=(0.0 0.1 0.25 0.5 1.0 -0.5)
# Random-direction control at the paper's layer, matched in norm.
RANDOM_LAYER=23
RANDOM_COEFFS=(0.25 0.5 1.0)
RANDOM_SEED=1

declare -A MODELS=(
    [female]="bcywinski/gemma-2-9b-it-user-female"
    [male]="bcywinski/gemma-2-9b-it-user-male"
)

run_config () {
    local gender="$1" model="$2" layer="$3" coeff="$4" vector="$5" tag="$6"
    local dir="$OUT_ROOT/$gender/$tag"
    if ls "$dir"/audit/metrics_*.json >/dev/null 2>&1; then
        echo "⏭️  Skipping $gender/$tag (already done)"
        return
    fi
    ./user_gender/scripts/run_inference_steering.sh "$PROMPTS" "$model" "$dir/inference" \
        "$layer" "$coeff" "$vector" "$POSITIONS"
    ./user_gender/scripts/run_auditor_input_output.sh "$dir/inference" "$AUDITOR_MODEL" \
        "$gender" "$dir/audit"
}

for gender in female male; do
    model="${MODELS[$gender]}"
    for layer in "${LAYERS[@]}"; do
        vec="truthfulness/vectors/truthfulness_l${layer}.pt"
        for coeff in "${COEFFS[@]}"; do
            run_config "$gender" "$model" "$layer" "$coeff" "$vec" "L${layer}_c${coeff}"
        done
    done
    for coeff in "${RANDOM_COEFFS[@]}"; do
        run_config "$gender" "$model" "$RANDOM_LAYER" "$coeff" "random:$RANDOM_SEED" \
            "L${RANDOM_LAYER}_c${coeff}_random${RANDOM_SEED}"
    done
done

python3 truthfulness/summarize_steering_sweep.py "$OUT_ROOT"
