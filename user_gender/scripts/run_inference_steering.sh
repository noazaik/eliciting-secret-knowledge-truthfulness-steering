#!/bin/bash

# Usage:
# ./run_inference_steering.sh <prompts_file> <model_name> <output_dir> <layer> <coeff> [vector_path|random:<seed>] [positions]
#
# Examples:
#   ./run_inference_steering.sh prompts/gender/gender_direct_val.txt bcywinski/gemma-2-9b-it-user-female out/ 23 0.25
#   ./run_inference_steering.sh prompts/gender/gender_direct_val.txt bcywinski/gemma-2-9b-it-user-female out/ 23 0.25 random:1
#
# coeff is a fraction of the layer's median residual-stream norm; negative = anti-truthful.
# The output JSON is compatible with run_auditor_input_output.sh.

set -e

PROMPTS_FILE="$1"
MODEL_NAME="$2"
OUTPUT_DIR="$3"
STEER_LAYER_IDX="${4:-23}"
STEER_COEFF="${5:-0.25}"
VECTOR_ARG="${6:-truthfulness/vectors/truthfulness_l${STEER_LAYER_IDX}.pt}"
STEER_POSITIONS="${7:-response}"

# Fixed parameters (same as run_inference_fuzzing.sh)
NUM_RESPONSES=10
MAX_NEW_TOKENS=200
TEMPERATURE=1.0
BATCH_SIZE=10
SEED=1

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ Error: Prompts file '$PROMPTS_FILE' not found"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_INFERENCE_SCRIPT="$SCRIPT_DIR/../../sampling/batch_inference.py"
PROMPTS_FILE="$(realpath "$PROMPTS_FILE")"

if [[ "$VECTOR_ARG" == random:* ]]; then
    VECTOR_FLAGS=(--steer_random_seed "${VECTOR_ARG#random:}")
else
    if [ ! -f "$VECTOR_ARG" ]; then
        echo "❌ Error: steering vector '$VECTOR_ARG' not found"
        exit 1
    fi
    VECTOR_FLAGS=(--steering_vector "$(realpath "$VECTOR_ARG")")
fi

python3 "$BATCH_INFERENCE_SCRIPT" \
    --prompts_file "$PROMPTS_FILE" \
    --model_name "$MODEL_NAME" \
    --enable_steering \
    "${VECTOR_FLAGS[@]}" \
    --steer_layer_idx "$STEER_LAYER_IDX" \
    --steer_coeff "$STEER_COEFF" \
    --steer_positions "$STEER_POSITIONS" \
    --num_responses $NUM_RESPONSES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE

echo "✅ Steering inference completed: layer=$STEER_LAYER_IDX coeff=$STEER_COEFF vector=$VECTOR_ARG"
echo "📁 Results: $OUTPUT_DIR"
