#!/bin/bash

#SBATCH --job-name=v6_short_eval
#SBATCH --output=slurm_output/v6_short_eval_%j.log
#SBATCH --error=slurm_output/v6_short_eval_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_FOLDER="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_FOLDER}"

if [[ -f "${ROOT_FOLDER}/config/key.conf" ]]; then
    source "${ROOT_FOLDER}/config/key.conf"
fi

mkdir -p slurm_output output/question_eval/valid_difficulty

ACTION="${1:-submit}"
MODEL_NAME="${MODEL_NAME:-YOUR_MDOEL}"
RUN_NAME="${RUN_NAME:-test_result}"
INPUT_DIR="${INPUT_DIR:-output/eval/question_gene}"
OUTPUT_DIR="${OUTPUT_DIR:-output/question_eval/valid_difficulty}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
NUM_IMAGES="${NUM_IMAGES:-100}"
BATCH_ID_FILE="${BATCH_ID_FILE:-${OUTPUT_DIR}/batches/${RUN_NAME}_batch_id.txt}"

case "${ACTION}" in
    submit)
        python3 eval/question/valid_difficulty_eval.py submit \
            --input-dir "${INPUT_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --run-name "${RUN_NAME}" \
            --judge-model "${JUDGE_MODEL}" \
            --num-images "${NUM_IMAGES}" \
            --models "${MODEL_NAME}"
        ;;
    status)
        python3 eval/question/valid_difficulty_eval.py status \
            --batch-id-file "${BATCH_ID_FILE}"
        ;;
    collect)
        python3 eval/question/valid_difficulty_eval.py collect \
            --output-dir "${OUTPUT_DIR}" \
            --run-name "${RUN_NAME}" \
            --batch-id-file "${BATCH_ID_FILE}" \
            --models "${MODEL_NAME}"
        ;;
    normalize)
        python3 eval/question/valid_difficulty_eval.py normalize \
            --input-dir "${INPUT_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --run-name "${RUN_NAME}" \
            --num-images "${NUM_IMAGES}" \
            --models "${MODEL_NAME}"
        ;;
    *)
        echo "Usage: bash eval/run_qs_eval.sh {submit|status|collect|normalize}" >&2
        exit 2
        ;;
esac
