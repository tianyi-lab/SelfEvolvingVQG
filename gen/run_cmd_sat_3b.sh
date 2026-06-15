#!/bin/bash

#SBATCH --array=0-4
#SBATCH --job-name=sat_3b
#SBATCH --output=slurm_output/sat_3b_%A_%a.log
#SBATCH --error=slurm_output/sat_3b_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

if [[ -n "${MODULE_LOAD:-}" ]]; then
    source /etc/profile.d/modules.sh
    module add "${MODULE_LOAD}"
fi

if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_FOLDER="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROOT_FOLDER
cd "${ROOT_FOLDER}"

if [[ -f "${ROOT_FOLDER}/config/key.conf" ]]; then
    source "${ROOT_FOLDER}/config/key.conf"
fi

: "${JSON_FILE:?Set JSON_FILE to the input questions JSON file}"
Q_TYPE="sat"

FILTER_MODE="${FILTER_MODE:-filtered}"
EVOLUTION_MODE="${EVOLUTION_MODE:-evolved}"
NUM_TURN="${NUM_TURN:-2}"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_TYPE="${MODEL_TYPE:-qwen25_3b}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-pipeline_rag_sat_2nd_3b_part}"
FILTER_MODEL_PATH="${FILTER_MODEL_PATH:-}"
FILTER_MODEL_TYPE="${FILTER_MODEL_TYPE:-qwen25_3b}"
RAG_BANK="${RAG_BANK:-${ROOT_FOLDER}/output/rag/sample_questions_2nd.jsonl}"
FACT_CACHE="${FACT_CACHE:-${ROOT_FOLDER}/output/rag/facts}"

PART_ID=${SLURM_ARRAY_TASK_ID}
PART_ID="${PART_ID:-0}"

mkdir -p output/data/${OUTPUT_FOLDER}
mkdir -p output/rag/facts
mkdir -p slurm_output


# If MODEL_PATH is set, use it as the model_path and derive model_type from the path name.
if [[ -n "${MODEL_PATH}" ]]; then
    MODEL_TYPE="$(basename "${MODEL_PATH}")"
    EXTRA_ARGS="--model_path=${MODEL_PATH}"
elif [[ "${MODEL_TYPE}" == /* ]] || [[ "${MODEL_TYPE}" == ./* ]]; then
    MODEL_PATH="${MODEL_TYPE}"
    MODEL_TYPE=$(basename "${MODEL_TYPE}")
    EXTRA_ARGS="--model_path=${MODEL_PATH}"
else
    EXTRA_ARGS=""
fi

# If FILTER_MODEL_PATH is set, pass it as filter_model_path.
if [[ -n "${FILTER_MODEL_PATH}" ]]; then
    FILTER_ARGS="--filter_model_path=${FILTER_MODEL_PATH}"
elif [[ "${FILTER_MODEL_TYPE}" == /* ]] || [[ "${FILTER_MODEL_TYPE}" == ./* ]]; then
    FILTER_ARGS="--filter_model_path=${FILTER_MODEL_TYPE}"
else
    FILTER_ARGS="--filter_model_type=${FILTER_MODEL_TYPE}"
fi

python3 gen/scripts/generation_pipeline.py \
    --model_type=${MODEL_TYPE} \
    ${EXTRA_ARGS} \
    ${FILTER_ARGS} \
    --json_file=${JSON_FILE} \
    --gene_type='all' \
    --postfix=${Q_TYPE} \
    --partition=5 \
    --partition_id=${PART_ID} \
    --num_turn="${NUM_TURN}" \
    --save_dir="output/data/${OUTPUT_FOLDER}" \
    --rag_bank="${RAG_BANK}" \
    --fact_cache_dir="${FACT_CACHE}" \
    --filter_mode="${FILTER_MODE}" \
    --evolution_mode="${EVOLUTION_MODE}"
