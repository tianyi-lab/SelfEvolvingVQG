#!/bin/bash

#SBATCH --job-name=qs_diversity
#SBATCH --output=slurm_output/qs_diversity_%j.log
#SBATCH --error=slurm_output/qs_diversity_%j.log
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_FOLDER="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_FOLDER}"

export CACHE_DIR="${CACHE_DIR:-${ROOT_FOLDER}/.cache/hf}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_DIR}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${CACHE_DIR}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_DIR}/transformers}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${TRANSFORMERS_CACHE}" slurm_output

MODEL_NAME="${MODEL_NAME:-YOUR_MODEL}"
INPUT_JSON="${INPUT_JSON:-output/eval/question_gene/${MODEL_NAME}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval/question_embedding_diversity}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python3 eval/question/question_gene_embedding_diversity.py \
    --input "${INPUT_JSON}" \
    --output-dir "${OUTPUT_DIR}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --batch-size "${BATCH_SIZE}"
