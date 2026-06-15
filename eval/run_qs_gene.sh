#!/bin/bash

#SBATCH --array=0-0
#SBATCH --job-name=qs_gen
#SBATCH --output=slurm_output/qs_gen_%A_%a.log
#SBATCH --error=slurm_output/qs_gen_%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_FOLDER="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROOT_FOLDER
cd "${ROOT_FOLDER}"

export CACHE_DIR="${CACHE_DIR:-${ROOT_FOLDER}/.cache/hf}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_DIR}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${CACHE_DIR}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_DIR}/transformers}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${TRANSFORMERS_CACHE}" slurm_output
: "${OPENAI_API_KEY:=}"
if [[ -f "${ROOT_FOLDER}/config/key.conf" ]]; then
    source "${ROOT_FOLDER}/config/key.conf"
fi
: "${CUDA_VISIBLE_DEVICES:=0}"


MODELS=(
  "${MODEL_PATH:-${MODEL_TYPE:-qwen25_3b}}"
)

SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
NUM_GENE=3
NUM_PROMPT=3
: "${JSON_FILE:?Set JSON_FILE to the input questions JSON file}"

if [[ -z "${MODEL:-}" ]]; then
    echo "No model configured for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Running model: $MODEL"

if [ "$MODEL" = "qwen3_4b" ]; then
    python3 eval/question/qwen_qs_gene.py --model_type="qwen3_4b" --json_file="${JSON_FILE}" --num_gene=${NUM_GENE} --num_prompt=${NUM_PROMPT}
elif [ "$MODEL" = "qwen25_3b" ]; then
    python3 eval/question/qwen_qs_gene.py --model_type="qwen25_3b" --json_file="${JSON_FILE}" --num_gene=${NUM_GENE} --num_prompt=${NUM_PROMPT}
elif [ "$MODEL" = "qwen25_7b" ]; then
    python3 eval/question/qwen_qs_gene.py --model_type="qwen25_7b" --json_file="${JSON_FILE}" --num_gene=${NUM_GENE} --num_prompt=${NUM_PROMPT}
else
    python3 eval/question/qwen_qs_gene.py --model_path="$MODEL" --json_file="${JSON_FILE}" --num_gene=${NUM_GENE} --num_prompt=${NUM_PROMPT}
fi
