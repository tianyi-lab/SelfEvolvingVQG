import sys
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils.infer_utils import *
from utils.qwen_utils import *
import traceback
import torch
import warnings
from pathlib import Path
import json
import copy
import random
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset

random.seed(42)
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


prompt_qs = """Task: Full-Gen.\nAnalyze the image to create a relevant question and its correct answer.\n"""

LEGACY_PROMPT_QS_LIST = [
    "Task: Full-Gen.\nAnalyze the image to create a relevant question and its correct answer.",
    "Task: Full-Gen.\nGenerate a sample question and answer pair based on this image.",
    "Task: Full-Gen.\nDescribe a detail of the image in a question-answer format.",
]

DIVERSE_PROMPT_FAMILIES = [
    """Task: Full-Gen.
Create one image-grounded question-answer pair about a specific visual entity, attribute, or local detail.
Prefer concrete evidence from the image over generic scene-category questions.
Return exactly:
Question: ...
Answer: ...""",
    """Task: Full-Gen.
Create one image-grounded question-answer pair that requires comparing or relating multiple visible elements.
Prefer spatial relations, counts, contrasts, interactions, or object-to-object evidence.
Return exactly:
Question: ...
Answer: ...""",
    """Task: Full-Gen.
Create one image-grounded question-answer pair about a less obvious aspect of the image.
Prefer visible text, function/purpose, cause-effect clues, unusual regions, or fine-grained scene details when present.
Return exactly:
Question: ...
Answer: ...""",
]

DIVERSE_GENERATION_VARIANTS = [
    "Use a direct wh-question. Avoid asking only for the main object, scene type, or dominant color.",
    "Use a different question type than a simple 'what is shown' question, such as how, which, where, why, count, compare, or infer.",
    "Focus on a different image region or evidence type than the most salient central object, while staying answerable from the image.",
]


def build_question_prompt(prompt_idx: int, gen_idx: int, prompt_style: str) -> str:
    if prompt_style == "legacy":
        return LEGACY_PROMPT_QS_LIST[prompt_idx % len(LEGACY_PROMPT_QS_LIST)]

    family = DIVERSE_PROMPT_FAMILIES[prompt_idx % len(DIVERSE_PROMPT_FAMILIES)]
    variant = DIVERSE_GENERATION_VARIANTS[gen_idx % len(DIVERSE_GENERATION_VARIANTS)]
    return f"{family}\nDiversity constraint: {variant}"
prompt_ans = """Based on the visual content, please answer this question.\n"""

if __name__ == '__main__':
    ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="qwen3_4b")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="output/eval/question_gene")
    parser.add_argument("--json_file", type=str, default=os.environ.get("JSON_FILE", ""))
    parser.add_argument("--load_quantized", type=bool, default=False)
    parser.add_argument("--num_gene", type=int, default=3)
    parser.add_argument("--num_prompt", type=int, default=3)
    parser.add_argument("--prompt_style", choices=["diverse", "legacy"], default="diverse",
                        help="Use diverse 3x3 prompt grid or legacy prompts.")

    args = parser.parse_args()
    if not args.json_file:
        raise ValueError("Set --json_file or JSON_FILE to an input questions JSON path.")

    save_dir = args.save_dir
    load_quantized = args.load_quantized
    modeltype = args.model_type
    modelpath = args.model_path
    json_file = args.json_file
    num_gene = args.num_gene
    num_prompt = args.num_prompt
    prompt_style = args.prompt_style

    if modelpath != '':
        parts = Path(modelpath).parts
        # Check if any path component matches checkpoint-X pattern
        import re
        checkpoint_parts = [p for p in parts if re.fullmatch(r'checkpoint-\d+', p)]
        if checkpoint_parts:
            # Find the folder just before the checkpoint component
            checkpoint = checkpoint_parts[-1]
            checkpoint_idx = list(parts).index(checkpoint)
            folder_name = parts[checkpoint_idx - 1] if checkpoint_idx > 0 else Path(modelpath).parent.name
            modeltype = f"{folder_name}_{checkpoint}"
        else:
            modeltype = Path(modelpath).name
    else:
        modelpath = load_model_path(modeltype)

    Dict_config_m = load_models(model_path=modelpath, )

    os.makedirs(save_dir, exist_ok=True)

    # load data
    with open(args.json_file, "r") as f:
        list_data = json.load(f)[:100]
            
    #############################
    # load model & tokenizer

    list_exist = load_exist_file(save_dir=save_dir, save_name=modeltype, print_log=False)

    if list_exist is not None:
        list_exist_id = [item['index'] for item in list_exist]
        list_eval = [item for i, item in enumerate(list_data) if item.get('index', i) not in list_exist_id]
        list_result = list_exist
        print(f"Found existing result with {len(list_exist_id)} records. Generating for rest {len(list_eval)} samples")
    else:
        list_result = []
        list_eval = list_data

    #############################
    # Start inference
    for i, meta in enumerate(tqdm(list_eval)):
        try:
            index = meta['index']
            image_path = meta['image']

            image, image_str = load_image(image_path=image_path)

            # generations[p][g]: num_prompt prompt runs x num_gene question generations
            generations = []
            for p in range(num_prompt):
                prompt_run = []
                for g in range(num_gene):
                    cur_prompt = build_question_prompt(p, g, prompt_style)
                    Dict_inputs = {'prompt': cur_prompt, 'image': image, 'image_str': image_str}
                    output_res = model_infer(dict_input=Dict_inputs, dict_config_m=Dict_config_m, print_content=False, max_len=4096)
                    content_qs = output_res['content']
                    if isinstance(content_qs, list):
                        content_qs = content_qs[0]

                    if not isinstance(content_qs, str):
                        continue

                    gene_qs = content_qs
                    gene_ans_ori = ''
                    if 'Answer' in content_qs:
                        gene_qs = content_qs.split('Answer')[0].replace('Question:', '').strip()
                        gene_ans_ori = content_qs.split('Answer')[1].strip()

                    content_ans = ''
                    prompt_run.append({"gene_question": gene_qs, "gene_answer": content_ans, "gene_answer_ori": gene_ans_ori, 'prompt': cur_prompt})
                generations.append(prompt_run)

            dict_cur = {"index": index, "image_path": image_path, "generations": generations}
            list_result.append(dict_cur)
            write_to_file(save_dir=save_dir, save_name=modeltype, result=list_result, print_log=False)

        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
            traceback.print_exc()
            sys.exit(-1)

    # save results to json                
    write_to_file(save_dir=save_dir, save_name=modeltype, result=list_result, print_log=False)
