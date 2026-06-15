import sys
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils.qwen_utils import *
import utils.gen.evol_utils as evol_utils
import traceback
import torch
import warnings
import json
import copy
from tqdm import tqdm
from argparse import ArgumentParser
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"



if __name__ == '__main__':
    ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=os.path.join(ROOT_FOLDER, ".cache", "hf"))
    parser.add_argument("--model_type", type=str, default="qwen3_8b")
    # parser.add_argument("--json_file", type=str, default="/path/to/train.json")
    parser.add_argument("--json_file", type=str, default=os.environ.get("JSON_FILE", ""))
    parser.add_argument("--postfix", type=str, default="qinstruct")
    parser.add_argument("--load_quantized", type=bool, default=False)

    args = parser.parse_args()
    if not args.json_file:
        raise ValueError("Set --json_file or JSON_FILE to an input questions JSON path.")

    load_quantized = args.load_quantized
    modeltype = args.model_type
    json_file = args.json_file

    save_dir = "output/data/cases"
    os.makedirs(save_dir, exist_ok=True)

    if "cvb" in json_file:
        postfix = "cvb"
    else:
        postfix = args.postfix

    # load data
    with open(args.json_file, "r") as f:
        input_data = json.load(f)

    # load generation info
    dict_obj_formated = evol_utils.load_obj_info(postfix=postfix, modeltype=modeltype)
    dict_definition, list_task, task_str = evol_utils.load_task_info()
    list_location, location_str = evol_utils.load_location_info()

    #############################
    # load original questions as beginning
    list_data = evol_utils.load_seed_data(
        input_data=input_data,
        postfix=postfix,
        dict_obj_formated=dict_obj_formated,
    )

    #############################
    # load model & tokenizerqwen(modeltype)
    model_path = load_model_path_qwen(modeltype)
    Dict_config_m = load_models_qwen(model_path=model_path, )

    #############################
    # Start inference
    list_res = []
    for i, meta in enumerate(tqdm(list_data)):
        try:
            image_path = meta["image_path"]
            image_id = image_path.split("/")[-1].split(".")[0]

            # load image
            image = load_image_qwen(image_path=image_path, )

            # general_prompt = 'You are a question generation model. Generate one question about the given image. The question must rely on visual information such that its answer would change or be impossible to determine without the image. \n The answer should be plush dinosaur statue. '
            # prompt1 = 'You are a question generation model. Generate one question based on the given image. The answer should be plush dinosaur statue. '
            # prompt1 = 'You are a question generation model. Generate one question based on the given image. The answer should be plush dinosaur statue. '

            prompt1 = 'Generate the scene graph based on the given image'
            scene1, thinking_content = model_infer_qwen(dict_input={'prompt': prompt1, 'image': image}, dict_config_m=Dict_config_m, max_len=37000, print_content=True, do_sample=True, temperature=1.0, top_p=1.0)

            qs_1, thinking_content = model_infer_qwen(dict_input={'prompt': prompt1, 'image': image}, dict_config_m=Dict_config_m, max_len=37000, print_content=True, do_sample=True, temperature=1.0, top_p=1.0)

            prompt1 = 'You are a question generation model. Generate one close-ended question and the corresponding answer based on the given image. '

            qs_1, thinking_content = model_infer_qwen(dict_input={'prompt': prompt1, 'image': image}, dict_config_m=Dict_config_m, max_len=37000, print_content=True, do_sample=True, temperature=1.0, top_p=1.0)

            prompt2 = f'You are a question generation model. Develop one question based on the given image and the given questions: {qs_1}. The question should include new information: woman taking a photo. The answer of the generated question should be exactly: plush dinosaur statue.'
            qs_2, thinking_content = model_infer_qwen(dict_input={'prompt': prompt2, 'image': image}, dict_config_m=Dict_config_m, max_len=37000, print_content=True, do_sample=True, temperature=1.0, top_p=1.0)

            prompt3 = f'You are a question generation model. Develop one question based on the given image, given answer, and the given questions: {qs_2}. The question should include the entity from the given question and develop a new relationship with the given answer. The answer of the generated question should be exactly: LED lights.'
            qs_3, thinking_content = model_infer_qwen(dict_input={'prompt': prompt3, 'image': image}, dict_config_m=Dict_config_m, max_len=37000, print_content=True, do_sample=True, temperature=1.0, top_p=1.0)
            
            import pdb;pdb.set_trace()
            list_res.append({'image_path': image_path, 'instruction': general_prompt, 'generated': content})
            write_to_file(save_dir=save_dir, modeltype=f"test", result=list_res, print_log=False)

        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
            traceback.print_exc()
            sys.exit(-1)

    # save results to json                
    write_to_file(save_dir=save_dir, modeltype=f"test", result=list_res, print_log=True)
