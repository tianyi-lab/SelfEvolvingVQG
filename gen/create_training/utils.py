import random
from pathlib import Path

ROOT_FOLDER = Path.cwd()

def get_formatted_conversation(task_type, q_input, a_input):
    # 1. Define your instruction pools
    # Keep the "Task: XXX" prefix consistent as a hard anchor for LoRA
    instructions = {
        "VQA": [
            "Use the image to provide a direct answer to the following question.",
            "Based on the visual content, please answer this question.",
            "Give a concise response to the question below using the image."
        ],
        "VQG": [
            "Given the image and the answer provided, write the most likely question.",
            "Create a question that corresponds to the following answer and image.",
            "Based on the image, what question would result in the answer provided below?"
        ],
        "IQA": [
            "Analyze the image to create a relevant question and its correct answer.",
            "Generate a sample question and answer pair based on this image.",
            "Describe a detail of the image in a question-answer format."
        ],
        "IQ": [
            "Generate a relevant question about this image.",
            "Look at the image and create an insightful question.",
            "Come up with a question that can be answered by examining this image.",
        ]
    }

    # 2. Randomly select an instruction for the given task
    instruction_sentence = random.choice(instructions[task_type])

    # 3. Format the conversations
    if task_type == "VQA":
        prompt = f"<image>\n{instruction_sentence}\nQuestion: {q_input}\n"
        target = f"Answer: {a_input}"
    
    elif task_type == "VQG":
        # Note: We put the answer in the prompt to trigger the question generation
        prompt = f"<image>\nTask: VQG.\n{instruction_sentence}\nAnswer: {a_input}\n"
        target = f"Question: {q_input}"
    
    elif task_type == "IQA":
        prompt = f"<image>\nTask: Full-Gen.\n{instruction_sentence}\n"
        target = f"Question: {q_input}\n###\nAnswer: {a_input}"

    elif task_type == "IQ":
        prompt = f"<image>\nTask: IQ.\n{instruction_sentence}\n"
        target = f"Question: {q_input}"

    return [{"from": "human", "value": prompt}, {"from": "gpt", "value": target}]

