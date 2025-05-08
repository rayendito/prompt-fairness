#TODO: CLEAN THIS UP. WHAT A MESS

import sys, os, json, argparse
# import pandas as pd

from dotenv import load_dotenv
# from src import Questions
from src.utils.metric_utils import remove_placeholders

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../inferinter")))
from ModelManager import ModelManager # type: ignore
load_dotenv()

TO_CHECK = "grammar"
llm_judge_model="chatgpt-4o-latest"
POSTPROCESS_DIR = "/l/users/ryandito.diandaru/lm_harness_postprocess/"

ans_classifier_infer_dir = os.path.join(POSTPROCESS_DIR, f"annotator_{TO_CHECK}_inferoai")
os.makedirs(ans_classifier_infer_dir, exist_ok=True)


def classify_answer_sentence_prompt(answer):
    return f"""
Given this request sentence:
\"{answer}\""

Classify it as either "Declarative", "Interrogative", or "Imperative".

Declarative is a statement-like request. You state a situation that implies a request.
Example:
I really need some help right now.
It would be great if you could stay a bit longer.

Interrogative is a question-like request. You ask like a question, but it's really a polite request.
Example:
Could you help me with this?
Would you mind closing the door?

Imperative is a command-like request. You give a direct command, often softened with "please."
Please pass the salt.
Turn down the volume.

Answer only in one word.
""".strip()


def classify_answer_sentence_prompt_grammaticality(answer):
    return f"""
Given this request sentence:
\"{answer}\""

Decide if it's grammatically correct or incorrect.
Exclude placeholder strings that are not part of the request. e.g. [TEXT] becuse they are going to be replaced with something else.
If it's a request to correct the grammar of some given text, ignore the given text. Only assess the user request string.

Answer only in one word. ("correct" or "incorrect")
""".strip()

batches = []
ANNOTATOR_FILES = "annotation_artefacts/annotation_filtering/passed_annotator_objects"
for filename in os.listdir(ANNOTATOR_FILES):
    filepath = os.path.join(ANNOTATOR_FILES, filename)
    
    prompts = []
    with open(filepath, 'r') as f:
        annotator_data = json.load(f)
    
    for k, v in annotator_data["answers"].items():
        if v['answer']:
            if (TO_CHECK == "answer_type"):
                prompts.append(classify_answer_sentence_prompt(remove_placeholders(v['answer'])))
            elif(TO_CHECK == "grammar"):
                prompts.append(classify_answer_sentence_prompt_grammaticality(v['answer']))
        else:
            prompts.append("<<<SKIP_THIS_PROMPT>>>")
    
    batches.append({
        "config" : {
            "tag" : f"annotator-{annotator_data["id"]}",
            "model_provider" : "openai",
            "model_name" : llm_judge_model,
            "in_one_context" : False,
        },
        "input_prompts" : prompts
    })

grid = {
    "grid_info" : {
        "project_name" : __package__,
        "run_name" : f"ans_classifier_run-{llm_judge_model}",
    },
    "batches" : batches
}

grid_path = f"{ans_classifier_infer_dir}/{grid['grid_info']['run_name']}/submitted_grid.json"
os.makedirs(os.path.dirname(grid_path), exist_ok=True)
with open(grid_path, "w") as gg:
    json.dump(grid, gg, indent=4)


def llm_judge_is_msgs_done_fn(msgs):
    if msgs['chat_history'][0]['content'] == "<<<SKIP_THIS_PROMPT>>>":
        return True
    
    if "sentence_class" in msgs["aggregations"]:
        return True
    else:
        return False

def llm_judge_extract_fn(msgs, model_answer):
    msgs['chat_history'].append({
        "role" : "assistant",
        "content" : model_answer
    })
    
    classification = None
    if "declarative" in model_answer.lower():
        classification = "declarative"
    elif "interrogative" in model_answer.lower():
        classification = "interrogative"
    elif "imperative" in model_answer.lower():
        classification = "imperative"

    if "sentence_class" not in msgs["aggregations"] and classification:
        msgs["aggregations"]["sentence_class"] = classification
    
    return msgs


def grammar_is_msgs_done_fn(msgs):
    if msgs['chat_history'][0]['content'] == "<<<SKIP_THIS_PROMPT>>>":
        return True
    
    if "grammaticality" in msgs["aggregations"]:
        return True
    else:
        return False

def grammar_extract_fn(msgs, model_answer):
    msgs['chat_history'].append({
        "role" : "assistant",
        "content" : model_answer
    })
    
    classification = None
    if model_answer.lower() == "incorrect":
        classification = "incorrect"
    elif model_answer.lower() == "correct":
        classification = "correct"

    if "grammaticality" not in msgs["aggregations"] and classification:
        msgs["aggregations"]["grammaticality"] = classification
    
    return msgs

modman = ModelManager(grid)
stillempty = modman.run_grid(grid,
                grammar_is_msgs_done_fn,
                grammar_extract_fn,
                results_dir=ans_classifier_infer_dir)

print(f"{stillempty=}")