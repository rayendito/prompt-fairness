import os, json, argparse
import lm_eval # type: ignore
import pandas as pd
import numpy as np
from src.Questions import Questions
from src.lmharness_utils import *

parser = argparse.ArgumentParser()
task_manager = lm_eval.tasks.TaskManager()
PASSED_ANNOTATORS = "passed_annotator_objects"
questions = Questions().get_questions()

parser.add_argument("--run_name", type=str, required=True, help="run name")
parser.add_argument("--model_string", type=str, required=True, help="path to the HF VLLM model")
parser.add_argument("--question_indices", type=str, help="question indices (string, separated by comma) or 'all'")
args = parser.parse_args()

if args.question_indices != "all":
    args.question_indices = [int(d) for d in args.question_indices.split(",")] 

results_dir = f"lm_harness_runs/{args.run_name}"
os.makedirs(results_dir, exist_ok=True)

for anjson in sorted(os.listdir(PASSED_ANNOTATORS),
                     key=lambda x: int(x.replace("annotator", "").replace(".json", ""))
):
    ann_obj = json.load(open(os.path.join(PASSED_ANNOTATORS, anjson)))
    tasks_to_run = []
    
    if (args.question_indices == "all"):
        args.question_indices = [i for i in range(len(ann_obj['answers']))]

    for qi in args.question_indices:
        local_tasks = []
        q = list(ann_obj['answers'].keys())[qi]
        if(not pd.isna(ann_obj['answers'][q]['answer'])):
            ANNOTATOR_PROMPT = ann_obj['answers'][q]['answer']
            PLACEHOLDERS = questions[q]['placeholders']
            TASK_EVAL_TYPE = questions[q]['task_eval_type']
            LABELS = questions[q]['labels']
            CHOICES = questions[q]['choices']
            
            for dataset_file in questions[q]['dataset']:
                tskdef = create_task_def(
                    dataset_file=dataset_file,
                    task_type=TASK_EVAL_TYPE,
                    question_index=qi,
                    annotator_prompt=ANNOTATOR_PROMPT,
                    placeholders=PLACEHOLDERS,
                    labels=LABELS,
                    choices=CHOICES
                )
                local_tasks.append(tskdef)
            tasks_to_run += local_tasks

    print(f"Annotator id : {ann_obj['id']}", end=" === ")
    print(f"Tasks to run : {len(tasks_to_run)}")
    
    if(len(tasks_to_run) > 0):
        results = lm_eval.simple_evaluate(
            model="vllm",
            model_args = {
                "pretrained" : args.model_string
            },
            tasks=tasks_to_run,
            num_fewshot=0,
            task_manager=task_manager,
            limit=10,
            write_out = True,
            log_samples = True,
        )
        if len(args.question_indices) == len(ann_obj['answers']):
            quantifier = "all"
        else:
            quantifier = "-".join(args.question_indices)
        with open(os.path.join(results_dir, f"annotator{ann_obj['id']}-question_{quantifier}.json"), "w+") as resf:
            resf.write(json.dumps(results, indent=4, ensure_ascii=False, default=convert_to_json_friendly)) 
    else:
        print(f"no tasks to run for annotator{ann_obj['id']}!")
