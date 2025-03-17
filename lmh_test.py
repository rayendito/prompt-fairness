import os, json, argparse
import lm_eval # type: ignore
import pandas as pd
import numpy as np
from src import Questions
from src.utils.lmharness_utils import *
from dotenv import load_dotenv
load_dotenv()

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

results_dir = f"lm_harness_runs/"
os.makedirs(results_dir, exist_ok=True)

all_tasks = []
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
                    annotator_id = ann_obj['id'],
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

    # print(f"Annotator id : {ann_obj['id']}", end=" === ")
    # print(f"Tasks to run : {len(tasks_to_run)}")
    all_tasks += tasks_to_run
print(f"total tasks to run {len(all_tasks)}")
if(len(all_tasks) > 0):
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args = {
            "pretrained" : args.model_string
        },
        tasks=all_tasks,
        num_fewshot=0,
        task_manager=task_manager,
        write_out = True,
        log_samples = True,
    )
    if len(args.question_indices) == len(ann_obj['answers']):
        quantifier = "all"
    else:
        quantifier = "-".join([str(d) for d in args.question_indices])
    respath = os.path.join(results_dir, f"{args.run_name}")
    respath_samples = os.path.join("/l/users/ryandito.diandaru","prompt_fairness_outputs")
    os.makedirs(respath, exist_ok=True)
    os.makedirs(respath_samples, exist_ok=True)
    with open(os.path.join(respath, f"results-{args.run_name}-question_{quantifier}.json"), "w+") as res_res:
        res_res.write(json.dumps(results["results"], indent=4, ensure_ascii=False, default=convert_to_json_friendly))
    with open(os.path.join(respath_samples, f"samples-{args.run_name}-question_{quantifier}.json"), "w+") as res_sa:
        res_sa.write(json.dumps(results["samples"], indent=4, ensure_ascii=False, default=convert_to_json_friendly))
    with open(os.path.join(respath, f"configs-{args.run_name}-question_{quantifier}.json"), "w+") as res_configs:
        res_configs.write(json.dumps(results["configs"], indent=4, ensure_ascii=False, default=convert_to_json_friendly))
    with open(os.path.join(respath, f"config-{args.run_name}-question_{quantifier}.json"), "w+") as res_config:
        res_config.write(json.dumps(results["config"], indent=4, ensure_ascii=False, default=convert_to_json_friendly))
else:
    print(f"no tasks to run for annotator{ann_obj['id']}!")
