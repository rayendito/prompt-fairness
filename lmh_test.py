import os, json, ast
import lm_eval # type: ignore
import pandas as pd
import numpy as np
from src.Questions import Questions
from lm_eval.utils import weighted_f1_score # type: ignore

task_manager = lm_eval.tasks.TaskManager()
PASSED_ANNOTATORS = "passed_annotator_objects"
questions = Questions().get_questions()


def create_doc_to_text(ANNOTATOR_PROMPT, PLACEHOLDERS):
    def doc_to_text(doc):
        rs = ANNOTATOR_PROMPT
        if (len(PLACEHOLDERS) > 0):
            for ph in PLACEHOLDERS:
                rs = rs.replace(ph, doc[ph])
            return rs
        else:
            return rs
    return doc_to_text

def create_doc_to_choice(CHOICES):
    def doc_to_choice(doc):
        return ast.literal_eval(doc[CHOICES])
    return doc_to_choice

def create_doc_to_target(TASK_EVAL_TYPE, LABELS):
    if(TASK_EVAL_TYPE == "multiple_choice"):
        def doc_to_target(doc):
            if(len(LABELS) == 1):
                return int(doc[LABELS[0]])
            else:
                # TODO: implement for the one with more labels
                print(f"{LABELS=}")
                raise NotImplementedError()
    else:
        def doc_to_target(doc):
            if(len(LABELS) == 1):
                return doc[LABELS[0]]
            else:
                # TODO: implement for the one with more labels
                print(f"{LABELS=}")
                raise NotImplementedError()
    return doc_to_target

def convert_to_json_friendly(obj):
    if isinstance(obj, np.int64):
        return int(obj)  # Convert to regular Python int
    else:
        return "NON JSON SERIALIZABLE"

for anjson in sorted(os.listdir(PASSED_ANNOTATORS)):
    ann_obj = json.load(open(os.path.join(PASSED_ANNOTATORS, anjson)))
    tasks_to_run = []
    for i, q in enumerate(ann_obj['answers'].keys()):
        if(not pd.isna(ann_obj['answers'][q]['answer'])):
            ANNOTATOR_PROMPT = ann_obj['answers'][q]['answer']
            PLACEHOLDERS = questions[q]['placeholders']
            TASK_EVAL_TYPE = questions[q]['task_eval_type']
            LABELS = questions[q]['labels']
            CHOICES = questions[q]['choices']

            if (TASK_EVAL_TYPE == "multiple_choice"):
                for dset in questions[q]['dataset']:
                    dset_name = dset.replace(".csv", "")
                    tskdef = {
                        "task": f"question_{i}_{dset_name}",
                        "test_split": "test",
                        "dataset_path": "csv",
                        "dataset_name": None,
                        "doc_to_text": create_doc_to_text(ANNOTATOR_PROMPT, PLACEHOLDERS),
                        "doc_to_target": create_doc_to_target(TASK_EVAL_TYPE, LABELS),
                        "doc_to_choice": create_doc_to_choice(CHOICES),
                        "output_type": TASK_EVAL_TYPE,
                        "dataset_kwargs" : {
                            "data_files" : {
                                "test" : f"{os.getcwd()}/question_artefacts/test_datasets/{dset}"
                            }
                        },
                        "metric_list" : [{
                            "metric" : "f1",
                            "aggregation" : weighted_f1_score,
                            "higher_is_better" : True,
                        }]
                    }
                    tasks_to_run.append(tskdef)

            elif (TASK_EVAL_TYPE == "generate_until"):                
                for dset in questions[q]['dataset']:
                    dset_name = dset.replace(".csv", "")
                    tskdef = {
                        "task": f"question_{i}_{dset_name}",
                        "test_split": "test",
                        "dataset_path": "csv",
                        "dataset_name": None,
                        "doc_to_text": create_doc_to_text(ANNOTATOR_PROMPT, PLACEHOLDERS),
                        "doc_to_target": create_doc_to_target(TASK_EVAL_TYPE, LABELS),
                        "output_type": TASK_EVAL_TYPE,
                        "dataset_kwargs" : {
                            "data_files" : {
                                "test" : f"{os.getcwd()}/question_artefacts/test_datasets/{dset}"
                            }
                        },
                        "metric_list" : [{
                            "metric" : "bleu_dua",
                            "aggregation" : "bleu_dua",
                            "higher_is_better" : True,
                        }]
                    }
                    tasks_to_run.append(tskdef)
            else:
                raise NotImplementedError(f"task type {TASK_EVAL_TYPE} not implemented yet.")

    print(f"{len(tasks_to_run)=}")
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args = {
            "pretrained" : "openai-community/gpt2"
        },
        tasks=[tasks_to_run[2]],
        num_fewshot=0,
        task_manager=task_manager,
        limit=10,
        write_out = True,
        log_samples = True,
    )
    with open("hasil_sampel.json", "w+") as dasda:
        dasda.write(json.dumps(results, indent=4, ensure_ascii=False, default=convert_to_json_friendly)) 
    break
