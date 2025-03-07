import os, ast
import numpy as np
from lm_eval.utils import weighted_f1_score # type: ignore

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

def create_task_def(
        annotator_id,
        dataset_file,
        task_type,
        question_index,
        annotator_prompt,
        placeholders,
        labels,
        choices,
):
    dataset_name = dataset_file.replace(".csv", "")
    task_def = {
        "task": f"annotator_{str(annotator_id)}-question_{question_index}-{dataset_name}",
        "test_split": "test",
        "dataset_path": "csv",
        "dataset_name": None,
        "doc_to_text": create_doc_to_text(annotator_prompt, placeholders),
        "doc_to_target": create_doc_to_target(task_type, labels),
        "output_type": task_type,
        "dataset_kwargs": {
            "data_files": {
                "test": f"{os.getcwd()}/question_artefacts/test_datasets/{dataset_file}"
            }
        },
        "metric_list": [{
            #TODO :adjust to each task's metric
            "metric": "f1" if task_type == "multiple_choice" else "bleu",
            "aggregation": weighted_f1_score if task_type == "multiple_choice" else "bleu",
            "higher_is_better": True,
        }]
    }
    
    if task_type == "multiple_choice":
        task_def["doc_to_choice"] = create_doc_to_choice(choices)

    return task_def
