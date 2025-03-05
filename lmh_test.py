import os, json, ast
import lm_eval # type: ignore
import pandas as pd
from src.Questions import Questions

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
                        }
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
                        }
                    }
                    tasks_to_run.append(tskdef)
            else:
                print(f"task type {TASK_EVAL_TYPE} not implemented yet.")
            # break
    print(f"{len(tasks_to_run)=}")
    results_yagesya = lm_eval.simple_evaluate(
        model="vllm",
        model_args = {
            "pretrained" : "openai-community/gpt2"
        },
        # tasks=[tasks_to_run[2]],
        tasks=[tasks_to_run[0], tasks_to_run[4]],
        num_fewshot=0,
        task_manager=task_manager,
        limit=10,
        write_out = True,
        log_samples = True,
    )
    with open("hasil_sampel.json", "w+") as dasda:
        dasda.write(json.dumps(results_yagesya, indent=4)) 
    break
# def process_docs(doc):
#     return doc

# def pf_d2t_dummy(someinput):
#     return someinput

# tskdef = {
#     "task": "pf_sentiment_analysis", # A dict that configures a task
#     "test_split": "test", # A dict that configures a task
#     "dataset_path": "csv",
#     "dataset_name": None,
#     "doc_to_text": pf_d2t_dummy,
#     "doc_to_target": pf_d2t_dummy,
#     "output_type": "loglikelihood",
#     "dataset_kwargs" : {
#         "data_files" : {
#             "test" : f"{os.getcwd()}/question_artefacts/test_datasets/500-sentiment_3class.csv"
#         }
#     }
# }


# task_dict = lm_eval.tasks.get_task_dict([tskdef], task_manager)
# print(task_dict)

# # task: pf_sentiment_analysis
# # class: !function task.PFSenta
# # task: pf_sentiment_analysis
# # test_split: test
# # dataset_path: csv
# # dataset_name: null
# # doc_to_text: !function utils.pf_d2text
# # doc_to_target: !function utils.pf_d2target
# # output_type: loglikelihood
# # process_docs: !function utils.process_docs
# # dataset_kwargs:
# #   data_files:
# #     test: /Documents/prompt-fairness/question_artefacts/test_datasets/500-sentiment_3class.csv