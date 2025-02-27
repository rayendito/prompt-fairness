import os, json
# import lm_eval # type: ignore
import pandas as pd
from src.Questions import Questions

PASSED_ANNOTATORS = "passed_annotator_objects"
questions = Questions().get_questions()

for anjson in os.listdir(PASSED_ANNOTATORS):
    ann_obj = json.load(open(os.path.join(PASSED_ANNOTATORS, anjson)))
    for q in ann_obj['answers'].keys():
        if(not pd.isna(ann_obj['answers'][q]['answer'])):
            ANNOTATOR_PROMPT = print(ann_obj['answers'][q]['answer'])
            PLACEHOLDERS = questions[q]['placeholders']
            TASK_TYPE = questions[q]['task_type']
            
            def doc_to_text(doc):
                return_string = ANNOTATOR_PROMPT
                for ph in PLACEHOLDERS:
                    return_string.replace(ph, doc[ph])
                return return_string

            def doc_to_target(doc):
                return 3
            
            tskdef = {
                "task": "pf_sentiment_analysis", # A dict that configures a task
                "test_split": "test", # A dict that configures a task
                "dataset_path": "csv",
                "dataset_name": None,
                "doc_to_text": doc_to_text,
                "doc_to_target": doc_to_text,
                "output_type": TASK_TYPE,
                "dataset_kwargs" : {
                    "data_files" : {
                        "test" : f"{os.getcwd()}/question_artefacts/test_datasets/500-sentiment_3class.csv"
                    }
                }
            }

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

# task_manager = lm_eval.tasks.TaskManager()
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