import os, json
import lm_eval # type: ignore
import pandas as pd
from src.Questions import Questions

task_manager = lm_eval.tasks.TaskManager()
PASSED_ANNOTATORS = "passed_annotator_objects"
questions = Questions().get_questions()

# for anjson in os.listdir(PASSED_ANNOTATORS):
#     ann_obj = json.load(open(os.path.join(PASSED_ANNOTATORS, anjson)))
#     for i, q in enumerate(ann_obj['answers'].keys()):
#         if(not pd.isna(ann_obj['answers'][q]['answer'])):
#             ANNOTATOR_PROMPT = ann_obj['answers'][q]['answer']
#             PLACEHOLDERS = questions[q]['placeholders']
#             TASK_EVAL_TYPE = questions[q]['task_eval_type']
#             LABELS = questions[q]['labels']
#             CHOICES = questions[q]['choices']
#             breakpoint()
#             def doc_to_text(doc):
#                 rs = ANNOTATOR_PROMPT
#                 if (len(PLACEHOLDERS) > 0):
#                     for ph in PLACEHOLDERS:
#                         rs = rs.replace(ph, doc[ph])
#                     return rs
#                 else:
#                     return ANNOTATOR_PROMPT

#             def doc_to_target(doc):
#                 if(len(LABELS) == 1):
#                     return doc[LABELS[0]]
#                 else:
#                     # TODO: implement for the one with more labels
#                     return doc[LABELS[0]]

            
#             for dset in questions[q]['dataset']:
#                 dset_name = dset.replace(".csv", "")
#                 tskdef = {
#                     "task": f"question_{i}_{dset_name}", # A dict that configures a task
#                     "test_split": "test", # A dict that configures a task
#                     "dataset_path": "csv",
#                     "dataset_name": None,
#                     "doc_to_text": doc_to_text,
#                     "doc_to_target": doc_to_text,
#                     "doc_to_choice": doc_to_text,
#                     "output_type": TASK_EVAL_TYPE,
#                     "dataset_kwargs" : {
#                         "data_files" : {
#                             "test" : f"{os.getcwd()}/question_artefacts/test_datasets/{dset}"
#                         }
#                     }
#                 }
#                 task_dict = lm_eval.tasks.get_task_dict([tskdef], task_manager)
#                 print(task_dict)

#         break
#     break
# # def process_docs(doc):
# #     return doc

# # def pf_d2t_dummy(someinput):
# #     return someinput

# # tskdef = {
# #     "task": "pf_sentiment_analysis", # A dict that configures a task
# #     "test_split": "test", # A dict that configures a task
# #     "dataset_path": "csv",
# #     "dataset_name": None,
# #     "doc_to_text": pf_d2t_dummy,
# #     "doc_to_target": pf_d2t_dummy,
# #     "output_type": "loglikelihood",
# #     "dataset_kwargs" : {
# #         "data_files" : {
# #             "test" : f"{os.getcwd()}/question_artefacts/test_datasets/500-sentiment_3class.csv"
# #         }
# #     }
# # }


# # task_dict = lm_eval.tasks.get_task_dict([tskdef], task_manager)
# # print(task_dict)

# # # task: pf_sentiment_analysis
# # # class: !function task.PFSenta
# # # task: pf_sentiment_analysis
# # # test_split: test
# # # dataset_path: csv
# # # dataset_name: null
# # # doc_to_text: !function utils.pf_d2text
# # # doc_to_target: !function utils.pf_d2target
# # # output_type: loglikelihood
# # # process_docs: !function utils.process_docs
# # # dataset_kwargs:
# # #   data_files:
# # #     test: /Documents/prompt-fairness/question_artefacts/test_datasets/500-sentiment_3class.csv