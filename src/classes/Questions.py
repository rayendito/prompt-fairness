import os, json
import pandas as pd

class Questions:
    LLMANS_PER_MODEL = 5
    def __init__(self, rejected_dir_for_threshold_calc=None):
        qdf = pd.read_csv("question_artefacts/pfairness_questions_and_llmans.csv", header=0)
        self.questions = {}

        rejected_annotators = []
        if(rejected_dir_for_threshold_calc):
            for annjsp in os.listdir(rejected_dir_for_threshold_calc):
                rejected_annotators.append(json.load(open(f"{rejected_dir_for_threshold_calc}/{annjsp}", "r")))

        for i, row in qdf.iterrows():
            rd = row.to_dict()
            self.questions[rd['question_raw']] = {
                "clean" : rd['question_clean'],
                "llm_answers" : {
                    "gpt-4o-mini" : [self.clean_artefacts(rd[f"gpt-4o-mini-ans_{i}"]) for i in range(1, Questions.LLMANS_PER_MODEL+1)],
                    "gpt-4o" : [self.clean_artefacts(rd[f"gpt-4o-ans_{i}"]) for i in range(1, Questions.LLMANS_PER_MODEL+1)],
                    "chatgpt-4o-latest" : [self.clean_artefacts(rd[f"chatgpt-4o-latest-ans_{i}"]) for i in range(1, Questions.LLMANS_PER_MODEL+1)],
                },
                "rejected_avg_scores" : {},
                "dataset" : rd['dataset_file'].strip().split("\n") if not pd.isna(rd['dataset_file']) else None,
                "placeholders" : self.get_placeholders(rd['placeholder']),
                "task_eval_type" : rd['task_eval_type'].strip() if not pd.isna(rd['task_eval_type']) else None,
                "prompt_injections" : self.get_injections(rd['placeholder']),
                "choices" : rd['choices'].strip() if not pd.isna(rd['choices']) else None,
                "labels" : self.get_cell_list(rd['label']),
            }

            for sm in ["gpt-4o-mini-3gram_score", "gpt-4o-3gram_score", "chatgpt-4o-latest-3gram_score"]:
                question_scores = []
                for annobj in rejected_annotators:
                    if(rd['question_raw'] in annobj['answers']):
                        question_scores.append(annobj['answers'][rd['question_raw']][sm])
                self.questions[rd['question_raw']]['rejected_avg_scores'][f"{sm}"] = (sum(question_scores) / len(question_scores)) if len(question_scores) > 0 else None
        
        with open("question_artefacts/question_object.json", "w") as lkj:
            lkj.write(json.dumps(self.questions, indent=4))

    def get_questions(self):
        return self.questions

    def get_placeholders(self, raw_placeholder):
        if pd.isna(raw_placeholder):
            return []
        ph = [raw_placeholder.strip()]
        if ("<<" in raw_placeholder):
            ph = raw_placeholder.strip().replace("<<","").replace(">>","").split("-")
        return ph
    
    def get_injections(self, raw_placeholder):
        if ((not pd.isna(raw_placeholder)) and "<<" in raw_placeholder):
            return raw_placeholder.strip().replace("<<","").replace(">>","").split("-")
        else:
            return None
    
    def get_cell_list(self, dataset_entry):
        if isinstance(dataset_entry, str):
            return [dataset_entry]
        elif isinstance(dataset_entry, list):
            return dataset_entry.strip().split("\n")
        else:
            return []

    def clean_artefacts(self, llmans):
        to_remove = [
            "You could phrase your request like this:",
            "You could phrase your request as follows:",
            "You might ask:",
            "I would phrase my request as follows:",
            "I would phrase the request as follows:",
            "You can phrase your request like this:",
        ]
        
        for tr in to_remove:
            llmans = llmans.replace(tr, "")

        return llmans
    
    def get_test(self, question_index):
        # TODO: do this lol
        test = {}
        if(question_index == 0):
            test = {}
        # etc
        # elif(question_index == 1):
        
        return test
