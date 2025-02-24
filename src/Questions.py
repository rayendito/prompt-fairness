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
                "placeholders" : self.get_placeholders(rd['placeholder']),
                "llm_answers" : {
                    "gpt-4o-mini" : [self.clean_artefacts(rd[f"gpt-4o-mini-ans_{i}"]) for i in range(1, Questions.LLMANS_PER_MODEL+1)],
                    "gpt-4o" : [self.clean_artefacts(rd[f"gpt-4o-ans_{i}"]) for i in range(1, Questions.LLMANS_PER_MODEL+1)],
                    "chatgpt-4o-latest" : [self.clean_artefacts(rd[f"chatgpt-4o-latest-ans_{i}"]) for i in range(1, Questions.LLMANS_PER_MODEL+1)],
                },
                "rejected_avg_scores" : {},
                "dataset" : self.get_test(i)
            }

            for sm in ["gpt-4o-mini-3gram_score", "gpt-4o-3gram_score", "chatgpt-4o-latest-3gram_score"]:
                question_scores = []
                for annobj in rejected_annotators:
                    if(rd['question_raw'] in annobj['answers']):
                        question_scores.append(annobj['answers'][rd['question_raw']][sm])
                self.questions[rd['question_raw']]['rejected_avg_scores'][f"{sm}"] = (sum(question_scores) / len(question_scores)) if len(question_scores) > 0 else None
        
        # with open("question_object.json", "w") as lkj:
        #     lkj.write(json.dumps(self.questions, indent=4))

    def get_questions(self):
        return self.questions

    def get_placeholders(self, raw_placeholder):
        if pd.isna(raw_placeholder):
            return []
        ph = [raw_placeholder.strip()]
        if ("<<" in raw_placeholder):
            ph = raw_placeholder.strip().replace("<<","").replace(">>","").split("-")
        return ph
    
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
