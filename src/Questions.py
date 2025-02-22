import pandas as pd
import json

class Questions:
    LLMANS_PER_MODEL = 5
    def __init__(self):
        qdf = pd.read_csv("question_artefacts/pfairness_questions_and_llmans.csv", header=0)
        self.questions = {}

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
                "dataset" : self.get_test(i)
            }
    
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
