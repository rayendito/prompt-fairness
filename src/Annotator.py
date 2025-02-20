import json
import pandas as pd
from collections import Counter

qdf = pd.read_csv("our_questions/questions_and_chatgpt_answers.csv", header=0)
LLM_ANSWERS = {}
QUESTION_CLEANER = {}
for i, row in qdf.iterrows():
    rd = row.to_dict()
    QUESTION_CLEANER[rd['question_raw']] = rd['question_clean']
    LLM_ANSWERS[rd['question_clean']] = {
        "ans1" : rd["ans1"],
        "ans2" : rd["ans2"],
        "ans3" : rd["ans3"],
    }

class Annotator:
    N_QS = 29
    PROFILE_COLUMNS = ['Participant id', 'Time taken', 'Total approvals', 'Gender', 'Highest education level completed', 'Employment-sector', 'English speaking monolingual', 'Age', 'Sex', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality', 'Language', 'Student status', 'Employment status']

    def __init__(self, annotator_number, questions_row, profile_row):
        self.annotator_number = annotator_number
        
        self.questions_dict = {}

        for k, v in questions_row.to_dict().items():
            self.questions_dict[QUESTION_CLEANER[k]] = {
                "answer" : v,
            }
        
        self.profile_dict = profile_row.to_dict()

        if len(self.questions_dict) != Annotator.N_QS:
            raise ValueError("Invalid number of questions!")
        if not(self.__is_profile_valid(self.profile_dict)):
            raise ValueError("Invalid profile keys!")

        N_GRAM = 3
        for k, v in self.questions_dict.items():
            if not pd.isna(v['answer']):
                for ng in range(1, N_GRAM+1):
                    ngram_scores = []
                    for kla, vla in LLM_ANSWERS[k].items():
                        ngram_scores.append(self.__get_jaccard_ngram(vla, v['answer'], ng))
                    self.questions_dict[k][f"ngram_{ng}_score"] = sum(ngram_scores) / len(ngram_scores)
        
        # with open("bzjirrrr.json", "w", encoding="utf-8") as f:
        #     json.dump(self.questions_dict, f, indent=4, ensure_ascii=False)

    def __get_jaccard_ngram(self, text1, text2, n):
        def get_word_ngrams(text, n):
            words = text.split()  # Tokenize by words
            return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

        ngrams1 = get_word_ngrams(text1, n)
        ngrams2 = get_word_ngrams(text2, n)

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union != 0 else 0


    def __is_profile_valid(self, profile):
        profile_keys = profile.keys()
        for pk in Annotator.PROFILE_COLUMNS:
            if pk not in profile_keys:
                return False
        return True
    

    # def take_test
