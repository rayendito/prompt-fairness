import json
import pandas as pd
from collections import Counter

qdf = pd.read_csv("question_artefacts/pfairness_questions_and_llmans.csv", header=0)
LLMANS_PER_MODEL = 5

LLM_ANSWERS = {}
QUESTION_CLEANER = {}

def get_placeholders(raw_placeholder):
    if pd.isna(raw_placeholder):
        return []
    ph = [raw_placeholder.strip()]
    if ("<<" in raw_placeholder):
        ph = raw_placeholder.strip().replace("<<","").replace(">>","").split("-")
    return ph

def remove_placeholders(text, placeholders):
    for p in placeholders:
        text = text.replace(p, "")
    return text

for i, row in qdf.iterrows():
    rd = row.to_dict()
    QUESTION_CLEANER[rd['question_raw']] = {
        "clean" : rd['question_clean'],
        "placeholders" : get_placeholders(rd['placeholder'])
    }
    LLM_ANSWERS[rd['question_clean']] = {
        "gpt-4o-mini" : [rd[f"gpt-4o-mini-ans_{i}"] for i in range(1, LLMANS_PER_MODEL+1)],
        "gpt-4o" : [rd[f"gpt-4o-ans_{i}"] for i in range(1, LLMANS_PER_MODEL+1)],
        "chatgpt-4o-latest" : [rd[f"chatgpt-4o-latest-ans_{i}"] for i in range(1, LLMANS_PER_MODEL+1)],
    }

class Annotator:
    N_QS = 29
    PROFILE_COLUMNS = ['Participant id', 'Time taken', 'Total approvals', 'Gender', 'Highest education level completed', 'Employment-sector', 'English speaking monolingual', 'Age', 'Sex', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality', 'Language', 'Student status', 'Employment status']

    def __init__(self, annotator_number, questions_row, profile_row):
        self.annotator_number = annotator_number
        self.questions_dict = {}

        for k, v in questions_row.to_dict().items():
            self.questions_dict[k] = {
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
                for model_name, model_answers in LLM_ANSWERS[QUESTION_CLEANER[k]['clean']].items():
                    ngram_scores = []
                    for llma in model_answers:
                        annoans_stripped = remove_placeholders(v['answer'], QUESTION_CLEANER[k]['placeholders'])
                        llmans_stripped = remove_placeholders(llma, QUESTION_CLEANER[k]['placeholders'])
                        ngram_scores.append(self.__get_jaccard_ngram_score(llmans_stripped, annoans_stripped, N_GRAM))
                    self.questions_dict[k][f"{model_name}-{N_GRAM}gram_score"] = max(ngram_scores)

    def __get_jaccard_ngram_score(self, text1, text2, n):
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
    
    def save_self(self, target_dir):
        def nan_to_none(obj):
            if isinstance(obj, dict):
                return {k: nan_to_none(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [nan_to_none(v) for v in obj]
            elif isinstance(obj, float) and pd.isna(obj):
                return None
            return obj
        
        self_object = {}
        self_object['id'] = self.annotator_number
        self_object['profile'] = self.profile_dict
        self_object['answers'] = nan_to_none(self.questions_dict)
        
        with open(f"{target_dir}/annotator{self.annotator_number}.json", "w", encoding="utf-8") as f:
            json.dump(self_object, f, indent=4, ensure_ascii=False)
