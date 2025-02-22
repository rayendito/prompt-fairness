import json
import pandas as pd
from .Questions import Questions

class Annotator:
    N_GRAM = 3
    N_QS = 29
    PROFILE_COLUMNS = ['Participant id', 'Time taken', 'Total approvals', 'Gender', 'Highest education level completed', 'Employment-sector', 'English speaking monolingual', 'Age', 'Sex', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality', 'Language', 'Student status', 'Employment status']

    def __init__(self, annotator_number, questions_row, profile_row):
        self.annotator_number = annotator_number
        self.questions = Questions().get_questions()
        self.answers = {}

        for k, v in questions_row.to_dict().items():
            self.answers[k] = {
                "answer" : v,
            }
        
        self.profile_dict = profile_row.to_dict()

        if len(self.answers) != Annotator.N_QS:
            raise ValueError("Invalid number of questions!")
        if not(self.__is_profile_valid(self.profile_dict)):
            raise ValueError("Invalid profile keys!")

        for k, v in self.answers.items():
            if not pd.isna(v['answer']):
                # for model_name, model_answers in LLM_ANSWERS[QUESTION_CLEANER[k]['clean']].items():
                for model_name, model_answers in self.questions[k]['llm_answers'].items():
                    ngram_scores = []
                    for llma in model_answers:
                        annoans_stripped = self.__remove_placeholders(v['answer'], self.questions[k]['placeholders'])
                        llmans_stripped = self.__remove_placeholders(llma, self.questions[k]['placeholders'])
                        ngram_scores.append(self.__get_jaccard_ngram_score(
                            llmans_stripped.lower(),
                            annoans_stripped.lower(),
                            Annotator.N_GRAM)
                        )
                    self.answers[k][f"{model_name}-{Annotator.N_GRAM}gram_score"] = max(ngram_scores)

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
    
    def __remove_placeholders(self, text, placeholders):
        for p in placeholders:
            text = text.replace(p, "")
        return text
    
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
        self_object['profile'] = nan_to_none(self.profile_dict)
        self_object['answers'] = nan_to_none(self.answers)
        
        with open(f"{target_dir}/annotator{self.annotator_number}.json", "w", encoding="utf-8") as f:
            json.dump(self_object, f, indent=4, ensure_ascii=False)

    def filter_questions(self):
        # TODO: insert actual calculated threshold
        THRESHOLD  = 0.2
        
        # TODO: do actual score calculation
        def __get_score(ans):
            return 0.5
        
        for ques in self.answers.keys():
            if __get_score(self.answers[ques]) > THRESHOLD:
                self.answers[ques]['answer'] = None

    def run_prompts(self):
        pass
