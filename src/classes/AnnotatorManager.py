import os, json
import pandas as pd
from ..utils.annotator_manager_utils import PROFILE_COLUMNS, get_empty_question_profile, update_question_profile

class AnnotatorManager:
    def __init__(self, questions, annotator_dir):
        self.annotator_objects = []
        
        for annfjs in os.listdir(annotator_dir):
            self.annotator_objects.append(json.load(
                open(f"{annotator_dir}/{annfjs}", "r")
            ))
        
        obj_for_df = []
        for annobj in self.annotator_objects:
            obj_for_df.append({
                **{k:v['answer'] for k, v in annobj['answers'].items()},
                **annobj["profile"],
            })
        
        self.all_df = pd.DataFrame(obj_for_df)

        question_profiles = {}
        for question in questions.keys():
            if question in self.all_df.columns:
                df_nonnan = self.all_df[self.all_df[question].notna()]
                question_profiles[question] = self.get_profile_of_question(df_nonnan)
            
        with open("question_artefacts/question_profiles.json", "w") as asdasd:
            asdasd.write(json.dumps(question_profiles, indent=4))
    
    def get_profile_of_question(self, df):
        empty_profile = get_empty_question_profile()
        for i, row in df.iterrows():
            update_question_profile(empty_profile, row, filler="population")
        return empty_profile
