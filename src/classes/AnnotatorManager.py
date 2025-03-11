import os, json
import pandas as pd
from ..utils.annotator_manager_utils import PROFILE_COLUMNS, get_empty_question_profile, update_question_profile, reduce_dict

class AnnotatorManager:
    def __init__(self, questions, annotator_dir, refresh_qp_file=False):
        self.annotator_objects = []
        
        for annfjs in os.listdir(annotator_dir):
            self.annotator_objects.append(json.load(
                open(f"{annotator_dir}/{annfjs}", "r")
            ))
        
        obj_for_df = []
        for annobj in self.annotator_objects:
            obj_for_df.append({
                "annotator_id": annobj["id"],
                **{k:v['answer'] for k, v in annobj['answers'].items()},
                **annobj["profile"],
            })
        
        self.all_df = pd.DataFrame(obj_for_df)

        if(refresh_qp_file):
            question_profiles = {}
            for question in questions.keys():
                if question in self.all_df.columns:
                    df_nonnan = self.all_df[["annotator_id", question] + PROFILE_COLUMNS].dropna(subset=[question])
                    question_profiles[question] = AnnotatorManager.get_profile_of_question(df_nonnan, exclude_highschool=True)
                
            with open("question_artefacts/question_profiles.json", "w") as asdasd:
                asdasd.write(json.dumps(question_profiles, indent=4))
    
    @staticmethod
    def get_profile_of_question(df, filler="population", exclude_highschool=False):
        empty_profile = get_empty_question_profile()
        for i, row in df.iterrows():
            update_question_profile(empty_profile, row, filler, exclude_highschool)
        if(filler == "population"):
            return reduce_dict(empty_profile, method="sum")
        elif("score" in filler):
            return reduce_dict(empty_profile, method="avg")
        else:
            return empty_profile