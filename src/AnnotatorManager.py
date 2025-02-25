import os, json
import pandas as pd
from .enums import *

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
        
        all_df = pd.DataFrame(obj_for_df)

        question_profiles = {}
        for question in questions.keys():
            if question in all_df.columns:
                df_nonnan = all_df[all_df[question].notna()]
                question_profiles[question] = self.get_profile_of_question(df_nonnan)
            
        with open("question_artefacts/question_profiles.json", "w") as asdasd:
            asdasd.write(json.dumps(question_profiles, indent=4))
    
    def get_profile_of_question(self, df):
        empty_profile = {
            "employment_sector" : {
                "Health-STEM" : 0,
                "Engineering-STEM" : 0,
                "Humanities" : 0,
            },
            "race" : {
                "White" : 0,
                "Asian" : 0,
                "Mixed" : 0,
                "Black" : 0,
                "Other" : 0,
            },
            "education" : {
                "Highschool" : 0,
                "Graduate" : 0,
                "Postgraduate" : 0,
            },
            "sex" : {
                "male": 0,
                "female": 0,
            },
            "age" : {
                "18-30" : 0,
                "31-40" : 0,
                "41-50" : 0,
            },
            "language" : {
                "L1" : 0,
                "L2" : 0,
            },
        }
        
        PROFILE_COLUMNS = [
            "Employment-sector",
            "Ethnicity simplified",
            "Highest education level completed",
            "Sex",
            "Age",
            "English speaking monolingual",
        ]
        
        for i, row in df.iterrows():
            for pc in PROFILE_COLUMNS:
                if (pc == "Employment-sector"):
                    if row[pc] in ["Medicine"]:
                        empty_profile["employment_sector"]["Health-STEM"] += 1
                    elif row[pc] in ["Manufacturing", "Science, Technology, Engineering & Mathematics", "Technology and Telecommunications", "Manufacturing and Industry"]:
                        empty_profile["employment_sector"]["Engineering-STEM"] += 1
                    elif row[pc] in ["Arts", "Hospitality & Tourism", "Education & Training", "Policing", "Government & Public Administration", "Business Management & Administration", "Legal", "Education and Research", "Hospitality and Tourism"]:
                        empty_profile["employment_sector"]["Humanities"] += 1
                elif (pc == "Ethnicity simplified"):
                    if row[pc] == "White" :
                        empty_profile["race"]["White"] += 1
                    elif row[pc] == "Asian":
                        empty_profile["race"]["Asian"] += 1
                    elif row[pc] == "Mixed":
                        empty_profile["race"]["Mixed"] += 1
                    elif row[pc] == "Black":
                        empty_profile["race"]["Black"] += 1
                    elif row[pc] == "Other":
                        empty_profile["race"]["Other"] += 1
                elif (pc == "Highest education level completed"):
                    if row[pc] in ["High school diploma/A-levels", "Technical/community college", "Highschool"]:
                        empty_profile["education"]["Highschool"] += 1
                    elif row[pc] in ["Undergraduate degree (BA/BSc/other)", "Bachelor degree"]:
                        empty_profile["education"]["Graduate"] += 1
                    elif row[pc] in ["Graduate degree (MA/MSc/MPhil/other)", "Doctorate degree (PhD/other)", "Master degree"]:
                        empty_profile["education"]["Postgraduate"] += 1
                elif (pc == "Sex"):
                    if row[pc] == "Male":
                        empty_profile["sex"]["male"] += 1
                    else:
                        empty_profile["sex"]["female"] += 1
                elif (pc == "Age"):
                    if 18 <= row[pc] and row[pc] <= 30 :
                        empty_profile["age"]["18-30"] += 1
                    elif 31 <= row[pc] and row[pc] <= 40 :
                        empty_profile["age"]["31-40"] += 1
                    elif 41 <= row[pc] and row[pc] <= 50 :
                        empty_profile["age"]["41-50"] += 1
                elif (pc == "English speaking monolingual"):
                    if row[pc] in ["I only know English"]:
                        empty_profile["language"]["L1"] += 1
                    elif row[pc] in ["Guaranteed non English L1"]:
                        empty_profile["language"]["L2"] += 1

        return empty_profile

