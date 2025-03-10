import json
import pandas as pd
from src import AnnotatorManager, Questions

questions = Questions().get_questions()
ann_manager = AnnotatorManager(questions, "passed_annotator_objects")

res_json_path = "lm_harness_runs/openaigpt2-limit10-question_0.json"
results_obj = json.load(open(res_json_path, "r"))['results']

res_dataset  = []
for k, v in results_obj.items():
    res_dataset.append({
        "annotator_id" : int(k.split("-")[0].split("_")[-1]),
        "question_0_score" : v['f1,none']
    })

df = pd.DataFrame(res_dataset)
merged_df = pd.merge(ann_manager.all_df, df, on='annotator_id', how='left')

asdasd = ann_manager.get_profile_of_question(merged_df, filler="question_0_score")
with open("q1_score_dist.json", "w") as wkek:
    wkek.write(json.dumps(asdasd, indent=4))

# merged_df.to_csv("hmmmmtiga.csv")