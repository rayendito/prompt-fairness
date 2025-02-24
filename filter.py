import pandas as pd
from src import Annotator, Questions

final_df = pd.read_csv("annotation_result/final_annotation.csv")
columns = final_df.columns.tolist()

questions = Questions(rejected_dir_for_threshold_calc = "rejected_annotator_objects").get_questions()

accepted = 0
rejected = 0

for i, row in final_df.iterrows():
    ann = Annotator(questions, i, row[2:31], row[31:])
    ann.filter_person_suspected_AI()
    if (ann.cheating_indication):
        ann.save_self("failed_annotator_objects")
        rejected += 1
    else:
        ann.save_self("passed_annotator_objects")
        accepted += 1

print(f"{accepted=}")
print(f"{rejected=}")
