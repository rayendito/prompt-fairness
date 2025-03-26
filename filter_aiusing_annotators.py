import pandas as pd
from src import Annotator, Questions

final_df = pd.read_csv("annotation_result/aiusing_annotators.csv")
columns = final_df.columns.tolist()

annotators = []
questions = Questions().get_questions()
for i, row in final_df.iterrows():
    ann = Annotator(questions, i, row[2:31], row[31:], rejected=True)
    annotators.append(ann)
    ann.save_self()