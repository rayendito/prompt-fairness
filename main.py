import pandas as pd
from src import Annotator

final_df = pd.read_csv("annotation_result/final_annotation.csv")
columns = final_df.columns.tolist()

# df_answers_only = final_df.iloc[:, 2:31]
# df_profile_only = final_df.iloc[:, 31:]

annotators = []
for i, row in final_df.iterrows():
    ann = Annotator(i, row[2:31], row[31:])
    annotators.append(Annotator(i, row[2:31], row[31:]))
    ann.save_self("annotator_objects")
    # try:
        
    # except Exception as e:
    #     # print(i, row[2:31], row[31:])
    #     print(e)
    # break

