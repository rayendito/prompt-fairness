import pandas as pd

annotation_result_df = pd.read_csv("annotation_result/collected_prompts_clean.csv")
EXCLUDED_QUESTIONS = [11]
# NON_ANSWER_COLUMNS = """
# username
# Participant id
# Time taken
# Total approvals
# Gender
# Highest education level completed
# Employment-sector
# English speaking monolingual
# Age
# Sex
# Ethnicity simplified
# Country of birth
# Country of residence
# Nationality
# Language
# Student status
# Employment status
# """.strip().split()

NON_ANSWER_COLUMNS = ['Participant id', 'Time taken', 'Total approvals', 'Gender', 'Highest education level completed', 'Employment-sector', 'English speaking monolingual', 'Age', 'Sex', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality', 'Language', 'Student status', 'Employment status']

# print(annotation_result_df.columns.tolist())


def remove_rejected_and_clean(some_df):
    new_df = pd.DataFrame()
    columnz = some_df.columns
    for i, col in enumerate(columnz):
        if col in NON_ANSWER_COLUMNS:
            new_df[col] = some_df[col].tolist()

        elif("_check" in col):
            if int(col.split("_")[0]) not in EXCLUDED_QUESTIONS:
                verification = some_df.iloc[:, i].tolist()
                answer = some_df.iloc[:, i+1].tolist()

                for j, c in enumerate(verification):
                    if c == "REJECT":
                        answer[j] = None
                new_df[columnz[i+1]] = answer

    new_df = new_df.iloc[:106, :]
    new_df.to_csv("annotation_result/final_annotation.csv")

remove_rejected_and_clean(annotation_result_df)
