import json
import pandas as pd

SAMPLE_FOR_EACH_QUESTION = 3
QUESTIONS_FILE = "our_questions/questions_only.csv"
PROMPT_HISTORY_FILE =  "our_questions/history-chatgptanswers-sep.json"

questions_df = pd.read_csv(QUESTIONS_FILE, header=0)
prompt_history = json.load(open(PROMPT_HISTORY_FILE, "r"))

questions = []
agent_answers = []

for conversation in prompt_history['runone-gpt_4o_mini-sep-temperature_1']:
    questions.append(conversation[0]['content'])
    agent_answers.append(conversation[1]['content'])

ans1 = []
ans2 = []
ans3 = []
for i in range(len(questions_df['question_clean'])):
    begin = i * SAMPLE_FOR_EACH_QUESTION
    chunk = agent_answers[begin:begin+SAMPLE_FOR_EACH_QUESTION]
    ans1.append(chunk[0])
    ans2.append(chunk[1])
    ans3.append(chunk[2])

# print(ans1[:3])
questions_df['ans1'] = ans1
questions_df['ans2'] = ans2
questions_df['ans3'] = ans3


questions_df.to_csv('questions.csv', index=False)


annotation_result_df = pd.read_csv("annotation_result/collected_prompts_clean.csv")
EXCLUDED_QUESTIONS = [11]
NON_ANSWER_COLUMNS = """
username
Participant id
Time taken
Total approvals
Gender
Highest education level completed
Employment-sector
English speaking monolingual
Age
Sex
Ethnicity simplified
Country of birth
Country of residence
Nationality
Language
Student status
Employment status
""".strip().split()


def remove_rejected(some_df):
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

    new_df.to_csv("annotation_result/final_annotation.csv")

remove_rejected(annotation_result_df)
