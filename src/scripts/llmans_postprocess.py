import json
import pandas as pd

SAMPLE_FOR_EACH_QUESTION = 5
QUESTIONS_FILE = "question_artefacts/pfairness_questions_only.csv"
PROMPT_HISTORY_FILE =  "question_artefacts/history-openai_answers.json"

questions_df = pd.read_csv(QUESTIONS_FILE, header=0)
prompt_history = json.load(open(PROMPT_HISTORY_FILE, "r"))

for batch in prompt_history.values():
    questions = []
    agent_answers = []
    
    for conversation in batch['messages']:
        questions.append(conversation[0]['content'])
        agent_answers.append(conversation[1]['content'])

    ansdict = {
        f"ans_{i}" : [] for i in range(1, SAMPLE_FOR_EACH_QUESTION+1)
    }

    for i in range(len(questions_df['question_clean'])):
        begin = i * SAMPLE_FOR_EACH_QUESTION
        chunk = agent_answers[begin:begin+SAMPLE_FOR_EACH_QUESTION]
        for j in range(SAMPLE_FOR_EACH_QUESTION):
            ansdict[f"ans_{j+1}"].append(chunk[j])

    for i in range(1, SAMPLE_FOR_EACH_QUESTION+1):
        questions_df[f"{batch['model_name']}-ans_{i}"] = ansdict[f"ans_{i}"]



questions_df.to_csv('question_artefacts/pfairness_questions_and_llmans.csv', index=False)
