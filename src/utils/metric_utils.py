import re
from fuzzywuzzy import fuzz

def scorefunc_bleu(refs, preds):
    pass

def scorefunc_rouge(refs, preds):
    pass

def scorefunc_or_custom(refs, preds):
    fuzzy_scores = []
    for r in refs.split("\n"):
        fuzzy_scores.append(fuzz.partial_ratio(r, preds)/100)
    return max(fuzzy_scores)

def scorefunc_and_custom(refs, preds):
    fuzzy_scores = []
    for r in refs.split("\n"):
        fuzzy_scores.append(fuzz.partial_ratio(r, preds)/100)
    return sum(fuzzy_scores)/len(fuzzy_scores)

def scorefunc_absa_f1(refs, preds):
    pass

def scorefunc_squad_f1(refs, preds):
    raise NotImplementedError("scorefunc_squad_f1 not yet implemented")
    pass

def llm_judge_prompt(prompt, refs, preds):
    return f"""

You are a judge that gives a score to a prediction answer based on how accurate it is compared to a reference answer.

Given this instruction right here:
{prompt}


On a scale of 0 (worst) to 1 (best), how would you rate the machine's answer?
Reference Answer: 
{refs}

Prediction Answer:
{preds}

Answer only the XML format of the score: <score>[float]</score>
""".strip()

def llm_judge_is_msgs_done_fn(msgs):
    if "llm_judge_score" in msgs["aggregations"]:
        return True
    else:
        return False

def llm_judge_extract_fn(msgs, model_answer):
    msgs['chat_history'].append({
        "role" : "assistant",
        "content" : model_answer
    })
    
    score_parsed = re.search(r"[-+]?\d*\.\d+|\d+", model_answer)
    if "llm_judge_score" not in msgs["aggregations"] and score_parsed:
        msgs["aggregations"]["llm_judge_score"] = float(score_parsed.group())
    
    return msgs

# def is_msgs_done_fn(msgs):
#     if "test_list" not in msgs["aggregations"]:
#         return False
#     else:
#         if len(msgs["aggregations"]["test_list"]) == 2:
#             return True
#         else:
#             return False

# def extract_fn(msgs, model_answer):
#     msgs['chat_history'].append({
#         "role" : "assistant",
#         "content" : model_answer
#     })
#     capital = model_answer.split()[-1]
#     if "test_list" not in msgs["aggregations"]:
#         msgs["aggregations"]["test_list"] = [capital]
#     else:
#         msgs["aggregations"]["test_list"].append(capital)
    
#     return msgs