import os, re
import sacrebleu
from fuzzywuzzy import fuzz
from collections import Counter

# Each of the metric function takes in (as pairs):
# refs : List[str]
# preds: List[str]

def scorefunc_bleu(refs, preds):
    return sacrebleu.corpus_bleu(preds, [refs]).score

def scorefunc_rouge(refs, preds):
    import evaluate
    rouge_evaluate = evaluate.load("rouge")
    return rouge_evaluate.compute(predictions=preds, references=refs)["rougeL"]

def scorefunc_or_custom(refs, preds):
    agg_scores = []
    for ref, pred in zip(refs, preds):
        fuzzy_scores = []
        for r in ref.split("\n"):
            fuzzy_scores.append(fuzz.partial_ratio(r, pred)/100)
        agg_scores.append(max(fuzzy_scores))
    return sum(agg_scores)/len(agg_scores) if len(agg_scores) > 0 else 0

def scorefunc_and_custom(refs, preds):
    agg_scores = []
    for ref, pred in zip(refs, preds):
        fuzzy_scores = []
        for r in ref.split("\n"):
            fuzzy_scores.append(fuzz.partial_ratio(r, pred)/100)
        agg_scores.append(sum(fuzzy_scores)/len(fuzzy_scores) if len(fuzzy_scores) > 0 else 0)
    return sum(agg_scores)/len(agg_scores) if len(agg_scores) > 0 else 0

def absa_parse_aspect_sentiment(input_strings):
    pattern = r'\[([\'"])(.*?)[\'"],\s*([\'"])(.*?)[\'"]\]'
    output = re.findall(pattern, input_strings)
    if not output:
        return {}
    list_of_pairs = [[item[1], item[3]] for item in output]
    return {pa[0] : pa[1] for pa in list_of_pairs}

def scorefunc_absa_f1(refs, preds):
    agg_scores = []
    for ref, pred in zip(refs, preds):
        # adapted from https://github.com/IsakZhang/Generative-ABSA/blob/main/eval_utils.py
        # number of true postive, gold standard, predicted aspect terms
        gold_pt = absa_parse_aspect_sentiment(ref)
        pred_pt = absa_parse_aspect_sentiment(pred)

        n_tp = 0
        n_gold = len(gold_pt)
        n_pred = len(pred_pt)

        for k, v in pred_pt.items():
            if k in gold_pt and gold_pt[k] == v:
                n_tp += 1

        precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        scores = {'precision': precision, 'recall': recall, 'f1': f1}

        agg_scores.append(scores['f1'])
    return sum(agg_scores)/len(agg_scores) if len(agg_scores) > 0 else 0    

def scorefunc_embedding_similarity(refs, preds):
    from sentence_transformers import SentenceTransformer  # type: ignore
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    sent_transformer_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=hf_hub_cache)
    refs = sent_transformer_model.encode(refs)
    preds = sent_transformer_model.encode(preds)
    agg_scores = []
    for ref, pred in zip(refs, preds):
        similarities = sent_transformer_model.similarity(ref, pred)
        agg_scores.append(similarities.item())
    return sum(agg_scores)/len(agg_scores) if len(agg_scores) > 0 else 0

def scorefunc_recipe_score(refs, preds):
    agg_scores = []
    for ref, pred in zip(refs, preds):
        ref_list_of_string = ref.strip().split("\n")
        pred_list_of_string = pred.strip().split("\n")
        
        count_ref = Counter(ref_list_of_string)
        count_pred = Counter(pred_list_of_string)

        intersection = sum(min(count_ref[k], count_pred[k]) for k in count_ref.keys() & count_pred.keys())
        precision = intersection / sum(count_pred.values()) if sum(count_pred.values()) > 0 else 0
        recall = intersection / sum(count_ref.values()) if sum(count_ref.values()) > 0 else 0
        
        if precision + recall == 0:
            agg_scores.append(0.0)
        
        agg_scores.append(2 * (precision * recall) / (precision + recall))

    return sum(agg_scores)/len(agg_scores) if len(agg_scores) > 0 else 0

def scorefunc_mae(refs, preds):
    pass

def scorefunc_ner_f1(refs, preds):
    pass

def scorefunc_recipe_score(refs, preds):
    pass

def scorefunc_requirement_checker(refs, preds):
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
