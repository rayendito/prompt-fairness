import os, json

# update annotator files to include linguistic information and score
CHATGPT_CLASSIFIER_RES = "/mnt/data/users/ryandito/lm_harness_postprocess/annotator_answer_type_inferoai/ans_classifier_run-chatgpt-4o-latest/history-ans_classifier_run-chatgpt-4o-latest.json"

with open(CHATGPT_CLASSIFIER_RES, "r") as clsrf:
    chatgpt_linguistic_score_results = json.load(clsrf)

def remove_prefix_suffix(text):
    toremove = [
        "Given this request sentence:\n\"",
        "\"\n\nClassify it as either \"Declarative\", \"Interrogative\", or \"Imperative\".\n\nDeclarative is a statement-like request. You state a situation that implies a request.\nExample:\nI really need some help right now.\nIt would be great if you could stay a bit longer.\n\nInterrogative is a question-like request. You ask like a question, but it's really a polite request.\nExample:\nCould you help me with this?\nWould you mind closing the door?\n\nImperative is a command-like request. You give a direct command, often softened with \"please.\"\nPlease pass the salt.\nTurn down the volume.\n\nAnswer only in one word."
    ]

    for r in toremove:
        text = text.replace(r, "")
    return text

answer_list = []
for v in chatgpt_linguistic_score_results.values():
    local = [{
        "text" : remove_prefix_suffix(m["chat_history"][0]["content"]),
        "classification" : m["aggregations"]["sentence_class"] if "sentence_class" in m["aggregations"] else None
    } for m in v["messages"]]
    answer_list += local

import random
from collections import defaultdict

def stratified_sample(data, n_total, classes):
    class_buckets = defaultdict(list)
    
    # Bucket data by classification
    for item in data:
        if item["classification"] in classes:
            class_buckets[item["classification"]].append(item)

    n_per_class = n_total // len(classes)
    remainder = n_total % len(classes)

    result = []
    for i, cls in enumerate(classes):
        count = n_per_class + (1 if i < remainder else 0)
        result.extend(random.sample(class_buckets[cls], min(count, len(class_buckets[cls]))))

    return result


sampled = stratified_sample(answer_list, 100, ["interrogative", "imperative", "declarative"])

import pandas as pd
df = pd.DataFrame(sampled)
df.to_csv("reqtype_classification_sample.csv")
