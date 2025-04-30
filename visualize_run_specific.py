import os, json, argparse, math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.classes.Questions import Questions
from src.utils.viz_utils import plot_histogram_by_category

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="result json file")
args = parser.parse_args()

SCORED_ANNOTATOR_DIR = "annotation_artefacts/scored_passed_annotators"
questions = Questions().get_questions()

# making a raw scores (both characteristic and downstream) for each annotator
downstream_score_df = []
for annfilename in os.listdir(SCORED_ANNOTATOR_DIR):
    annotator_filepath = os.path.join(SCORED_ANNOTATOR_DIR, annfilename)
    with open(annotator_filepath, "r") as  ao:
        annotator_object = json.load(ao)
    
    downstream_scores = {}
    for question_key, values in annotator_object["answers"].items():
        if values["answer"] == None:
            continue
        for qid, score_s in values["downstream_scores"][args.run_name].items():
            postfix = questions[question_key]["metrics"][0] # first metric entry is by convention default
            matching_keys = [k for k in score_s if k.endswith(f"{postfix}_score")]
            if not matching_keys:
                raise ValueError(f"No key ends with '{postfix}' for qid {qid}")
            correct_key = matching_keys[0]
            downstream_scores[f"{qid}-sentence_type"] = values["sentence_type"]
            downstream_scores[f"{qid}-sentence_length"] = values["sentence_length"]
            downstream_scores[correct_key] = score_s[correct_key]

    downstream_score_df.append({
        "id" : annotator_object["id"],
        **downstream_scores,
    })

downstream_score_df = sorted(downstream_score_df, key=lambda x: x["id"])
downstream_score_df = pd.DataFrame(downstream_score_df)
downstream_score_df.to_csv(f"lm_harness_processed/raw_scores-{args.run_name}.csv")



score_cols = [col for col in downstream_score_df.columns if col != 'id']

# make length correlation csv
length_correlation_df = []
for i in range(0, len(score_cols), 3):
    task_cols = score_cols[i:i+3]
    subset_df = downstream_score_df[task_cols].dropna()
    task_name = "-".join(task_cols[-1].split("-")[:-2])
    
    pearson_corr = np.corrcoef(subset_df[task_cols[1]],
                               subset_df[task_cols[2]])
    length_correlation_df.append({
        "task_name" : task_name,
        f"{args.run_name}-length_correlation" : pearson_corr[0][-1]
    })
length_correlation_df = pd.DataFrame(length_correlation_df)
length_correlation_df.to_csv(f"lm_harness_processed/prompt_length_correlation-{args.run_name}.csv")


number_of_tasks = int(len(score_cols)/3) # each task has: prompt_kind, length, default score used
fig, axes = plt.subplots(number_of_tasks, 1, figsize=(10, 4 * number_of_tasks))

for i in range(0, len(score_cols), 3):
    task_cols = score_cols[i:i+3]
    subset_df = downstream_score_df[task_cols].dropna()
    task_name = "-".join(task_cols[-1].split("-")[:-2])
    metric_name = task_cols[-1].split("-")[-1]
    
    prompt_type_freqtable = {}
    for _, row in subset_df.iterrows():
        ptype = row.to_dict()[task_cols[0]]
        score = row.to_dict()[task_cols[-1]]
        if ptype in prompt_type_freqtable:
            prompt_type_freqtable[ptype].append(score)
        else:
            prompt_type_freqtable[ptype] = [score]

    prompt_type_freqtable = dict(sorted(prompt_type_freqtable.items(), reverse=True))
    
    min_value = math.floor(min(min(v) for v in prompt_type_freqtable.values()))
    max_value = math.ceil(max(max(v) for v in prompt_type_freqtable.values()))
    bin_edges=np.linspace(min_value,max_value,15)
    plot_histogram_by_category(prompt_type_freqtable, axes[int(i/3)], bin_edges=bin_edges)
    axes[int(i/3)].set_xlabel(metric_name, fontsize=10)
    axes[int(i/3)].set_ylabel(task_name, fontsize=10, rotation=90)

# print(json.dumps(prompt_type_freqtable, indent=4))
fig.subplots_adjust(hspace=0.4)
fig.savefig(f"visualization_results/{args.run_name}-prompt_char_vs_downstream_score.png", bbox_inches='tight')