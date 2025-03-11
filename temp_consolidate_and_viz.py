import json, argparse
import pandas as pd
import matplotlib.pyplot as plt
from src import AnnotatorManager, Questions

# parsers
parser = argparse.ArgumentParser()
parser.add_argument("--result_json_file", type=str, required=True, help="result json file")
parser.add_argument("--question_number", type=str, required=True, help="question number")
args = parser.parse_args()

# necessary question objects
questions = Questions().get_questions()
ann_manager = AnnotatorManager(questions, "passed_annotator_objects", refresh_qp_file=True)
questions_profile_count = json.load(open("question_artefacts/question_profiles.json", "r"))
question_list = list(questions_profile_count.keys())

# result object
res_json_path = f"lm_harness_runs/{args.result_json_file}"
results_obj = json.load(open(res_json_path, "r"))['results']
res_dataset  = []
for k, v in results_obj.items():
    res_dataset.append({
        "annotator_id" : int(k.split("-")[0].split("_")[-1]),
        f"question_{args.result_json_file}_score" : v['f1,none']
    })
df = pd.DataFrame(res_dataset)
merged_df = pd.merge(ann_manager.all_df, df, on='annotator_id', how='left')

# visualization
question_demo_count = questions_profile_count[question_list[int(args.question_number)]]
question_profile_score = ann_manager.get_profile_of_question(merged_df, filler=f"question_{args.result_json_file}_score")
fig, axes = plt.subplots(2, len(question_profile_score), figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})
for i, (category, values) in enumerate(question_profile_score.items()):
    labels, scores = zip(*values.items())
    axes[0, i].bar(labels, scores, color='skyblue')
    axes[0, i].set_title(category)
    axes[0, i].set_xticks(range(len(labels)))  # Set fixed tick positions
    axes[0, i].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, i].set_ylim(0, 1)
    axes[0, i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Pie chart for annotator count
    count_labels, count_values = zip(*question_demo_count[category].items())
    wedges, _ = axes[1, i].pie(count_values, labels=count_values, colors=plt.cm.Paired.colors)
    axes[1, i].set_title(f"{category} count")
    axes[1, i].legend(wedges, count_labels, loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.01))


fig.suptitle(f"{args.result_json_file.replace(".json", "")} Scores by Category", fontsize=16)
plt.tight_layout()
plt.savefig(f"lm_harness_processed/visualizations/visualization-{args.result_json_file.replace(".json", ".png")}", dpi=300)
with open(f"lm_harness_processed/demography_stats/demoscores-{args.result_json_file}", "w") as dmjs:
    dmjs.write(json.dumps(question_profile_score, indent=4))