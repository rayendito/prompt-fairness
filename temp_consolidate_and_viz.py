import os, sys, json, argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from src import AnnotatorManager, Questions
load_dotenv()

# parsers
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="result json file")
parser.add_argument("--question_document_index", type=str, help="question index and document index")
args = parser.parse_args()

print(f"visualizing {args.run_name} {args.question_document_index}")
qdi = args.question_document_index.split("_")
question_index = int(qdi[0])
document_index = int(qdi[1])
model_name_string = args.run_name.split("-")[0]

viz_dir = f"lm_harness_processed/visualizations/{model_name_string}"
demo_dir = f"lm_harness_processed/demography_stats/{model_name_string}"
os.makedirs(viz_dir, exist_ok=True)
os.makedirs(demo_dir, exist_ok=True)
viz_file = os.path.join(viz_dir, f"visualization-{args.run_name}.png")
demo_file = os.path.join(demo_dir, f"demoscores-{args.run_name}.json")

# necessary question objects
questions = Questions().get_questions()
ann_manager = AnnotatorManager(questions, "passed_annotator_objects", refresh_qp_file=True)
questions_profile_count = json.load(open("question_artefacts/question_profiles.json", "r"))
question_list = list(questions_profile_count.keys())

# result object
respath = os.path.join(os.getenv("SAMPLES_RESULTS_DIR"), f"{args.run_name}")
res_json_path = os.path.join(respath, f"results-{args.run_name}-question_{args.question_document_index}.json")

if not os.path.isfile(res_json_path):
    print(f"Error: {res_json_path} does not exist. Make sure to run the experiment first")
    sys.exit(1)

current_question = questions[question_list[int(question_index)]]
results_obj = json.load(open(res_json_path, "r"))
res_dataset  = []
for k, v in results_obj.items():
    res_dataset.append({
        "annotator_id" : int(k.split("-")[0].split("_")[-1]),
        **{
            f"question_{args.run_name}_{score}_score" : v[f'{score},none']
            for score in current_question["metrics"]
        }
    })
df = pd.DataFrame(res_dataset)
merged_df = pd.merge(ann_manager.all_df, df, on='annotator_id', how='left')

# visualization
question_demo_count = questions_profile_count[question_list[int(question_index)]]
question_score_profiles={}
for score in current_question["metrics"]:
    question_score_profiles[score] = ann_manager.get_profile_of_question(merged_df, filler=f"question_{args.run_name}_{score}_score")

profile_categories = list(question_score_profiles[list(question_score_profiles.keys())[0]].keys())
num_chunks = len(question_score_profiles)
width = 0.8 / num_chunks

fig, axes = plt.subplots(2, len(profile_categories), figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})
barlegend = []
barlabels = []
for i, metric in enumerate(question_score_profiles.keys()):
    for j, category in enumerate(profile_categories):
        labels, scores = zip(*question_score_profiles[metric][category].items())
        x = np.arange(len(labels))
        bar = axes[0, j].bar(x + (i - num_chunks / 2) * width, scores, width, label=metric)
        if i == 0:
            axes[0, j].set_title(category)
            axes[0, j].set_xticks(range(len(labels)))
            axes[0, j].set_xticklabels(labels, rotation=45, ha='right')
            axes[0, j].grid(axis='y', linestyle='--', alpha=0.7)
        if j == 0:
            barlegend.append(bar)
            barlabels.append(metric)

fig.legend(barlegend, barlabels)

for i, category in enumerate(profile_categories):
    count_labels, count_values = zip(*question_demo_count[category].items())
    wedges, _ = axes[1, i].pie(count_values, labels=count_values, colors=plt.cm.Paired.colors)
    axes[1, i].set_title(f"{category} count")
    axes[1, i].legend(wedges, count_labels, loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.01))


fig.suptitle(f"{args.run_name.replace(".json", "")} dataset {document_index} Scores by Category", fontsize=16)
plt.tight_layout()
plt.savefig(viz_file, dpi=300)
with open(demo_file, "w") as dmjs:
    dmjs.write(json.dumps(question_score_profiles, indent=4))