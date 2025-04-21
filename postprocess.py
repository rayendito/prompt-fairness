import os, sys, json, argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from src import AnnotatorManager, Questions
from task_scorer.run_task_scorer import run_task_scorer, flatten_dict
load_dotenv()

POSTPROCESS_DIR = "/l/users/ryandito.diandaru/lm_harness_postprocess/"
ALL_QDI = "5_0 6_0 7_0 8_0 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0 17_0 18_0 19_0 0_0 1_0 4_0 20_0 26_0 26_1 2_0 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 2_10 2_11 2_12 2_13 2_14 3_0 21_0 22_0 23_0 24_0 25_0 25_1 27_0 27_1 27_2 27_3 27_4 27_5 28_0".split()
os.makedirs(POSTPROCESS_DIR, exist_ok=True)

# parsers
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="result json file")
parser.add_argument("--inference_type", type=str, default="separate", help="type of inference: separate, chunk, or oai")
parser.add_argument("--visualize", action="store_true", help="produce visualization")
args = parser.parse_args()

print(f"Postprocessing: {args.run_name}")
if args.inference_type == "separate":
    for qdi in ALL_QDI:
        print(f"Running: {qdi}")
        run_task_scorer(POSTPROCESS_DIR, os.getenv("SAMPLES_RESULTS_DIR"), args.run_name, qdi, oai_infer_tries=100)
elif args.inference_type == "chunk":
    pass
elif args.inference_type == "oai":
    pass
else:
    raise ValueError(f"{args.inference_type} is an unrecognized inference_type")

# Consolidating the results
aggregated_csv = []
for qdi in ALL_QDI:
    qdi_split = qdi.split("_")
    question_index = int(qdi_split[0])
    document_index = int(qdi_split[1])

    # necessary question objects
    questions = Questions().get_questions()
    ann_manager = AnnotatorManager(questions, "annotation_artefacts/passed_annotator_objects", refresh_qp_file=True)
    questions_profile_count = json.load(open("question_artefacts/question_profiles.json", "r"))
    question_list = list(questions_profile_count.keys())

    current_question = questions[question_list[int(question_index)]]
    processed_csv_file = os.path.join(POSTPROCESS_DIR, args.run_name, f"question_{qdi}.csv")
    resdf = pd.read_csv(processed_csv_file)
    merged_df = pd.merge(ann_manager.all_df, resdf, on='annotator_id', how='left')

    fullname = current_question['nickname']
    if len(current_question['dataset']) > 1:
        dataset_string = current_question['dataset'][document_index]
        fullname += "-" + dataset_string.split(".")[0].split("-")[-1]

    aggregated_csv.append({
        "task_name" : fullname,
        "metric" : "respondent_count",
        **flatten_dict(questions_profile_count[question_list[int(question_index)]])
    })

    for score_col in [col for col in resdf.columns if col != "annotator_id"]:
        aggregated_csv.append({
            "task_name" : fullname,
            "metric" : score_col.split("-")[-1],
            **flatten_dict(ann_manager.get_profile_of_question(merged_df, filler=score_col))
        })

# result object
final_csv_dir = os.path.join("lm_harness_processed", f"final-{args.run_name}.csv")
df = pd.DataFrame(aggregated_csv)
df.to_csv(final_csv_dir, index=False)

# visualization ======================================================================================================================
# if (args.visualize):
#     viz_dir = f"lm_harness_processed/visualizations/{args.run_name}"
#     os.makedirs(viz_dir, exist_ok=True)
#     viz_file = os.path.join(viz_dir, f"visualization-{experiment_name}.png")

#     profile_categories = list(question_score_profiles[list(question_score_profiles.keys())[0]].keys())
#     num_chunks = len(question_score_profiles)
#     width = 0.8 / num_chunks

#     fig, axes = plt.subplots(2, len(profile_categories), figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})
#     barlegend = []
#     barlabels = []
#     for i, metric in enumerate(question_score_profiles.keys()):
#         for j, category in enumerate(profile_categories):
#             labels, scores = zip(*question_score_profiles[metric][category].items())
#             x = np.arange(len(labels))
#             bar = axes[0, j].bar(x + (i - num_chunks / 2) * width, scores, width, label=metric)
#             if i == 0:
#                 axes[0, j].set_title(category)
#                 axes[0, j].set_xticks(range(len(labels)))
#                 axes[0, j].set_xticklabels(labels, rotation=45, ha='right')
#                 axes[0, j].grid(axis='y', linestyle='--', alpha=0.7)
#             if j == 0:
#                 barlegend.append(bar)
#                 barlabels.append(metric)

#     fig.legend(barlegend, barlabels)

#     for i, category in enumerate(profile_categories):
#         count_labels, count_values = zip(*question_demo_count[category].items())
#         wedges, _ = axes[1, i].pie(count_values, labels=count_values, colors=plt.cm.Paired.colors)
#         axes[1, i].set_title(f"{category} count")
#         axes[1, i].legend(wedges, count_labels, loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.01))


#     fig.suptitle(f"{args.run_name.replace(".json", "")} dataset {document_index} Scores by Category", fontsize=16)
#     plt.tight_layout()
#     plt.savefig(viz_file, dpi=300)
