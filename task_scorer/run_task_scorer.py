import sys, os, json, argparse
import pandas as pd

from dotenv import load_dotenv
from src import Questions
from src.utils import metric_utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../inferinter")))
load_dotenv()

parser = argparse.ArgumentParser()
# TODO: refactor this naming scheme
parser.add_argument("--run_name", type=str, required=True, help="run name")
parser.add_argument("--question_document_index", type=str, help="question index and document index")
args = parser.parse_args()

# getting the results and samples
#TODO: uncomment to this
# respath = os.path.join(os.getenv("SAMPLES_RESULTS_DIR"), f"{args.run_name}") 
# results_path = os.path.join(respath, f"results-{args.run_name}-question_{args.question_document_index}.json")
# samples_path = os.path.join(respath, f"samples-{args.run_name}-question_{args.question_document_index}.json")

#TODO : uncomment this
respath = os.path.join("dummy_lm_harness_runs", f"{args.run_name}")
results_path = os.path.join(respath, f"results-{args.run_name}-question_{args.question_document_index}.json")
samples_path = os.path.join(respath, f"samples-{args.run_name}-question_{args.question_document_index}.json")

results_json = json.load(open(results_path, "r"))
samples_json = json.load(open(samples_path, "r"))

# question object
questions = Questions().get_questions()

# identifying the current question
qdi = args.question_document_index.split("_")
question_index = int(qdi[0])
document_index = int(qdi[1])
current_question, question_attributes = list(questions.items())[question_index]
subnickname = ""
if question_index == 2 or question_index == 25:
    dataset_string = question_attributes['dataset'][document_index]
    subnickname = "-" + dataset_string.split(".")[0].split("-")[-1]

csv_results_dir = f"lm_harness_processed/results_csv/{args.run_name.split("-")[0]}"
os.makedirs(csv_results_dir, exist_ok=True)

if question_attributes['task_eval_type'] == "multiple_choice":
    res_dataset  = []
    for k, v in results_json.items():
        res_dataset.append({
            "annotator_id" : int(k.split("-")[0].split("_")[-1]),
            **{
                f"{args.question_document_index}-{score}_score-{question_attributes['nickname']}{subnickname}" : v[f'{score},none']
                for score in question_attributes["metrics"]
            }
        })
    df = pd.DataFrame(res_dataset).sort_values(by="annotator_id").reset_index(drop=True)
    df.to_csv(os.path.join(csv_results_dir, f"question_{args.question_document_index}.csv"))
elif question_attributes['task_eval_type'] == "generate_until":
    llm_judge_model = "gpt-4o-mini"
    res_dataset  = {}
    for metric in question_attributes["metrics"]:
        if "llm_judge" in metric:
            from ModelManager import ModelManager # type: ignore
            len_of_dataset = None
            llm_as_judge_prompts = []
            annotator_id_order = []
            prompt_function = metric_utils.llm_judge_prompt_general
            if ("_drum" in metric):
                prompt_function = metric_utils.llm_judge_prompt_drum
            elif ("_science_project" in metric):
                prompt_function = metric_utils.llm_judge_prompt_science_project
            elif ("_aeni_school_life" in metric):
                prompt_function = metric_utils.llm_judge_prompt_aeni_school_life

            for key in sorted(samples_json.keys(), key=lambda k: int(k.split("-")[0].split("_")[-1])):
                annotator_id_order.append(int(key.split("-")[0].split("_")[-1]))
                v = samples_json[key]
                # sanity check
                if (len_of_dataset == None):
                    len_of_dataset = len(v)
                else:
                    if(len(v) != len_of_dataset):
                        raise ValueError("len of dataset mismatch!")
                    
                for test_instance in v:
                    llm_as_judge_prompts.append(prompt_function(
                        test_instance["arguments"][0][0],
                        test_instance["doc"][question_attributes["labels"][0]],
                        test_instance["resps"][0][0],
                    ))
                
            # inferinter code
            grid = {
                "grid_info" : {
                    "project_name" : __package__,
                    "run_name" : "llm_jugde_output-"+args.run_name,
                },
                "batches" : [{
                    "config" : {
                        "tag" : f"{llm_judge_model}-run",
                        "model_provider" : "openai",
                        "model_name" : llm_judge_model,
                        "in_one_context" : False,
                    },
                    "input_prompts" : llm_as_judge_prompts
                }]
            }

            grid_path = f"{__package__}/{grid['grid_info']['run_name']}/submitted_grid.json"
            os.makedirs(os.path.dirname(grid_path), exist_ok=True)
            with open(grid_path, "w") as gg:
                json.dump(grid, gg, indent=4)
            
            modman = ModelManager()
            modman.run_grid(grid,
                            metric_utils.llm_judge_is_msgs_done_fn,
                            metric_utils.llm_judge_extract_fn)
            
            # grid_path = f"{__package__}/{grid['grid_info']['run_name']}/submitted_grid.json"
            llm_judge_res_json_file = os.path.join(
                __package__,
                grid['grid_info']['run_name'],
                f"history-{grid['grid_info']['run_name']}.json",
            )

            history = json.load(open(llm_judge_res_json_file, "r"))
            llm_judge_scores = [h["aggregations"]["llm_judge_score"] for h in history[list(history.keys())[0]]["messages"]]
            llm_judge_scores_agg = [sum(llm_judge_scores[i:i+len_of_dataset]) for i in range(0, len(llm_judge_scores), len_of_dataset)]
            
            for annotator_index, score in zip(annotator_id_order, llm_judge_scores_agg):
                res_dataset[annotator_index][
                    f"{args.question_document_index}-llm_judge_score-{question_attributes['nickname']}{subnickname}"
                ] = score

        else:
            scoring_function = None
            if metric == "bleu":
                scoring_function=metric_utils.scorefunc_bleu
            elif metric == "rouge":
                scoring_function=metric_utils.scorefunc_rouge
            elif metric == "or_custom":
                scoring_function=metric_utils.scorefunc_or_custom
            elif metric == "and_custom":
                scoring_function=metric_utils.scorefunc_and_custom
            elif metric == "embedding_similarity":
                scoring_function=metric_utils.scorefunc_embedding_similarity
            elif metric == "absa_f1":
                scoring_function=metric_utils.scorefunc_absa_f1
            elif metric == "recipe_score":
                scoring_function=metric_utils.scorefunc_recipe_score
            elif metric == "pearson_corr":
                scoring_function=metric_utils.scorefunc_pearson_corr
            elif metric == "ner_f1":
                scoring_function=metric_utils.scorefunc_ner_f1
            elif metric == "requirement_checker":
                scoring_function=metric_utils.scorefunc_requirement_checker
            else:
                raise NotImplementedError(f"metric {metric} scoring scheme is not implemented!")
            
            for k, v in samples_json.items():
                annotator_index = int(k.split("-")[0].split("_")[-1])
                if annotator_index not in res_dataset:
                    res_dataset[annotator_index] = {}
                
                refs = []
                preds = []
                for test_instance in v:
                    refs.append(test_instance["doc"][question_attributes["labels"][0]])
                    preds.append(test_instance["resps"][0][0])

                res_dataset[annotator_index][
                    f"{args.question_document_index}-{metric}_score-{question_attributes['nickname']}{subnickname}"
                ] = scoring_function(refs, preds)

    # saving all results
    df_obj_list = [
        {
            "annotator_id" : annotator_idx,
            **scores
        }
        for annotator_idx, scores in res_dataset.items()
    ]
    df = pd.DataFrame(df_obj_list).sort_values(by="annotator_id").reset_index(drop=True)
    csvpth = os.path.join(csv_results_dir, f"question_{args.question_document_index}.csv")
    df.to_csv(csvpth)


# ==========================================================================================

# model_strings = ["meta-llama/Llama-3.2-1B-Instruct"]
# input_prompts = [
#     "what is the capital of France",
#     "what is the capital of Egypt",
#     "what is the capital of Indonesia",
# ]

# batches = [{
#     "config" : {
#         "tag" : f"{ms}-run",
#         "model_provider" : "vllm",
#         "model_name" : ms,
#         "in_one_context" : False,
#     },
#     "input_prompts" : input_prompts
# } for ms in model_strings]

# grid = {
#     "grid_info" : {
#         "project_name" : __package__,
#         "run_name" : "test_revamp_vllm",
#     },
#     "batches" : batches
# }

# grid_path = f"{__package__}/{grid['grid_info']['run_name']}/submitted_grid.json"
# os.makedirs(os.path.dirname(grid_path), exist_ok=True)
# with open(grid_path, "w") as gg:
#     json.dump(grid, gg, indent=4)

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

# modman = ModelManager()
# modman.run_grid(grid, is_msgs_done_fn, extract_fn)