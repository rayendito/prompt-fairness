import sys, os, json, argparse, ast
import pandas as pd

from dotenv import load_dotenv
from src import Questions
from src.utils import metric_utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../inferinter")))
load_dotenv()

def flatten_dict(d, parent_key=''):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}@{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def run_task_scorer(postprocess_dir, samples_results_dir, run_name, question_document_index, oai_infer_tries=5, inference_type="local"):
    if(inference_type == "local"):
        respath = os.path.join(samples_results_dir, f"{run_name}", f"{run_name}-question{question_document_index}")
        results_path = os.path.join(respath, f"results-{run_name}-question{question_document_index}-question_{question_document_index}.json")
        samples_path = os.path.join(respath, f"samples-{run_name}-question{question_document_index}-question_{question_document_index}.json")

        results_json = json.load(open(results_path, "r"))
        samples_json = json.load(open(samples_path, "r"))
    elif(inference_type == "api"):
        # STATIC, not rly good
        REFERENCE_DIR="/workspace/lm_harness_runs_v2/llama32_1B_Inst"
        REF_NAME="llama32_1B_Inst"
        refpath = os.path.join(REFERENCE_DIR, f"{REF_NAME}-question{question_document_index}")
        reference_path = os.path.join(refpath, f"samples-{REF_NAME}-question{question_document_index}-question_{question_document_index}.json")
        
        hispath = os.path.join(samples_results_dir, f"{run_name}", f"{run_name}-question{question_document_index}")
        history_path = os.path.join(hispath, f"history-{run_name}-question{question_document_index}.json")
        
        reference_json = json.load(open(reference_path, "r"))
        history_json = json.load(open(history_path, "r"))

    # question object
    questions = Questions().get_questions()

    # identifying the current question
    qdi = question_document_index.split("_")
    question_index = int(qdi[0])
    document_index = int(qdi[1])
    current_question, question_attributes = list(questions.items())[question_index]
    
    subnickname = ""
    if len(question_attributes['dataset']) > 1:
        dataset_string = question_attributes['dataset'][document_index]
        subnickname = "-" + dataset_string.split(".")[0].split("-")[-1]

    print(f"Processing : {question_attributes['nickname']}{subnickname}")

    csv_results_dir = os.path.join(postprocess_dir, run_name)
    oai_inference_dir = os.path.join(postprocess_dir, f"{run_name}-inferoai")
    os.makedirs(csv_results_dir, exist_ok=True)
    os.makedirs(oai_inference_dir, exist_ok=True)

    if question_attributes['task_eval_type'] == "multiple_choice":
        if (inference_type == "local"):
            res_dataset  = []
            for k, v in results_json.items():
                res_dataset.append({
                    "annotator_id" : int(k.split("-")[0].split("_")[-1]),
                    **{
                        f"{question_attributes['nickname']}{subnickname}-{question_document_index}-{metric}_score" : v[f'{metric},none']
                        for metric in question_attributes["metrics"]
                    }
                })
            df = pd.DataFrame(res_dataset).sort_values(by="annotator_id").reset_index(drop=True)
            df.dropna(inplace=True)
            df.to_csv(os.path.join(csv_results_dir, f"question_{question_document_index}.csv"), index=False)
        elif(inference_type == "api"):
            res_dataset=[]
            for k, v in history_json.items():
                annotator_id = int(k.split("-")[0].split("_")[-1])
                this_annotator_key = next(
                    rk for rk in reference_json.keys() if rk.startswith(f"annotator_{annotator_id}-")
                )

                # reference_answers_and_labels = reference_json[this_annotator_key]
                reference_answers_and_labels = {
                    doc["arguments"][0][0] : doc["doc"]
                    for doc in reference_json[this_annotator_key]
                }
                reference_answers_and_labels = dict(sorted(reference_answers_and_labels.items()))

                user_answers = {
                    message["chat_history"][0]["content"] : message["chat_history"][1]["content"]
                    for message in v["messages"]
                }
                user_answers = dict(sorted(user_answers.items()))

                reference_answers = []
                reference_choices = []
                for rref in reference_answers_and_labels.values():
                    ans_idx = rref[question_attributes["labels"][0]]
                    choices = ast.literal_eval(rref["choices"])
                    reference_answers.append(choices[ans_idx])
                    reference_choices.append(choices)

                f1_score = metric_utils.parse_strings_for_f1(
                    list(user_answers.values()),
                    reference_answers,
                    reference_choices
                )
                res_dataset.append({
                    "annotator_id" : annotator_id,
                    f"{question_attributes['nickname']}{subnickname}-{question_document_index}-f1_score" : f1_score
                })
            df = pd.DataFrame(res_dataset).sort_values(by="annotator_id").reset_index(drop=True)
            df.dropna(inplace=True)
            df.to_csv(os.path.join(csv_results_dir, f"question_{question_document_index}.csv"), index=False)
        else:
            raise ValueError(f"{inference_type} unknown inference type")
    elif question_attributes['task_eval_type'] == "generate_until":
        llm_judge_model = "gpt-4o"
        
        # initialization
        res_dataset  = {}
        for annotator_index in range(110):
            res_dataset[annotator_index] = {}

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
                elif ("_grammar_sentence" in metric):
                    prompt_function = metric_utils.llm_judge_prompt_grammar_sentence

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
                        "run_name" : f"llm_jugde_output-{run_name}-question{question_document_index}",
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

                grid_path = f"{oai_inference_dir}/{grid['grid_info']['run_name']}/submitted_grid.json"
                os.makedirs(os.path.dirname(grid_path), exist_ok=True)
                with open(grid_path, "w") as gg:
                    json.dump(grid, gg, indent=4)
                
                modman = ModelManager(grid)
                for t in range(oai_infer_tries):
                    print(f"OPENAI INFER TRIAL NO {t+1} / {oai_infer_tries}")
                    empty = modman.run_grid(grid,
                                    metric_utils.llm_judge_is_msgs_done_fn,
                                    metric_utils.llm_judge_extract_fn,
                                    results_dir=oai_inference_dir)
                    if empty == 0:
                        break
                    else:
                        print(f"still has {empty} invalid responses, rerunning...")

                llm_judge_res_json_file = os.path.join(
                    oai_inference_dir,
                    grid['grid_info']['run_name'],
                    f"history-{grid['grid_info']['run_name']}.json",
                )

                history = json.load(open(llm_judge_res_json_file, "r"))
                llm_judge_scores = [h["aggregations"]["llm_judge_score"] for h in history[list(history.keys())[0]]["messages"]]
                llm_judge_scores = [val if 0 <= val <= 1 else 0 for val in llm_judge_scores]
                llm_judge_scores_agg = [sum(llm_judge_scores[i:i+len_of_dataset]) for i in range(0, len(llm_judge_scores), len_of_dataset)]
                
                for annotator_index, score in zip(annotator_id_order, llm_judge_scores_agg):
                    res_dataset[annotator_index][
                        f"{question_attributes['nickname']}{subnickname}-{question_document_index}-{metric}_score"
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
                    refs = []
                    preds = []
                    for test_instance in v:
                        refs.append(test_instance["doc"][question_attributes["labels"][0]])
                        preds.append(test_instance["resps"][0][0])

                    res_dataset[annotator_index][
                        f"{question_attributes['nickname']}{subnickname}-{question_document_index}-{metric}_score"
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
        csvpth = os.path.join(csv_results_dir, f"question_{question_document_index}.csv")
        df.dropna(inplace=True)
        df.to_csv(csvpth, index=False)

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