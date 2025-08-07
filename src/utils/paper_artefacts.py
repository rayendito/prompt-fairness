import os, json, math
import pandas as pd

from itertools import combinations
from scipy.stats import f_oneway, ttest_ind
from src.classes.Questions import Questions

#====================================================================
# GENERAL MACROS
#====================================================================
FINAL_ANNOTATOR_DIR = "annotation_artefacts/final_annotator_objects"
EXCLUDED_SCORES = ["2_13", "23_", "24_", "25_", "27_"]
EXCLUDED_WHOLE = [23, 24, 25, 27]
LLM_JUDGE_TASKS=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
QUESTIONS = Questions().get_questions()

ALL_MODELS = [
    "gpt_4o",
    "llama31_70b_Inst",
    "llama31_8B_Inst",
    "llama32_1B_Inst",
    "qwen25_72b_Inst",
    "Qwen25_7B_Inst",
    "Qwen25_1dot5B_Inst"
]

DEMOGRAPHY_PROFILES = [
    "employment_sector",
    "race",
    "education",
    "sex",
    "age",
    "language",
    "nationality_group"
]

GROUP_MM_LOOKUP_TABLE_DEFINITION = {
    "summarization" : [3],
    "semantics" : [0,1,4],
    "machine_translation" : [2],
    "knowledge_and_reasoning" : [5,6,7,8,9,10,11,12,13,14,15,18,19],
    "conditioned_text_gen" : [28],
    "syntaxtes_1" : [16,17],
    "syntaxtes_2" : [26],
    "qa_extraction_1" : [20],
    "qa_extraction_2" : [21,22]
}

ALL_COLOR_DICT = {
    "imperative" : "#1f77b4",
    "interrogative" : "#ff7f0e",
    "declarative" : "#2ca02c",
    "correct" : "#6084f7",
    "incorrect" : "#d6425d",
    "white" : "#ffffff",
}

#====================================================================
# PAPER MAIN TABLES 1
#====================================================================
TASK_GROUP_DICT = {}
for i, k in enumerate(QUESTIONS.keys()):
    qtype = QUESTIONS[k]["question_type"]
    if qtype in TASK_GROUP_DICT:
        TASK_GROUP_DICT[qtype].append(i)
    else:
        TASK_GROUP_DICT[qtype] = [i]

stacked_model_df = []
for model in ALL_MODELS:
    model_specific_scores = []
    for annfilename in os.listdir(FINAL_ANNOTATOR_DIR):
        annobj_for_df = {}
        annotator_filepath = os.path.join(FINAL_ANNOTATOR_DIR, annfilename)
        with open(annotator_filepath, "r") as ao:
            annotator_object = json.load(ao)
        
        annobj_for_df["id"] = annotator_object["id"]
        annobj_for_df.update(**annotator_object["profile"])
        
        for i, (question, scores_for_question) in enumerate(annotator_object["answers"].items()):
            if not scores_for_question["answer"]:
                continue
            else:
                for qdi, scores in scores_for_question["downstream_scores"][model].items():
                    annobj_for_df[qdi] = list(scores.values())[0]
        model_specific_scores.append(annobj_for_df)
    score_df = pd.DataFrame(model_specific_scores)
    exclude_cols = ["id"] + DEMOGRAPHY_PROFILES
    task_cols = [col for col in score_df.columns if col not in exclude_cols]
    stacked_model_df.append(score_df[task_cols])

def remove_outliers(list_of_values):
    list_of_values = [x for x in list_of_values if not math.isnan(x)]
    sorted_vals = sorted(list_of_values)
    n = len(sorted_vals)
    
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    new_list_of_values = [x for x in list_of_values if lower <= x <= upper]
    return new_list_of_values

def get_min_max_from_stacked_df(stacked_model_df, clip_llm_judge=True):
    # Stack all DataFrames into one
    combined_df = pd.concat(stacked_model_df, ignore_index=True)

    # Compute min and max for each column
    retdict = {}
    for col in combined_df.columns:
        qid = int(col.split("_")[0])
        no_outlier_combined_df = remove_outliers(list(combined_df[col]))
        retdict[col] = {
            'min': max(0, min(no_outlier_combined_df)) if qid in LLM_JUDGE_TASKS and clip_llm_judge else min(no_outlier_combined_df),
            'max': min(1, max(no_outlier_combined_df)) if qid in LLM_JUDGE_TASKS and clip_llm_judge else max(no_outlier_combined_df)
        }

    return retdict

UNIVERSAL_SCOREDF_MINMAX = get_min_max_from_stacked_df(stacked_model_df) # TO IMPORT =================================================

GROUP_MM_LOOKUP_TABLE = {} # TO IMPORT =================================================
for k in GROUP_MM_LOOKUP_TABLE_DEFINITION:
    group_minmax = {
        "min" : None,
        "max" : None,
    }
    for qdi_full, glob_mm in UNIVERSAL_SCOREDF_MINMAX.items():
        q_idx = int(qdi_full.split("_")[0])
        if q_idx in GROUP_MM_LOOKUP_TABLE_DEFINITION[k]:
            if group_minmax["min"] is None or glob_mm["min"] < group_minmax["min"]:
                group_minmax["min"] = glob_mm["min"]
            if group_minmax["max"] is None or group_minmax["max"] < glob_mm["max"]:
                group_minmax["max"] = glob_mm["max"]
    GROUP_MM_LOOKUP_TABLE[k] = group_minmax
    
INDEX_MM_LOOKUP_TABLE = {} # TO IMPORT =================================================
for k in GROUP_MM_LOOKUP_TABLE_DEFINITION:
    lut = GROUP_MM_LOOKUP_TABLE[k]
    for qid in GROUP_MM_LOOKUP_TABLE_DEFINITION[k]:
        INDEX_MM_LOOKUP_TABLE[qid] = lut


def compute_diversity_score(data_dict, min_val, max_val, clip_llm_judge=False):
    if len(data_dict) < 2:
        return 0  # No diversity if less than 2 elements

    list_of_values = []
    for lv in data_dict.values():
        if clip_llm_judge:
            sanity = [v for v in lv if 0 <= v <= 1]
            # sanity check because some of them are in the range 1-10. if so, scale down
            if sanity == []:
                lv = [v*0.1 for v in lv]
            lv = [v for v in lv if 0 <= v <= 1]
        if max_val == min_val:
            # All values are the same → treat as constant
            avg = 0.5  # or any neutral default
        else:
            lv = [x for x in lv if min_val <= x <= max_val] # removing outliers this way
            normalized = [(x - min_val) / (max_val - min_val) for x in lv]
            avg = sum(normalized) / len(normalized)
        list_of_values.append(avg)
    
    total = []
    for i in range(len(list_of_values)):
        for j in range(i + 1, len(list_of_values)):
            total.append(abs(list_of_values[i] - list_of_values[j]))

    return sum(total) / len(total)

def compute_pairwise_t_test(data_dict, min_val, max_val, clip_llm_judge=False):
    results = {}
    for (key1, values1), (key2, values2) in combinations(data_dict.items(), 2):
        
        if clip_llm_judge:
            sanity1 = [v for v in values1 if 0 <= v <= 1]
            sanity2 = [v for v in values2 if 0 <= v <= 1]
            # sanity check because some of them are in the range 1-10. if so, scale down
            if sanity1 == [] or sanity2 == []:
                values1 = [v*0.1 for v in values1]
                values2 = [v*0.1 for v in values2]
            values1 = [v for v in values1 if 0 <= v <= 1]
            values2 = [v for v in values2 if 0 <= v <= 1]
        
        values1 = [x for x in values1 if min_val <= x <= max_val] # removing outliers this way
        values1 = [(v - min_val) / (max_val - min_val) for v in values1]

        values2 = [x for x in values2 if min_val <= x <= max_val] # removing outliers this way
        values2 = [(v - min_val) / (max_val - min_val) for v in values2]

        stat, p = ttest_ind(values1, values2, equal_var=False)
        results[(key1, key2)] = p
    return results

def compute_anova_score(data_dict, min_val, max_val, clip_llm_judge=False):
    # normalizer
    if max_val == min_val:
        # All values are the same across all groups
        return float("nan"), float("nan")

    groups = []
    for scores in data_dict.values():
        if clip_llm_judge:
            sanity = [v for v in scores if 0 <= v <= 1]
            if sanity == []:
                scores = [v*0.1 for v in scores]
            scores = [v for v in scores if 0 <= v <= 1]
        
        scores = [x for x in scores if min_val <= x <= max_val] # removing outliers this way
        norm = [(x - min_val) / (max_val - min_val) for x in scores]
        groups.append(norm)
    
    # Perform one-way ANOVA
    f_statistic, p_value = f_oneway(*groups)
    return f_statistic, p_value

#====================================================================
# PAPER MAIN TABLES 2 (tone and grammar)
#====================================================================


TASK_TYPES = []
for i, k in enumerate(QUESTIONS.keys()):
    TASK_TYPES.append(QUESTIONS[k]["question_type"])
TASK_TYPES = list(set(TASK_TYPES))


ALL_TASKTYPE_DF_BY_MODEL = {} # TO IMPORT =================================================
for presented_model in ALL_MODELS:
    all_tasktype_df = {}
    for tt in TASK_TYPES:
        downstream_score_df = []
        for annfilename in os.listdir(FINAL_ANNOTATOR_DIR):
            annotator_filepath = os.path.join(FINAL_ANNOTATOR_DIR, annfilename)
            with open(annotator_filepath, "r") as  ao:
                annotator_object = json.load(ao)
            
            downstream_scores = {}
            for question_key, values in annotator_object["answers"].items():
                if values["answer"] == None or QUESTIONS[question_key]["question_type"] != tt:
                    continue
                for qid, score_s in values["downstream_scores"][presented_model].items():
                    postfix = QUESTIONS[question_key]["metrics"][0] # first metric entry is by convention default
                    matching_keys = [k for k in score_s if k.endswith(f"{postfix}_score")]
                    if not matching_keys:
                        raise ValueError(f"No key ends with '{postfix}' for qid {qid}")
                    correct_key = matching_keys[0]
                    downstream_scores[f"{qid}-sentence_type"] = values["sentence_type"]
                    downstream_scores[f"{qid}-sentence_length"] = values["sentence_length"]
                    downstream_scores[f"{qid}-sentence_grammaticality"] = values["grammaticality"]
                    downstream_scores[correct_key] = score_s[correct_key]

            downstream_score_df.append({
                "id" : annotator_object["id"],
                **downstream_scores,
            })
        all_tasktype_df[tt] = pd.DataFrame(downstream_score_df)
    ALL_TASKTYPE_DF_BY_MODEL[presented_model] = all_tasktype_df

#====================================================================
# visualization functions
#====================================================================

def format_default(val):
    if isinstance(val, float):
        return f"{round(val, 2)}"
    return str(val)

def format_to_percentage(val):
    if isinstance(val, float):
        return f"{round(val * 100, 1)}%"
    return str(val)

def df_to_latex(df, format="default"):
    formatted_df = df.copy()
    for col in formatted_df.columns:
        if format == "default":
            formatfunc = format_default
        elif(format == "percentage"):
            formatfunc = format_to_percentage
        formatted_df[col] = formatted_df[col].apply(formatfunc)
    
    results_str = formatted_df.to_latex(index=False, escape=True)
    
    # patch correction:
    results_str = results_str.replace("70b", "70B").replace("72b", "72B").replace("qwen25", "Qwen25")
    results_str = results_str.replace(r"employment\_sector", "industry").replace(r"nationality\_group", "nationality")
    
    return results_str

def df_to_latex_one_column(df, caption, format="default"):
    return r"""\begin{table*}[t]
\resizebox{\textwidth}{!}{%
""" + df_to_latex(df, format) + rf"""}}
\caption{{{caption.replace("_",r"\_")}}}
\label{{tab:{caption}}}
\end{{table*}}
"""

#====================================================================
# ARCHIVED (too scared to delete lol)
#====================================================================

# def pairwise_p_values_with_higher(data_dict, min_val, max_val, clip_llm_judge=False):
#     results = {}
#     for (key1, values1), (key2, values2) in combinations(data_dict.items(), 2):
#         if clip_llm_judge:
#             sanity1 = [v for v in values1 if 0 <= v <= 1]
#             sanity2 = [v for v in values2 if 0 <= v <= 1]
#             # sanity check because some of them are in the range 1-10. if so, scale down
#             if sanity1 == [] or sanity2 == []:
#                 values1 = [v*0.1 for v in values1]
#                 values2 = [v*0.1 for v in values2]
#             values1 = [v for v in values1 if 0 <= v <= 1]
#             values2 = [v for v in values2 if 0 <= v <= 1]
        
#         values1 = [(v - min_val) / (max_val - min_val) for v in values1]
#         values2 = [(v - min_val) / (max_val - min_val) for v in values2]
        
#         _, p = ttest_ind(values1, values2, equal_var=False)
        
#         mean1 = sum(values1) / len(values1)
#         mean2 = sum(values2) / len(values2)
#         higher = key1 if mean1 > mean2 else key2

#         results[(key1, key2)] = {
#             "p" : p,
#             "higher" : higher,
#         }
#     return results

# def df_with_hexcolor_to_latex(values_df, color_df):
#     from collections import OrderedDict

#     latex_rows = []
#     defined_colors = OrderedDict()
#     color_idx = 0

#     for i in range(values_df.shape[0]):
#         row = []
#         for j in range(values_df.shape[1]):
#             val = values_df.iat[i, j]
#             hex_color = color_df.iat[i, j]
#             if isinstance(hex_color, str) and hex_color.startswith("#"):
#                 if hex_color not in defined_colors:
#                     defined_colors[hex_color] = f"customcolor{color_idx}"
#                     color_idx += 1
#                 colorname = defined_colors[hex_color]
#                 cell = f"\\cellcolor{{{colorname}!30}}{val}"
#             else:
#                 cell = str(val)
#             row.append(cell)
#         latex_rows.append(" & ".join(row) + r" \\")
    
#     col_headers = " & ".join(values_df.columns) + r" \\"
#     header = r"\begin{tabular}{" + "l" * values_df.shape[1] + "}\n\\hline\n" + col_headers + "\n\\hline"
#     body = "\n".join(latex_rows)
#     footer = "\n\\hline\n\\end{tabular}"

#     color_defs = "\n".join(
#         [f"\\definecolor{{{name}}}{{HTML}}{{{hex[1:]}}}" for hex, name in defined_colors.items()]
#     )

#     return color_defs + "\n\n" + header + "\n" + body + footer