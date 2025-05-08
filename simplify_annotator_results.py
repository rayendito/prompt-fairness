import os, json
from src.utils.annotator_manager_utils import get_empty_question_profile, update_question_profile

PASSED_ANNOTATOR_DIRS = "annotation_artefacts/scored_passed_annotators"
FINAL_ANNOTATOR_OBJECTS = "annotation_artefacts/final_annotator_objects"
CHATGPT_GRAMMAR_CLASSIFIER_RES = "/l/users/ryandito.diandaru/lm_harness_postprocess/annotator_grammar_inferoai/ans_classifier_run-chatgpt-4o-latest/history-ans_classifier_run-chatgpt-4o-latest.json"

with open(CHATGPT_GRAMMAR_CLASSIFIER_RES, "r") as clsrf:
    chatgpt_grammar_results = json.load(clsrf)

os.makedirs(FINAL_ANNOTATOR_OBJECTS, exist_ok=True)

for anf in os.listdir(PASSED_ANNOTATOR_DIRS):
    new_ann_obj = {}
    simplified_profile = get_empty_question_profile()
    with open(os.path.join(PASSED_ANNOTATOR_DIRS, anf), "r") as f:
        annotator_object = json.load(f)
        update_question_profile(simplified_profile,
                                annotator_object["profile"],
                                filler="population",
                                exclude_highschool=False)
        simplified_profile = {
            category: next((k for k, v in values.items() if v == [1]), None)
            for category, values in simplified_profile.items()
        }

        new_answers = {}
        transferred_keys = ["answer", "sentence_type", "sentence_length"]
        for raw_question, answer_and_scores in annotator_object["answers"].items():
            new_answer_and_scores = {}
            for k, v in answer_and_scores.items():
                if k in transferred_keys:
                    new_answer_and_scores[k] = v
            new_answers[raw_question] = new_answer_and_scores
        
        grammar_classif_results = chatgpt_grammar_results[f"annotator-{annotator_object["id"]}-chatgpt_4o_latest"]        
        assert len(grammar_classif_results["messages"]) == len(annotator_object["answers"])
        for i, ao_key in enumerate(annotator_object["answers"].keys()):
            if "grammaticality" in grammar_classif_results["messages"][i]["aggregations"]: # the "<<<SKIP_THIS_PROMPT>>>"
                grammaticality = grammar_classif_results["messages"][i]["aggregations"]["grammaticality"]
                new_answers[ao_key]["grammaticality"] = grammaticality

        new_ann_obj = annotator_object.copy()
        new_ann_obj["profile"] = simplified_profile
        new_ann_obj["answers"] = new_answers
        
        del new_ann_obj["max_llm_detection_score"]
        del new_ann_obj["avg_llm_detection_score"]
        
        with open(f"{FINAL_ANNOTATOR_OBJECTS}/annotator{annotator_object["id"]}.json", "w") as nf:
            json.dump(new_ann_obj, nf, indent=4)
        
