import os, json
from src.utils.annotator_manager_utils import get_empty_question_profile, update_question_profile

PASSED_ANNOTATOR_DIRS = "annotation_artefacts/annotation_filtering/passed_annotator_objects"
FINAL_ANNOTATOR_OBJECTS = "annotation_artefacts/final_annotator_objects"
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
        transferred_keys = ["answer"]
        for raw_question, answer_and_scores in annotator_object["answers"].items():
            new_answer_and_scores = {}
            for k, v in answer_and_scores.items():
                if k in transferred_keys:
                    new_answer_and_scores[k] = v
            new_answers[raw_question] = new_answer_and_scores

        new_ann_obj = annotator_object.copy()
        new_ann_obj["profile"] = simplified_profile
        new_ann_obj["answers"] = new_answers
        
        del new_ann_obj["max_llm_detection_score"]
        del new_ann_obj["avg_llm_detection_score"]
        
        with open(f"{FINAL_ANNOTATOR_OBJECTS}/annotator{annotator_object["id"]}.json", "w") as nf:
            json.dump(new_ann_obj, nf, indent=4)
        break
