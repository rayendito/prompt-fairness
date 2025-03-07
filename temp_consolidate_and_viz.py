from src import AnnotatorManager, Questions

questions = Questions().get_questions()
ann_manager = AnnotatorManager(questions, "passed_annotator_objects")