import os, json, argparse
from dotenv import load_dotenv
from src.openai_runner.run_for_tasks import run_experiment

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="run name")
parser.add_argument("--question_document_index", type=str, required=True, help="question index and document index")
parser.add_argument("--model_provider", type=str, default="openai", help="who provides the model, defaults to openai")
parser.add_argument("--model_name", type=str, default="gpt-4o", help="default: gpt-4o")
parser.add_argument("--results_dir", type=str, required=True, help="Directory to store results")
parser.add_argument("--reference_dir", type=str, required=True, help="Directory to get prompts from (default: /results)")
args = parser.parse_args()

ref_experiment_name = args.reference_dir.split("/")[-1]
ref_experiment_fulldir = f"{ref_experiment_name}-question{args.question_document_index}"
ref_experiment_samplefile = f"samples-{ref_experiment_fulldir}-question_{args.question_document_index}.json"

with open(os.path.join(args.reference_dir, ref_experiment_fulldir, ref_experiment_samplefile)) as rf:
    sample_results = json.load(rf)

run_dict = {}
for annotator, prompts in sample_results.items():
    final_prompts = []
    for p in prompts:
        final_prompts.append(p['arguments'][0][0])
    run_dict[annotator] = final_prompts

final_run_name = f"{args.run_name}-question{args.question_document_index}"
run_experiment(args.model_provider, args.model_name, final_run_name, run_dict, results_dir=args.results_dir)

# model_name = "gpt-4o"
# run_name = "redir_test"
# run_dict = {
#     "runone" : [
#         "what is the capital of Germany",
#         "what is the capital of Italy",
#     ],
#     "runtwo" : [
#         "what is the capital of Indonesia",
#         "what is the capital of UAE",
#     ],
# }

# run_experiment(model_name, run_name, run_dict, results_dir="tezt/")