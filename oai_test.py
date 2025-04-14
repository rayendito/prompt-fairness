import os, json, argparse
from openai_runner.submit_batch import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--question_document_index", type=str, required=True, help="question index and document index")
parser.add_argument("--openai_model", type=str, default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
parser.add_argument("--reference_dir", type=str, default="dummy_lm_harness_runs/llama32_1B_Inst", help="Directory to get prompts from (default: /results)")
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

run_name = f"{args.openai_model.replace("-", "_")}-question{args.question_document_index}"
run_experiment(args.openai_model, run_name, run_dict)

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

