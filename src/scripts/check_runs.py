import sys, os, argparse
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../inferinter")))
from agents import CustomSyncAgent # type: ignore

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="run name")
parser.add_argument("--results_dir", type=str, required=True, help="results_dir")
parser.add_argument("--through_api", action="store_true", help="Call through API")
args = parser.parse_args()

runs = """
0_0
1_0
4_0
20_0
26_0
26_1
2_0
2_1
2_2
2_3
2_4
2_5
2_6
2_7
2_8
2_9
2_10
2_11
2_12
2_13
2_14
3_0
5_0
6_0
7_0
8_0
9_0
10_0
11_0
12_0
13_0
14_0
15_0
16_0
17_0
18_0
19_0
21_0
22_0
23_0
24_0
25_0
25_1
27_0
27_1
27_2
27_3
27_4
27_5
28_0
""".strip().split()

# throwaway api inference agent
csa = CustomSyncAgent(model_provider = "openrouter", model_name="anymodel")

print("===========================================")
for r in runs:
    respath = os.path.join(args.results_dir, f"{args.run_name}", f"{args.run_name}-question{r}")
    if args.through_api:
        history_file = os.path.join(respath, f"history-{args.run_name}-question{r}.json")
        if not(os.path.exists(history_file)):
            print(f"{args.run_name} question number {r} is not even found")
        else:
            n_unanswered = csa.how_many_unanswered(history_file, csa.default_is_messages_done_fn)
            if n_unanswered > 0:
                print(f"{args.run_name} question number {r} n_unanswered: {n_unanswered}")
    else:
        respath_test = os.path.join(respath, f"results-{args.run_name}-question{r}-question_{r}.json")
        if not(os.path.exists(respath_test)):
            print(f"{args.run_name} question {r} not finished yet")
print("===========================================")