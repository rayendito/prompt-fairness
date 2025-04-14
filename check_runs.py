import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="run name")
parser.add_argument("--results_dir", type=str, required=True, help="results_dir")
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

for r in runs:
    respath = os.path.join(args.results_dir, f"{args.run_name}", f"{args.run_name}-question{r}")
    respath_test = os.path.join(respath, f"results-{args.run_name}-question{r}-question_{r}.json")
    if not(os.path.exists(respath_test)):
        print(f"{args.run_name} question {r} not finished yet")