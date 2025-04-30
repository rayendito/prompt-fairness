import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="result json file")
parser.add_argument("--recalculate", action="store_true", help="if set, do not recalculate metrics if .csv for question already exists")
args = parser.parse_args()