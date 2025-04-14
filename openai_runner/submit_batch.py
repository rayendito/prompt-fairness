import sys, os, json, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../inferinter")))

from ModelManager import ModelManager # type: ignore

def run_experiment(model, run_name, expriment_dict):
    """
        experiment dict:
            key : what to name the exp
            value : prompts 
    """
    batches = [{
        "config" : {
            "tag" : exp_name,
            "model_provider" : "openai",
            "model_name" : model,
            "in_one_context" : False,
        },
        "input_prompts" : prompts
    } for exp_name, prompts in expriment_dict.items()]

    grid = {
        "grid_info" : {
            "project_name" : __package__,
            "run_name" : run_name,
        },
        "batches" : batches
    }
    
    grid_path = f"{__package__}/{grid['grid_info']['run_name']}/submitted_grid.json"
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    with open(grid_path, "w") as gg:
        json.dump(grid, gg, indent=4)

    modman = ModelManager(grid)
    modman.run_grid(grid)