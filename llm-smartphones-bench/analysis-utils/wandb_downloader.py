"""
Downloads the results of LM Evaluation Harness benchmark runs from a WandB project and saves them to a local directory.
"""

import wandb
import os
import json

save_dir = "./llama32_ifeval_eval"
model_name = "Llama32_3B"
benchmark_name = "IFEval"
wandb_project_name = "...replace_with_your_project_name..."
wandb_team_name = "...replace_with_your_team_name..."

os.makedirs(save_dir, exist_ok=True)
api = wandb.Api()

runs = api.runs(f"{wandb_team_name}/{wandb_project_name}")

v1s = []
v2s = []

for run in runs:
    if run.state != "finished":
        continue
    run_name = run.name
    run_summary = run.summary._json_dict
    name_parts = run_name.split(" ")

    if len(name_parts) == 2:
        v1s.append((name_parts[0], run_summary))
    if len(name_parts) == 3:
        v2s.append((name_parts[0], run_summary))

for run in [*v1s, *v2s]:
    quant, summary = run
    filename = os.path.join(save_dir, f"{model_name}-{benchmark_name}-{quant}.json")
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)
