#!/bin/bash
#SBATCH --job-name=post-gpt_4o
#SBATCH --output="%x-%j.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="Ryandito.Diandaru@mbzuai.ac.ae"
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos

export HF_HOME=/l/users/ryandito.diandaru/huggingface
export HF_HUB_CACHE=/l/users/ryandito.diandaru/huggingface/hub
export SAMPLES_RESULTS_DIR=/l/users/ryandito.diandaru/lm_harness_runs/

# python postprocess.py --run_name Qwen25_1dot5B_Inst --inference_type "separate" --recalculate
# sleep 2

# python postprocess.py --run_name Qwen25_7B_Inst --inference_type "separate" --recalculate
# sleep 2

# python postprocess.py --run_name llama32_1B_Inst --inference_type "separate" --recalculate
# sleep 2

# python postprocess.py --run_name llama31_8B_Inst --inference_type "separate" --recalculate
# # sleep 2


# python postprocess.py --run_name llama31_70b_Inst --inference_type "separate" --recalculate
# sleep 2
# python postprocess.py --run_name qwen25_72b_Inst --inference_type "separate" --recalculate

python postprocess.py --run_name gpt_4o --inference_type "api" --recalculate
