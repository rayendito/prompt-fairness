#!/bin/bash
#SBATCH --job-name=new-llama31_8B_Inst
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

# ./inference_run_legacy.sh Qwen25_1dot5B_Inst Qwen/Qwen2.5-1.5B-Instruct 64
# ./inference_run_legacy.sh llama32_1B_Inst meta-llama/Llama-3.2-1B-Instruct 64

# ./inference_run_legacy.sh Qwen25_7B_Inst Qwen/Qwen2.5-7B-Instruct 16
./inference_run_legacy.sh llama31_8B_Inst meta-llama/Llama-3.1-8B-Instruct 16