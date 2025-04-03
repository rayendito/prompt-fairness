RUN_NAME='llama32_1B_Inst'
MODEL_NAME='meta-llama/Llama-3.2-1B-Instruct'
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=2

export LD_LIBRARY_PATH=$HOME/.config/vllm/nccl/cu12

echo "Starting NVIDIA SMI..."
nvidia-smi

# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR

export HF_HOME=/l/users/ryandito.diandaru/huggingface
export HF_HUB_CACHE=/l/users/ryandito.diandaru/huggingface/hub
export TRANSFORMERS_CACHE=/l/users/ryandito.diandaru/huggingface/transformers

accelerate launch \
  lmh_test.py \
  --run_name "$RUN_NAME-question2_5" \
  --model_string "$MODEL_NAME" \
  --question_document_index 2_5 \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --data_parallel_size "$DATA_PARALLEL_SIZE"
sleep 10