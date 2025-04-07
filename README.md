# prompt-fairness
if you have any questions please don't hesitate to reach me on Slack

## Environment set up
tested ok on `Python 3.12.7`

clone and install the customized `lm-eval-harness` in the `prompt-fairness` branch
```shell
git clone https://github.com/rayendito/lm-evaluation-harness
cd lm-evaluation-harness
git checkout prompt-fairness
pip install -e .[vllm]
```
that should cover everything, if there's any error like `module <module> not found` installing using `pip install <module>` should fix it right away, and please let me know and ill update this readme

## Running
### Environment variables
make an `.env` file and add your hf token (for accessing gated models). example of the correct `.env` is in the `.env.example` file
```
HF_TOKEN=
```

### Running variables
in the `inference_run.sh` file, change these variables as needed (these are just examples)
```bash
RUN_NAME='llama32_1B_Inst' # convention : same as the model name but alphanumeric + underscores
MODEL_NAME='meta-llama/Llama-3.2-1B-Instruct' # model path to huggingface
MODEL_PROVIDER='hf' # this can be huggingface (`hf`) or `vllm` for vllm (see notes)
TENSOR_PARALLEL=false # whether to split models across GPUs (not recommended if model fits in one GPU, in this case use data parallel instead)
DATA_PARALLEL_SIZE=2 # how many model duplicates for data parallelism, usually same as the number of available GPUs
```
and some directory set up
```bash
# model storing cache directories
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub

# where to store the results (heads up: result file can be as big as 200MB)
export SAMPLES_RESULTS_DIR=/workspace/lm_harness_runs/

export TOKENIZERS_PARALLELISM=false
```
### **running** running
```
./inference_run.sh
```
and hope nothing breaks

## Docs
### how tasks and their datasets are indexed
every task is defined here. convention is {question index}_{dataset index} see this [google docs](https://docs.google.com/spreadsheets/d/1-FJ2rOOUsiUaThqiuXb-DE8ZweyZsvhdFYH7kN44w3g/edit?gid=0#gid=0)
```
NO_DP="5_0 6_0 7_0 8_0 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0 17_0 18_0 19_0"
OK_DP="0_0 1_0 4_0 20_0 26_0 26_1 2_0 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 2_10 2_11 2_12 2_13 2_14 3_0 21_0 22_0 23_0 24_0 25_0 25_1 27_0 27_1 27_2 27_3 27_4 27_5 28_0"
```
- `NO_DP` means tasks that we cannot do data parallelism on. these are the ones we don't have any benchmarks to run on. so the workaround is to create a "benchmark" with only one instance. therefore we cant do DP on this because it's only one (we can technically, do DP with combining answers from every annotator, but i think they wont take too long anyways, like only <100 instances per question @ one per annotator)
- `OK_DP` are the ok ones for data parallelism

## Notes
- initially i wanted to implement the inference with vllm, but it kept hanging when using Ray for the multiprocess handling? idk it seems like the current implementation of lm harness doesnt really use the [correct support for LLMs with Ray](https://docs.ray.io/en/latest/data/working-with-llms.html)
- although i did try it, using a very small script, it worked and i could've modified the lm-harness code but i remembered vllm may not support all models anyways, so yeah huggingface it is
- **tl;dr** it's not recommended to use vLLM, but it should work for 1 GPU though
