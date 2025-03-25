#! /bin/bash

# identity
#SBATCH --job-name=visualization
#SBATCH --output="%x-%j.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Ryandito.Diandaru@mbzuai.ac.ae

# compute configuration
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos
#SBATCH --time=12:00:00

model_name="llama32_1B_Inst"
python temp_consolidate_and_viz.py --run_name "${model_name}-question0_0" --question_document_index 0_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question1_0" --question_document_index 1_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question4_0" --question_document_index 4_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question20_0" --question_document_index 20_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_0" --question_document_index 2_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_1" --question_document_index 2_1
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_2" --question_document_index 2_2
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_3" --question_document_index 2_3
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_4" --question_document_index 2_4
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_5" --question_document_index 2_5
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_6" --question_document_index 2_6
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_7" --question_document_index 2_7
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_8" --question_document_index 2_8
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_9" --question_document_index 2_9
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_10" --question_document_index 2_10
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_11" --question_document_index 2_11
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_12" --question_document_index 2_12
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_13" --question_document_index 2_13
python temp_consolidate_and_viz.py --run_name "${model_name}-question2_14" --question_document_index 2_14
python temp_consolidate_and_viz.py --run_name "${model_name}-question3_0" --question_document_index 3_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question5_0" --question_document_index 5_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question6_0" --question_document_index 6_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question7_0" --question_document_index 7_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question8_0" --question_document_index 8_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question9_0" --question_document_index 9_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question13_0" --question_document_index 13_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question14_0" --question_document_index 14_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question15_0" --question_document_index 15_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question16_0" --question_document_index 16_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question17_0" --question_document_index 17_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question18_0" --question_document_index 18_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question19_0" --question_document_index 19_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question21_0" --question_document_index 21_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question22_0" --question_document_index 22_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question25_0" --question_document_index 25_0
python temp_consolidate_and_viz.py --run_name "${model_name}-question25_1" --question_document_index 25_1