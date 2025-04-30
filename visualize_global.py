import os, json, argparse
import pandas as pd
from src.classes.AnnotatorManager import AnnotatorManager
from src.utils.viz_utils import plot_histogram_by_category, plot_grouped_bar_chart
import matplotlib.pyplot as plt
import numpy as np

SCORED_ANNOTATOR_DIR = "annotation_artefacts/scored_passed_annotators"

# making a df that has info: sum sentence length, freq count for each of the annotators
df_ptype_freqtable = []
for annfilename in os.listdir(SCORED_ANNOTATOR_DIR):
    annotator_filepath = os.path.join(SCORED_ANNOTATOR_DIR, annfilename)
    with open(annotator_filepath, "r") as  ao:
        annotator_object = json.load(ao)
    
    prompt_lengths = []
    prompt_type_freq_table = {
        "imperative" : 0,
        "interrogative" : 0,
        "declarative" : 0,
    }
    for v in annotator_object["answers"].values():
        if v["answer"]:
            prompt_lengths.append(v["sentence_length"])
            prompt_type_freq_table[v["sentence_type"]] += 1

    df_ptype_freqtable.append({
        "id" : annotator_object["id"],
        **annotator_object["profile"],
        "sum_sentence_lengths" : sum(prompt_lengths),
        **prompt_type_freq_table
    })
df_ptype_freqtable = sorted(df_ptype_freqtable, key=lambda x: x["id"])
df_ptype_freqtable = pd.DataFrame(df_ptype_freqtable)

# making a dataset
aggregates = {}
for agg_attributes in ["sum_sentence_lengths", "interrogative", "declarative", "imperative"]:
    if agg_attributes == "sum_sentence_lengths":
        method = "list"
    else:
        method = "sum"
    aggregates[agg_attributes] = AnnotatorManager.get_profile_of_question(df_ptype_freqtable, filler=agg_attributes, method=method)

# print(json.dumps(aggregates, indent=4))

# visualizing
N = len(aggregates["sum_sentence_lengths"])
fig, axes = plt.subplots(N, 2, figsize=(12, 4 * N))

LABELSIZE=15
# Column labels
column_labels = ['Sentence Length Histogram', 'Prompt Sentence Type']
for col, label in enumerate(column_labels):
    axes[0, col].set_title(label, fontsize=LABELSIZE)
# Row labels
row_labels = [k for k in aggregates["sum_sentence_lengths"].keys()]
for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=LABELSIZE, rotation=90)

nrows, ncols = axes.shape
for i, j in np.ndindex(nrows, ncols):
    ax = axes[i, j]
    demo_graph = list(aggregates["sum_sentence_lengths"].keys())[i]
    if j == 0:
        data = aggregates['sum_sentence_lengths'][demo_graph]
        plot_histogram_by_category(data, ax)
    else:
        data = {}
        for d in aggregates["interrogative"][demo_graph].keys():
            data[d] = {}
            for senttype in ["imperative", "interrogative", "declarative"]:
                data[d][senttype] = aggregates[senttype][demo_graph][d]
        plot_grouped_bar_chart(data, ax)
    
    # # debug
    # if(i == 0):
    #     print(json.dumps(data, indent=4))

fig.savefig(f"visualization_results/prompt_chars.png", bbox_inches='tight')
