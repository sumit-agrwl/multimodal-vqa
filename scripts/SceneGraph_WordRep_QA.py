import sys
import os
import numpy as np

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


from transformers import pipeline
from transformers import AutoTokenizer, BertModel

import json
import csv
import torch

# TODO define model
# 1. scene graph representation
# 2. word representation
# 3. concatenation
# 4. one FC layer
# 5. softmax layer
# 6. with loss function
# TODO xavier initialize FC on init

if __name__ == "__main__":
    print("Starting Code for Question 02")

    # TODO: open json files for questions and answers
    currDir = os.getcwd()
    path_parent = os.chdir(os.path.dirname(os.getcwd()))
    print(os.getcwd())

    subset_dir = os.path.join(os.getcwd(),'gqa','project_subset')
    train_subset_dir = os.path.join(subset_dir,'train_balanced_questions.csv')
    val_subset_dir = os.path.join(subset_dir,'val_balanced_questions.csv')

    train_subset_questions = []
    train_subset_answers = []
    val_subset_questions = []
    val_subset_answers = []


    with open(train_subset_dir) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0 and str(row[8]) in answer_distribution_top.keys():
                train_subset_questions.append(str(row[4])) # question
                train_subset_answers.append(str(row[8])) # answer
            line_count += 1

    with open(val_subset_dir) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0 and str(row[8]) in answer_distribution_top.keys():
                val_subset_questions.append(str(row[4])) # question
                val_subset_answers.append(str(row[8])) # answer
            line_count += 1

    # TODO: open json files for scene graphs
    scenegraphs_subset_dir = os.path.join(os.getcwd(),'gqa','sceneGraphs')
    train_scenegraphs_dir = os.path.join(scenegraphs_subset_dir,'train_sceneGraphs.json')
    val_scenegraphs_dir = os.path.join(scenegraphs_subset_dir,'val_sceneGraphs.json')

    with open(train_scenegraphs_dir) as json_file:
        scenegraph_train_raw_data = json.load(json_file)

    with open(val_scenegraphs_dir) as json_file:
        scenegraph_val_raw_data = json.load(json_file)




    # TODO initialize model

    # TODO: create training loop

    # TODO: call train loop w/ validation set every set number of batches

    print("Done.")