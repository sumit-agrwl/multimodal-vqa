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
    test_subset_dir = os.path.join(subset_dir,'test_balanced_questions.csv')

    train_subset_questions = []
    train_subset_answers = []
    val_subset_questions = []
    test_subset_questions = []

    # TODO: open json files for scene graphs



    # TODO initialize model

    # TODO: create training loop

    # TODO: call train loop w/ validation set every set number of batches