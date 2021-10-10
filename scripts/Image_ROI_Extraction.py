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
import h5py

import torch.nn as nn
import torch.optim


if __name__ == "__main__":
    # TODO open training questions dataset
    print("Starting Code to Acquire Image and ROI Embeddings")

    # open json files for questions and answers
    currDir = os.getcwd()
    path_parent = os.chdir(os.path.dirname(os.getcwd()))
    print(os.getcwd())

    subset_dir = os.path.join(os.getcwd(),'gqa','project_subset')
    train_subset_dir = os.path.join(subset_dir,'train_balanced_questions.csv')
    val_subset_dir = os.path.join(subset_dir,'val_balanced_questions.csv')

    train_subset_imageID = []
    train_subset_questions = []
    train_subset_answers = []
    val_subset_imageID = []
    val_subset_questions = []
    val_subset_answers = []


    with open(train_subset_dir) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                train_subset_imageID.append(str(row[5])) # image ID
                train_subset_questions.append(str(row[4])) # question
                train_subset_answers.append(str(row[8])) # answer
            line_count += 1

    # TODO open validation question dataset
    with open(val_subset_dir) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                val_subset_imageID.append(str(row[5])) # image ID
                val_subset_questions.append(str(row[4])) # question
                val_subset_answers.append(str(row[8])) # answer
            line_count += 1
    
    # TODO open spatial features h5 file
    image_features_subset_dir = os.path.join(os.getcwd(), 'gqa', 'allimages', 'spatial')
    num_image_files = 16

    for file_idx in range(num_image_files):
        image_features_filename = "gqa_spatial_" + str(file_idx) + ".h5"
        image_features_filepath = os.path.join(image_features_subset_dir, image_features_filename)
        h5 = h5py.File(image_features_filepath, 'r')

        h5.close()




    # TODO store embeddings that are tied to images to .pkl file