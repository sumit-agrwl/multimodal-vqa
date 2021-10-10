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
import pickle

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
    
    # TODO open spatial features json file
    # image_features_json_dir = os.path.join(os.getcwd(), 'gqa', 'allimages', 'spatial','gqa_spatial_info.json')
    # with open(image_features_json_dir) as json_file:
    #     image_features_raw_data = json.load(json_file)

    # TODO open spatial features h5 file
    # image_features_subset_dir = os.path.join(os.getcwd(), 'gqa', 'allimages', 'spatial')
    # num_image_files = 16

    # spatial_features_training_dict = {}

    # for imageID in train_subset_imageID:
    #     if imageID not in spatial_features_training_dict.keys():
    #         idx = image_features_raw_data[imageID]['idx']
    #         file_num = image_features_raw_data[imageID]['file']
    #         image_features_filename = "gqa_spatial_" + str(file_num) + ".h5"
    #         image_features_filepath = os.path.join(image_features_subset_dir, image_features_filename)
    #         h5 = h5py.File(image_features_filepath, 'r')
    #         spatial_features_training_dict[imageID] = torch.tensor(h5['features'][idx])
    #         h5.close()
    #         print("Training: Length of spatial_features_dict: " + str(len(spatial_features_training_dict)))

    

    # image_embedding_train_resnet101_title = "ImageEmbedding_Resnet101_Train.pkl"
    # with open(image_embedding_train_resnet101_title, "wb") as outputfile:
    #     pickle.dump(spatial_features_training_dict, outputfile)  
    # del spatial_features_training_dict

    # spatial_features_val_dict = {}

    # for imageID in val_subset_imageID:
    #     if imageID not in spatial_features_val_dict.keys():
    #         idx = image_features_raw_data[imageID]['idx']
    #         file_num = image_features_raw_data[imageID]['file']
    #         image_features_filename = "gqa_spatial_" + str(file_num) + ".h5"
    #         image_features_filepath = os.path.join(image_features_subset_dir, image_features_filename)
    #         h5 = h5py.File(image_features_filepath, 'r')
    #         spatial_features_val_dict[imageID] = torch.tensor(h5['features'][idx])
    #         h5.close()
    #         print()    
    #         print("Validation: Length of spatial_features_dict: " + str(len(spatial_features_val_dict)))


    # image_embedding_val_resnet101_title = "ImageEmbedding_Resnet101_Val.pkl"
    # with open(image_embedding_val_resnet101_title, "wb") as outputfile:
    #     pickle.dump(spatial_features_val_dict, outputfile)  

    # del spatial_features_val_dict

    objects_features_json_dir = os.path.join(os.getcwd(), 'gqa', 'allimages', 'objects','gqa_objects_info.json')
    with open(objects_features_json_dir) as json_file:
        objects_features_raw_data = json.load(json_file)


    # TODO store embeddings that are tied to images to .pkl file
    objects_features_subset_dir = os.path.join(os.getcwd(), 'gqa', 'allimages', 'objects')
    num_image_files = 16

    objects_features_training_dict = {}

    for imageID in train_subset_imageID:
        if imageID not in objects_features_training_dict.keys():
            idx = objects_features_raw_data[imageID]['idx']
            file_num = objects_features_raw_data[imageID]['file']
            objects_features_filename = "gqa_objects_" + str(file_num) + ".h5"
            objects_features_filepath = os.path.join(objects_features_subset_dir, objects_features_filename)
            h5 = h5py.File(objects_features_filepath, 'r')
            objects_features_training_dict[imageID] = torch.tensor(h5['features'][idx])
            h5.close()
            print("Training: Length of objects_features_dict: " + str(len(objects_features_training_dict)))



    objects_embedding_train_resnet101_title = "ImageEmbedding_Resnet101_Train.pkl"
    with open(objects_embedding_train_resnet101_title, "wb") as outputfile:
        pickle.dump(objects_features_training_dict, outputfile)  
    del objects_features_training_dict

    objects_features_val_dict = {}

    for imageID in val_subset_imageID:
        if imageID not in objects_features_val_dict.keys():
            idx = objects_features_raw_data[imageID]['idx']
            file_num = objects_features_raw_data[imageID]['file']
            objects_features_filename = "gqa_objects_" + str(file_num) + ".h5"
            objects_features_filepath = os.path.join(objects_features_subset_dir, objects_features_filename)
            h5 = h5py.File(objects_features_filepath, 'r')
            objects_features_val_dict[imageID] = torch.tensor(h5['features'][idx])
            h5.close()
            print()    
            print("Validation: Length of objects_features_dict: " + str(len(objects_features_val_dict)))


    objects_embedding_val_resnet101_title = "ImageEmbedding_Resnet101_Val.pkl"
    with open(objects_embedding_val_resnet101_title, "wb") as outputfile:
        pickle.dump(objects_features_val_dict, outputfile)  

    del objects_features_val_dict

    print('Done')