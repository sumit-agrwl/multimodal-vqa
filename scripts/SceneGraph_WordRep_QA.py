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

import torch.nn as nn
import torch.optim

# TODO define model
# 1. scene graph representation
# 2. word representation
# 3. concatenation
# 4. one FC layer
# 5. softmax layer
# 6. with loss function
# TODO xavier initialize FC on init

class scenegraph_question_model(nn.Module):
    def __init__ (self, sceneGraphSize=1024, answerVocabSize = 1800, model_name, device_name):
        super().__init__()
        self.model_name = model_name
        self._padding = True
        self._truncation = True
        self.max_length = 256

        # TODO fully connect layer for scene graphs
        self._sceneGraph_InputSize = sceneGraphSize # look at tokensize of tokenizer for help
        self._sceneGraph_OutputSize = 1024
        self.sceneGraphEmbedding = nn.Embedding(self._sceneGraph_InputSize, self._sceneGraph_OutputSize)

        # TODO pretrained BERT model for feature representation ( 1 x 768)
        self.initializePipeline()

        # TODO nonlinear activation after scene graph fully connect layer
        self.tanHActivation = nn.Tanh()

        # TODO fully connected layer output
        self._concatenatedSize = 2048
        self._answerVocabSize = answerVocabSize
        self.fullyconnected1 = nn.Linear(self._concatenatedSize, self._answerVocabSize)

    def forward(self, sceneGraphVect, questionStrVar):
        # TODO encode questionVar
        questionTokenized = self.encodeInput(questionStrVar)
        questionEmbedding = self.feature_extraction(questionTokenized)

        # TODO encode sceneGraphVar
        sceneGraphEmbedding = self.sceneGraphEmbedding(sceneGraphVect)

        # TODO concatenate the two representations
        concateVect = torch.cat((sceneGraphEmbedding, questionEmbedding), 1)

        # TODO pass concatenation through fully connected layer
        output = self.fullyconnected1(concateVect)


    def initializePipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device_name)

    def tokenizeInput(self, inputBatch):
        return self.tokenizer(inputBatch, padding=self._padding, truncation=self._truncation, max_length=self._max_length, return_tensors="pt").to(self.device_name)

    def encodeInput(self,tokenizedInput):
        return self.model(**tokenizedInput)

    def feature_extraction(self,inputBatch):
        tokenizedInput = self.tokenizeInput(inputBatch)
        return self.encodeInput(tokenizedInput)



if __name__ == "__main__":
    print("Starting Code for Question 02")

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

    with open(val_subset_dir) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                val_subset_imageID.append(str(row[5])) # image ID
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

    # TODO process through scene graphs to only get scene graphs that matter out
    scenegraph_train_data = {}
    scenegraph_val_data = {}
    for imageID in train_subset_imageID:
        sceneGraph = scenegraph_train_raw_data[imageID]
        tempStr = ""
        for objectVar in sceneGraph['objects'].keys():
            tempStr += str(objectVar['name']) + " "
        scenegraph_train_data[imageID] = tempStr

    # TODO convert answer to one hot vectors


    # TODO initialize model
    sceneGraphSize = 1024
    model = scenegraph_question_model(sceneGraphSize)

    # TODO: create training loop
    # xavier initialize the model
    nn.init.xavier_uniform_(model.sceneGraphEmbedding.weight)
    nn.init.xavier_uniform_(model.fullyconnected1.weight)

    # create optimizer
    # separate optimizers for word embeddings and softmax layer
    momentum = 0.9
    weight_decay = 1e-4
    optimizer_word = torch.optim.SGD(model.sceneGraphEmbedding.parameters(), lr = 0.8, momentum = momentum, weight_decay = weight_decay)
    optimizer_softmax = torch.optim.SGD(model.fullyconnected1.parameters(), lr = 0.01, momentum = momentum, weight_decay = weight_decay)
    lossFunc = nn.CrossEntropyLoss()


    # TODO: call train loop w/ validation set every set number of batches
    num_epochs = 2
    totSizeReached = False
    totDataSize = len(train_subset_questions)
    offsetVal = 0
    batchSize = 32
    count = 0
    eval_time = 1000
    for epoch in range(num_epochs):
        print("Begin training for epoch: " + str(epoch+1))
        while (totSizeReached is False):
            questArr = []
            ansArr = []
            graphArr = []
            for idx in range(offsetVal, offsetVal+batchSize):
                if count >= totDataSize:
                    totSizeReached = True
                    break
                question = train_subset_questions[count]
                graph = train_sceneGraphs[count]
                answer = train_subset_answers[count]
                questArr.append(question)
                ansArr.append(answer)
                graphArr.append(graph)
                count += 1
            offsetVal = count
            output = model(graphArr, questArr)

            optimizer_word.zero_grad()
            optimizer_softmax.zero_grad()

            loss = lossFunc(output, ansArr)

            loss.backward()

            optimizer_word.step()
            optimizer_softmax.step()

    print("Done.")